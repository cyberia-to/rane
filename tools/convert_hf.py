#!/usr/bin/env python3
"""Convert Qwen3-0.6B HuggingFace safetensors → ANE checkpoint format.

Usage:
    python3 convert_hf.py [--model Qwen/Qwen3-0.6B] [--output ane_qwen3_06b_dyn_ckpt.bin]

Downloads the model from HuggingFace (cached), converts weight layout,
writes the binary checkpoint that ./infer can load directly.
"""
import argparse
import struct
import numpy as np
import sys
import os

# Qwen3-0.6B architecture constants (must match models/qwen3_06b.h)
DIM = 1024
HIDDEN = 3072
HEADS = 16
KV_HEADS = 8
HD = 128
Q_DIM = HEADS * HD      # 2048
KV_DIM = KV_HEADS * HD  # 1024
SEQ = 256
NLAYERS = 28
VOCAB = 151936

# Derived weight sizes
WQ_SZ = Q_DIM * DIM
WK_SZ = KV_DIM * DIM
WV_SZ = KV_DIM * DIM
WO_SZ = DIM * Q_DIM
W1_SZ = HIDDEN * DIM
W2_SZ = DIM * HIDDEN
W3_SZ = HIDDEN * DIM


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-0.6B HF → ANE checkpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--output", default="ane_qwen3_06b_dyn_ckpt.bin", help="Output checkpoint path")
    args = parser.parse_args()

    try:
        from safetensors import safe_open
    except ImportError:
        print("pip install safetensors", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    # Download model
    print(f"Downloading {args.model}...")
    model_path = snapshot_download(args.model)
    print(f"Model cached at: {model_path}")

    # Find safetensors files
    st_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
    if not st_files:
        print("No .safetensors files found!", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(st_files)} safetensors file(s)")

    # Load all tensors (Qwen3 uses bfloat16 — need torch for bf16→f32)
    try:
        import torch
        use_torch = True
    except ImportError:
        use_torch = False

    tensors = {}
    for sf in st_files:
        path = os.path.join(model_path, sf)
        if use_torch:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key).float().numpy()
        else:
            # Fallback: read raw bf16 bytes, manual convert via struct
            with safe_open(path, framework="numpy", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    print(f"Loaded {len(tensors)} tensors")

    # Print tensor names for debugging
    for k, v in sorted(tensors.items()):
        print(f"  {k}: {v.shape} {v.dtype}")

    # Verify shapes
    # Qwen3 naming: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    #               model.layers.{i}.mlp.{gate,up,down}_proj.weight
    #               model.layers.{i}.input_layernorm.weight
    #               model.layers.{i}.post_attention_layernorm.weight
    #               model.norm.weight
    #               model.embed_tokens.weight

    def get(name):
        t = tensors[name].astype(np.float32)
        return t

    # Write checkpoint
    print(f"\nWriting checkpoint: {args.output}")
    with open(args.output, "wb") as f:
        # CkptHdr (must match config.h CkptHdr struct exactly)
        # struct CkptHdr {
        #   int magic, version, step, total_steps;
        #   int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
        #   float lr, loss;
        #   double cum_compile, cum_train, cum_wall;
        #   int cum_steps, cum_batches, adam_t;
        #   int kv_heads, head_dim, q_dim;
        # }
        header = struct.pack(
            "iiii"      # magic, version, step, total_steps
            "iiiiii"    # n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len
            "ff"        # lr, loss
            "ddd"       # cum_compile, cum_train, cum_wall
            "iii"       # cum_steps, cum_batches, adam_t
            "iii",      # kv_heads, head_dim, q_dim
            0x424C5A54, 5, 0, 0,          # magic, version=5 (QK-norm), step=0, total_steps=0
            NLAYERS, VOCAB, DIM, HIDDEN, HEADS, SEQ,
            0.0, 0.0,                      # lr, loss (not training)
            0.0, 0.0, 0.0,                # cum_compile, cum_train, cum_wall
            0, 0, 0,                       # cum_steps, cum_batches, adam_t
            KV_HEADS, HD, Q_DIM,
        )
        f.write(header)
        print(f"  Header: {len(header)} bytes")

        # Per-layer weights + zero Adam state
        adam_zeros_per_weight = None  # will compute per tensor
        for L in range(NLAYERS):
            prefix = f"model.layers.{L}"

            # Qwen3 attention: q_proj [Q_DIM, DIM], k_proj [KV_DIM, DIM], etc.
            Wq = get(f"{prefix}.self_attn.q_proj.weight")  # [Q_DIM, DIM]
            Wk = get(f"{prefix}.self_attn.k_proj.weight")  # [KV_DIM, DIM]
            Wv = get(f"{prefix}.self_attn.v_proj.weight")  # [KV_DIM, DIM]
            Wo = get(f"{prefix}.self_attn.o_proj.weight")  # [DIM, Q_DIM]

            # Qwen3 MLP: gate_proj=W1, down_proj=W2, up_proj=W3
            W1 = get(f"{prefix}.mlp.gate_proj.weight")     # [HIDDEN, DIM]
            W2 = get(f"{prefix}.mlp.down_proj.weight")     # [DIM, HIDDEN]
            W3 = get(f"{prefix}.mlp.up_proj.weight")       # [HIDDEN, DIM]

            rms_att = get(f"{prefix}.input_layernorm.weight")           # [DIM]
            rms_ffn = get(f"{prefix}.post_attention_layernorm.weight")  # [DIM]

            # QK-norm weights (Qwen3-specific)
            q_norm_w = get(f"{prefix}.self_attn.q_norm.weight")        # [HD]
            k_norm_w = get(f"{prefix}.self_attn.k_norm.weight")        # [HD]

            # Verify shapes
            assert Wq.shape == (Q_DIM, DIM), f"L{L} Wq: {Wq.shape} != ({Q_DIM}, {DIM})"
            assert Wk.shape == (KV_DIM, DIM), f"L{L} Wk: {Wk.shape} != ({KV_DIM}, {DIM})"
            assert Wv.shape == (KV_DIM, DIM), f"L{L} Wv: {Wv.shape} != ({KV_DIM}, {DIM})"
            assert Wo.shape == (DIM, Q_DIM), f"L{L} Wo: {Wo.shape} != ({DIM}, {Q_DIM})"
            assert W1.shape == (HIDDEN, DIM), f"L{L} W1: {W1.shape} != ({HIDDEN}, {DIM})"
            assert W2.shape == (DIM, HIDDEN), f"L{L} W2: {W2.shape} != ({DIM}, {HIDDEN})"
            assert W3.shape == (HIDDEN, DIM), f"L{L} W3: {W3.shape} != ({HIDDEN}, {DIM})"
            assert rms_att.shape == (DIM,), f"L{L} rms_att: {rms_att.shape}"
            assert rms_ffn.shape == (DIM,), f"L{L} rms_ffn: {rms_ffn.shape}"
            assert q_norm_w.shape == (HD,), f"L{L} q_norm: {q_norm_w.shape} != ({HD},)"
            assert k_norm_w.shape == (HD,), f"L{L} k_norm: {k_norm_w.shape} != ({HD},)"

            # Write weights (row-major float32, matching save_checkpoint order)
            f.write(Wq.tobytes())
            f.write(Wk.tobytes())
            f.write(Wv.tobytes())
            f.write(Wo.tobytes())
            f.write(W1.tobytes())
            f.write(W2.tobytes())
            f.write(W3.tobytes())
            f.write(rms_att.tobytes())
            f.write(rms_ffn.tobytes())
            f.write(q_norm_w.tobytes())
            f.write(k_norm_w.tobytes())

            # Adam state: 2 floats (m,v) per weight element + 2 rms + 2 qk-norm
            # Order: Wq(m,v), Wk(m,v), Wv(m,v), Wo(m,v), W1(m,v), W2(m,v), W3(m,v), rms_att(m,v), rms_ffn(m,v), q_norm(m,v), k_norm(m,v)
            adam_sizes = [WQ_SZ, WK_SZ, WV_SZ, WO_SZ, W1_SZ, W2_SZ, W3_SZ, DIM, DIM, HD, HD]
            for sz in adam_sizes:
                f.write(np.zeros(sz, dtype=np.float32).tobytes())  # m
                f.write(np.zeros(sz, dtype=np.float32).tobytes())  # v

            if L % 7 == 0:
                print(f"  Layer {L}/{NLAYERS}")

        # rms_final (model.norm.weight)
        rms_final = get("model.norm.weight")  # [DIM]
        assert rms_final.shape == (DIM,)
        f.write(rms_final.tobytes())

        # Adam state for rms_final (m, v)
        f.write(np.zeros(DIM, dtype=np.float32).tobytes())
        f.write(np.zeros(DIM, dtype=np.float32).tobytes())

        # Embedding (model.embed_tokens.weight) [VOCAB, DIM]
        embed = get("model.embed_tokens.weight")
        assert embed.shape == (VOCAB, DIM), f"embed: {embed.shape} != ({VOCAB}, {DIM})"
        f.write(embed.tobytes())

        # Adam state for embed (m, v)
        f.write(np.zeros(VOCAB * DIM, dtype=np.float32).tobytes())
        f.write(np.zeros(VOCAB * DIM, dtype=np.float32).tobytes())

    file_size = os.path.getsize(args.output)
    print(f"\nDone! {args.output}: {file_size / 1e9:.2f} GB")
    print(f"Run inference: python3 infer.py --ckpt {args.output} \"Once upon a time\"")


if __name__ == "__main__":
    main()
