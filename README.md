# ane — Pure Rust Apple Neural Engine

compile MIL programs, load into ANE hardware, run inference. zero dependencies. zero ObjC.

## quickstart

```rust
use ane::{MilProgram, AneSurface, AneModel, f32_to_fp16, fp16_to_f32};

let program = ane::mil::matmul(64, 64, 64);
let mut model = AneModel::compile(&program, &[]).unwrap();
model.load().unwrap();

let input = AneSurface::new(program.input_bytes()).unwrap();
let output = AneSurface::new(program.output_bytes()).unwrap();
model.run(&input, &output).unwrap();
```

requirements: macOS + Apple Silicon. nothing else.

## performance (M1 Pro, Qwen3-0.6B)

| metric | ObjC (old) | Rust | |
|--------|-----------|------|---|
| prefill (4 tokens) | 395ms | 363ms | Rust 1.09x faster |
| decode | 87.8ms/tok | 82.7ms/tok | Rust 1.06x faster |
| throughput | 11.4 tok/s | 12.1 tok/s | Rust 1.06x faster |

all 12 ANE kernels verified on hardware. inline NEON asm for fp16 conversion. Accelerate.framework (cblas, vDSP, vvexpf) for CPU ops.

## structure

```
src/
    lib.rs              public API
    ffi.rs              IOKit, CoreFoundation, IOSurface, libobjc FFI
    accel.rs            Accelerate.framework FFI (cblas, vDSP, vecLib)
    surface.rs          IOSurface wrapper, NEON fp16 conversion
    model.rs            AneModel compile/load/run/unload
    config.rs           ModelConfig (Qwen3-0.6B, Stories-110M)
    weights.rs          checkpoint load, LayerWeights, KVCache
    mil/
      mod.rs            matmul builder, weight blob format
      sdpa.rs           SDPA forward + backward (RoPE, GQA, causal mask)
      ffn.rs            fused SwiGLU FFN with residual
      projection.rs     QKV, Wo, backward projection kernels
    ops/
      rmsnorm.rs        vDSP-optimized forward/backward
      rope.rs           rotary position embedding
      attention.rs      cblas_sgemm causal attention + cached decode
      loss.rs           cross-entropy with vvexpf softmax
      adam.rs           AdamW optimizer
      embed.rs          embedding lookup, vocab compaction
      activation.rs     SiLU, GQA tile/reduce
examples/
    matmul.rs           minimal ANE matmul demo
    compile_kernels.rs  verify all 12 kernels compile on ANE
    bench.rs            kernel throughput benchmark
    infer.rs            full inference (ANE prefill + CPU decode + KV-cache)
    train.rs            training pipeline (forward + backward + optimizer)
tools/
  convert_hf.py         HuggingFace → ANE checkpoint converter
  infer.py              tokenization wrapper for inference
  dashboard.py          TUI training dashboard
  tokenize.py           training data preparation
```

## examples

```bash
# verify ANE access
cargo run --example matmul

# compile all 12 Qwen3-0.6B kernels on ANE
cargo run --example compile_kernels

# benchmark kernel throughput
cargo run --release --example bench

# inference with checkpoint
python3 tools/convert_hf.py  # download + convert weights
echo -e "2610\n330\n279\n525\n" | cargo run --release --example infer -- --ckpt ane_qwen3_06b_dyn_ckpt.bin

# training
cargo run --release --example train -- --ckpt ane_qwen3_06b_dyn_ckpt.bin --data tinystories_data00.bin
```

## how it works

```
your Rust code
  → ane crate (objc_msgSend FFI to libobjc)
    → AppleNeuralEngine.framework (dlopen at runtime)
      → _ANEInMemoryModel (compile MIL → ANE bytecode)
        → XPC to aned daemon
          → IOKit H11ANEIn driver
            → ANE hardware
```

no ObjC compiler. no Swift. no headers. no linking. no dependencies.

IOSurfaces for zero-copy CPU↔ANE data transfer. inline NEON asm for fp16↔f32 conversion. Accelerate.framework for BLAS/vDSP.

see [ANE_API.md](ANE_API.md) for complete API reference.

## license

cyber license: don't trust. don't fear. don't beg.
