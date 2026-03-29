# ane — Pure Rust Apple Neural Engine

compile MIL programs, load into ANE hardware, run inference and training. zero dependencies. zero ObjC.

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

| metric | Rust | tokens/s |
|--------|------|----------|
| prefill (4 tokens) | 363ms | — |
| decode | 82.7ms/tok | 12.1 |

all 12 ANE kernels verified on hardware. Accelerate.framework for CPU ops. inline NEON asm for fp16.

## how it works

```
Rust code → objc_msgSend FFI
  → AppleNeuralEngine.framework (dlopen)
    → compile MIL → ANE bytecode
      → XPC to aned daemon
        → IOKit H11ANEIn → ANE hardware
```

no ObjC compiler. no Swift. no headers. no linking.
IOSurfaces for zero-copy CPU↔ANE. see [how-ane-works](docs/explanation/how-ane-works.md) for details.

## usage

```bash
cargo run --example matmul                # verify ANE access
cargo run --example compile_kernels       # compile all 12 Qwen3-0.6B kernels
cargo run --release --example bench       # kernel throughput benchmark
cargo run --release -p ane-tools --bin convert_hf   # download + convert weights
cargo run --release --example infer -- --ckpt ane_qwen3_06b_dyn_ckpt.bin
cargo run --release -p ane-tools --bin chat
cargo run --release --example train -- --scratch --steps 100
```

## structure

```
src/
  lib.rs                public API: AneModel, AneSurface, MilProgram, AneError
  model.rs              compile / load / run / unload lifecycle
  surface.rs            IOSurface wrapper, NEON fp16 conversion
  ffi.rs                IOKit, CoreFoundation, IOSurface, libobjc FFI
  accel.rs              Accelerate.framework (cblas, vDSP, vecLib)
  staging.rs            IOSurface weight/activation staging for ANE kernels
  config.rs             model configs (Qwen3-0.6B, Stories-110M)
  weights.rs            checkpoint I/O, LayerWeights, KVCache
  mil/                  MIL program builder → ANE bytecode
    sdpa.rs             SDPA forward + backward (RoPE, GQA, causal mask)
    ffn.rs              fused SwiGLU FFN with residual
    projection.rs       QKV, Wo, backward projection kernels
  ops/                  CPU kernels
    rmsnorm.rs          vDSP-optimized forward/backward
    rope.rs             rotary position embedding
    attention.rs        causal attention + cached decode (cblas)
    loss.rs             cross-entropy (vvexpf softmax)
    adam.rs             AdamW optimizer
    embed.rs            embedding lookup, vocab compaction
    activation.rs       SiLU, GQA tile/reduce
    sample.rs           top-k sampling, argmax, PRNG
  probe/                ane_probe binary — 7-level reverse engineering probe
examples/
  matmul.rs             minimal ANE matmul demo
  compile_kernels.rs    verify all 12 kernels compile on ANE
  bench.rs              kernel throughput benchmark
  infer.rs              inference (ANE prefill + CPU decode + KV-cache)
  train.rs              training (forward + backward + optimizer)
tools/                  separate crate (ane-tools), heavy dependencies
  convert_hf.rs         HuggingFace → ANE checkpoint converter
  tokenize.rs           training data preparation
  chat.rs               interactive chat with tokenization
specs/
  api.md                API specification
docs/
  explanation/
    how-ane-works.md    how the crate reaches ANE hardware
```

## docs

- [specs/api.md](specs/api.md) — API specification: concepts, lifecycle, apple mapping
- [docs/explanation/how-ane-works.md](docs/explanation/how-ane-works.md) — how ane bypasses Apple's three barriers

## license

cyber license: don't trust. don't fear. don't beg.
