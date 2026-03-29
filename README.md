# ane

> every Mac has a neural engine. no one lets you use it.

ane is pure Rust access to Apple Neural Engine — the 15.8 TOPS
accelerator sitting idle inside every Apple Silicon chip. no ObjC
compiler. no Swift. no CoreML. no Python. no dependencies.

you write a matrix operation. ane compiles it to ANE bytecode, uploads
to hardware, runs it, and gives you the result. one crate. one call.

```rust
use ane::{MilProgram, AneSurface, AneModel};

let program = ane::mil::matmul(64, 64, 64);
let mut model = AneModel::compile(&program, &[]).unwrap();
model.load().unwrap();

let input = AneSurface::new(program.input_bytes()).unwrap();
let output = AneSurface::new(program.output_bytes()).unwrap();
model.run(&input, &output).unwrap();
```

requirements: macOS + Apple Silicon.

---

## why this exists

Apple ships a 15.8 TOPS neural accelerator in every M-series chip.
the only official way to use it is CoreML — a framework that decides
when (and whether) your model runs on ANE. no direct access. no
documentation. no public API.

ane reaches ANE hardware through the same private ObjC classes that
CoreML uses internally. it calls `objc_msgSend` directly from Rust —
no ObjC compiler needed, `objc_msgSend` is just a C function. three
private frameworks loaded via `dlopen`. weight data passes through
IOSurfaces — kernel-managed shared memory, zero copies.

the result: compile a MIL program, load it into ANE SRAM, dispatch,
read the output. four calls. the same four calls for a 64×64 matmul
or a 28-layer transformer.

---

## what it runs

Qwen3-0.6B on M1 Pro:

| | speed | vs CoreML path |
|---|---|---|
| prefill (4 tokens) | 363ms | 1.09x faster |
| decode | 82.7ms/tok | 1.06x faster |
| throughput | 12.1 tok/s | — |

all 12 ANE kernels (3 forward + 9 backward) compile and run on
hardware. training loop with AdamW and cosine LR schedule included.

---

## the stack

```
your Rust code
  → ane crate (objc_msgSend to libobjc)
    → AppleNeuralEngine.framework (dlopen)
      → MIL text → ANE bytecode
        → XPC to aned daemon
          → IOKit H11ANEIn driver
            → ANE hardware
```

three barriers. three bypasses:

| barrier | how ane gets through |
|---------|---------------------|
| IOKit entitlement | XPC to aned (the daemon that has the entitlement) |
| private frameworks | dlopen at runtime, class names via ObjC runtime |
| undocumented MIL format | reverse-engineered from CoreML model bundles |

see [how-ane-works](docs/explanation/how-ane-works.md) for the full story.

---

## usage

```bash
# verify ANE access works
cargo run --example matmul

# compile all 12 Qwen3-0.6B kernels on ANE
cargo run --example compile_kernels

# benchmark
cargo run --release --example bench

# download and convert Qwen3-0.6B weights
cargo run --release -p ane-tools --bin convert_hf

# inference
cargo run --release --example infer -- --ckpt ane_qwen3_06b_dyn_ckpt.bin

# chat
cargo run --release -p ane-tools --bin chat

# train from scratch
cargo run --release --example train -- --scratch --steps 100
```

---

## structure

two crates. the core library has zero external dependencies.

```
src/                    ane crate — zero deps, system frameworks only
  lib.rs                AneModel, AneSurface, MilProgram, AneError
  model.rs              compile / load / run / unload
  surface.rs            IOSurface wrapper, NEON fp16
  staging.rs            weight/activation staging for ANE dispatch
  ffi.rs                IOKit, CoreFoundation, IOSurface, libobjc
  accel.rs              Accelerate.framework (cblas, vDSP, vecLib)
  mil/                  MIL program builder
  ops/                  CPU kernels (rmsnorm, attention, loss, adam, ...)
  probe/                ane_probe — 7-level reverse engineering tool

tools/                  ane-tools crate — heavy deps (safetensors, ureq, ...)
  convert_hf.rs         HuggingFace → ANE checkpoint
  chat.rs               interactive chat
  tokenize.rs           training data prep

specs/api.md            API specification
```

---

## docs

- [specs/api.md](specs/api.md) — API specification with Apple ObjC mapping
- [docs/explanation/how-ane-works.md](docs/explanation/how-ane-works.md) — how ane bypasses Apple's three barriers

---

## license

cyber license: don't trust. don't fear. don't beg.
