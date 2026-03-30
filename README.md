# ane

> every Mac has a neural engine. no one lets you use it.

ane is a pure Rust driver for Apple Neural Engine — the 15.8 TOPS
accelerator sitting idle inside every Apple Silicon chip. no ObjC
compiler. no Swift. no CoreML. no Python. no dependencies.

compile a MIL program, load it into ANE SRAM, dispatch, read the output.

```rust
use ane::{MilProgram, AneSurface, AneModel};

let program = ane::mil::matmul(64, 64, 64);
let mut model = AneModel::compile(&program, &[]).unwrap();
model.load().unwrap();

let input = AneSurface::new(program.input_bytes()).unwrap();
let output = AneSurface::new(program.output_bytes()).unwrap();
model.run(&input, &output).unwrap();
```

requires macOS + Apple Silicon.

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

## api

```rust,ignore
// compile MIL to ANE bytecode
AneModel::compile(program: &MilProgram, weights: &[(&str, &[u8])]) -> Result<Self>

// upload bytecode to ANE SRAM
model.load() -> Result<()>

// dispatch on hardware (synchronous)
model.run(input: &AneSurface, output: &AneSurface) -> Result<()>

// free SRAM (also automatic on drop)
model.unload() -> Result<()>

// shared-memory tensor buffer (IOSurface)
AneSurface::new(bytes) -> Result<Self>
AneSurface::with_shape(channels, spatial) -> Result<Self>
surface.with_data(|&[u16]|)
surface.with_data_mut(|&mut [u16]|)

// MIL program builder
MilProgram::matmul(ic, oc, seq) -> Self
MilProgram::from_text(text, in_ch, in_sp, out_ch, out_sp) -> Self

// fp16 conversion (NEON-accelerated)
ane::f32_to_fp16(f32) -> u16
ane::fp16_to_f32(u16) -> f32
```

---

## usage

```bash
# verify ANE access works
cargo run --example matmul

# 7-level ANE reverse engineering probe
cargo run --bin ane_probe
```

---

## structure

one crate. zero external dependencies — only macOS system frameworks.

```
src/
  lib.rs                AneModel, AneSurface, MilProgram, AneError
  model.rs              compile / load / run / unload
  surface.rs            IOSurface wrapper, NEON fp16
  ffi.rs                IOKit, CoreFoundation, IOSurface, libobjc
  mil/mod.rs            MIL program builder (matmul, header, footer)
  probe/                ane_probe — 7-level reverse engineering tool
examples/
  matmul.rs             64x64 matmul on ANE hardware
specs/
  api.md                API specification with Apple ObjC mapping
docs/
  explanation/
    how-ane-works.md    how ane bypasses Apple's three barriers
```

model-specific code (transformer kernels, CPU ops, training,
inference, tools) lives in [cyb/llm](https://github.com/cyberia-to/cyb)
where ane serves as the ANE backend driver.

---

## contributing

we welcome pull requests. especially:

- **hardware** — tested only on M1 Pro. M2/M3/M4 reports are gold
- **MIL operations** — more verified ops expand what ane can compile
- **performance** — faster compile/load/dispatch cycles
- **safety** — soundness fixes in the FFI layer

---

## license

cyber license: don't trust. don't fear. don't beg.
