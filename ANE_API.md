# Apple Neural Engine — Rust API

pure Rust access to Apple Neural Engine. zero dependencies. zero ObjC.

## quickstart

```rust
use ane::{MilProgram, AneSurface, AneModel, f32_to_fp16, fp16_to_f32};

// compile a 64×64 matmul for ANE
let program = MilProgram::matmul(64, 64, 64);
let mut model = AneModel::compile(&program, &[]).unwrap();
model.load().unwrap();

// create I/O surfaces
let input = AneSurface::new(program.input_bytes()).unwrap();
let output = AneSurface::new(program.output_bytes()).unwrap();

// fill input, run, read output
input.with_data_mut(|data| { /* write fp16 */ });
model.run(&input, &output).unwrap();
output.with_data(|data| { /* read fp16 */ });
```

add to any Rust project:
```toml
[dependencies]
ane = { path = "../rust_ane" }
```

requirements: macOS + Apple Silicon. nothing else.

---

## architecture

```
your Rust code
  → ane crate (pure Rust FFI)
    → libobjc.dylib (objc_msgSend, objc_getClass, sel_registerName)
      → AppleNeuralEngine.framework (loaded via dlopen at runtime)
        → _ANEInMemoryModel (compile, load, evaluate)
          → XPC to aned daemon
            → IOKit H11ANEIn driver
              → ANE hardware
```

no ObjC compiler. no Swift. no headers. no linking.

the crate uses `objc_msgSend` FFI to call ObjC classes from AppleNeuralEngine.framework at runtime. these are the same calls the ObjC training code makes — but from pure Rust.

IOSurfaces (shared-memory tensors) are created via the public IOSurface.framework C API. no entitlements needed.

---

## crate structure

```
rust_ane/
  src/
    lib.rs        public API: AneModel, AneSurface, MilProgram, AneError
    ffi.rs        raw FFI bindings: IOKit, CoreFoundation, IOSurface, libobjc, dlopen
    surface.rs    AneSurface — IOSurface wrapper with safe lock/unlock
    model.rs      AneModel — compile → load → run → unload lifecycle
    mil.rs        MilProgram — MIL text builder with matmul generator
    main.rs       ane_probe — 7-level reverse engineering probe (binary)
  examples/
    matmul.rs     minimal matmul demo
```

---

## API reference

### AneModel

the core type. compiles MIL to ANE bytecode, loads into hardware, runs inference.

```rust
// compile MIL program with optional weight blobs
let mut model = AneModel::compile(&program, &weights)?;

// load compiled bytecode into ANE SRAM
model.load()?;

// run on ANE hardware (synchronous, blocks until done)
model.run(&input_surface, &output_surface)?;

// unload from ANE (also called automatically on Drop)
model.unload()?;
```

`compile` calls these ObjC methods internally:
1. `_ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:`
2. `_ANEInMemoryModel.inMemoryModelWithDescriptor:`
3. `model.compileWithQoS:options:error:` (calls ANECCompile → produces bytecode)

`load` calls `model.loadWithQoS:options:error:` — uploads bytecode to ANE SRAM.

`run` wraps IOSurfaces in `_ANEIOSurfaceObject`, builds `_ANERequest`, calls `evaluateWithQoS:options:request:error:`.

Drop calls `unloadWithQoS:error:` and cleans up the temp directory.

### AneSurface

shared-memory tensor buffer backed by IOSurface. zero-copy between CPU and ANE.

```rust
// allocate by byte size
let surface = AneSurface::new(8192)?;

// allocate by ANE tensor shape [1, C, 1, S] in fp16
let surface = AneSurface::with_shape(64, 64)?;  // 64 channels × 64 spatial × 2 bytes

// write fp16 data (locks surface, calls closure, unlocks)
surface.with_data_mut(|data: &mut [u16]| {
    data[0] = f32_to_fp16(1.0);  // 0x3C00
});

// read fp16 data (read-only lock)
surface.with_data(|data: &[u16]| {
    let val = fp16_to_f32(data[0]);  // 1.0
});

// metadata
surface.id();    // IOSurface ID
surface.size();  // allocation in bytes
```

Drop calls `CFRelease` to free the IOSurface.

### MilProgram

builds MIL (Model Intermediate Language) text for ANE compilation.

```rust
// built-in matmul: y = x @ W
// input: [1, ic, 1, seq+oc] — activations packed with weights
// output: [1, oc, 1, seq]
let program = MilProgram::matmul(ic, oc, seq);

// custom MIL text
let program = MilProgram::from_text(mil_str, in_ch, in_sp, out_ch, out_sp);

// query shapes
let (channels, spatial) = program.input_shape();   // e.g. (64, 128)
let (channels, spatial) = program.output_shape();  // e.g. (64, 64)
let bytes = program.input_bytes();                 // channels × spatial × 2
```

### fp16 conversion

```rust
let fp16: u16 = f32_to_fp16(1.0);   // 0x3C00
let f32: f32 = fp16_to_f32(0x3C00); // 1.0
```

### AneError

```rust
pub enum AneError {
    SurfaceCreationFailed,
    ClassNotFound(&'static str),      // ObjC class not in framework
    DescriptorCreationFailed,
    ModelCreationFailed,
    CompilationFailed(String),        // includes NSError description
    LoadFailed(String),
    EvalFailed(String),
    UnloadFailed(String),
}
```

implements `Display` and `Error`.

---

## MIL program format

MIL is a text-based IR compiled to ANE bytecode. the `ane` crate generates it programmatically.

### structure

```
program(1.3)
[buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
})]
{
    func main<ios18>(tensor<fp16, [1, IC, 1, SP]> x) {
        // operations
    } -> (output_var);
}
```

the outer `{{ }}` double-braces around the dict entries are mandatory — the parser treats them as dict-literal delimiters. single braces `{ }` wrap individual key-value pairs.

`func main<ios18>` — the `ios18` target selects the ANE instruction set for Apple Silicon.

### supported operations (verified on ANE hardware)

| operation | MIL syntax | notes |
|---|---|---|
| const | `type var = const()[name=..., val=...]` | compile-time constants |
| slice_by_size | `slice_by_size(x=input, begin=offsets, size=dims)` | tensor slicing |
| reshape | `reshape(shape=new_shape, x=input)` | reshape without data copy |
| transpose | `transpose(perm=permutation, x=input)` | dimension permutation |
| matmul | `matmul(transpose_x=bF, transpose_y=bF, x=A, y=B)` | matrix multiplication |
| add | `add(x=A, y=B)` | element-wise addition |
| mul | `mul(x=A, y=B)` | element-wise multiply |
| sub | `sub(x=A, y=B)` | element-wise subtract |
| concat | `concat(axis=int, interleave=bool, values=(A,B,...))` | concatenation |
| softmax | `softmax(axis=int, x=input)` | softmax along axis |
| sigmoid | `sigmoid(x=input)` | element-wise sigmoid |
| conv | `conv(x=input, weight=W, ...)` | convolution (1×1 = dense) |
| cast | `cast(dtype="fp16", x=input)` | type conversion |
| quantize | `quantize(input=x, scale=s, zero_point=z, ...)` | fp16 → int8 |
| dequantize | `dequantize(input=x, scale=s, ...)` | int8 → fp16 |

operations that parse but ANE rejects: `reduce_mean`, `rsqrt`, `reduce_sum`, `pow`. these must stay on CPU.

### dynamic matmul pattern

weights and activations packed into one input tensor:

```
input [1, IC, 1, SEQ+OC]:
  spatial[0..SEQ]     = activations
  spatial[SEQ..SEQ+OC] = weight matrix (transposed)
```

the MIL program slices, reshapes, and matmuls. this avoids multiple input bindings and lets one IOSurface carry everything a kernel needs per layer.

`MilProgram::matmul(ic, oc, seq)` generates this pattern automatically.

### BLOBFILE references (static weights)

for pre-compiled weights embedded in MIL:

```
tensor<fp16, [1,1,256,128]> rope_cos = const()[
    val=tensor<fp16, [1,1,256,128]>(
        BLOBFILE(path=string("@model_path/weights/rope_cos.bin"), offset=uint64(64))
    )
];
```

`@model_path` resolves to the temp directory. pass weight blobs via the `weights` argument to `AneModel::compile()`.

---

## weight blob binary format

### fp16 blob (128-byte header)

```
offset  content
0x00    0x01 (version)
0x04    0x02 (format flag)
0x40    0xEF 0xBE 0xAD 0xDE (magic: DEADBEEF little-endian)
0x44    0x01 (data type: fp16)
0x48    uint32 data_size_bytes
0x50    uint32 data_offset (128)
0x80    _Float16[] weights
```

use `build_weight_blob()` to create blobs from fp16 data:

```rust
use ane::build_weight_blob;
let blob = build_weight_blob(&fp16_values);
```

### int8 blob (64-byte header)

```
offset  content
0x00    0xEF 0xBE 0xAD 0xDE (magic)
0x04    0x01 (version)
0x0A    0x08 (8-bit marker)
0x40    int8[] weights
```

---

## tensor layout convention

ANE uses 4D tensors: `[batch, channels, height, width]`. for 1D sequences: `[1, C, 1, S]`.

for LLM inference:
- activations: `[1, dim, 1, seq]` — model dimension is channels, sequence is spatial
- dynamic weights packed into spatial alongside activations

example: QKV projection input `[1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM]`
- spatial `0..SEQ` = activations (xnorm)
- spatial `SEQ..SEQ+Q_DIM` = Wq weights
- spatial `SEQ+Q_DIM..` = Wk, Wv weights

---

## execution model

- one `AneModel` = one compiled kernel (one MIL function)
- requests are built internally by `run()` — no manual request management
- IOSurfaces are reusable: write new data, call `run()` again
- `run()` is synchronous (blocks until ANE completes)
- multiple models can be loaded simultaneously
- ANE execution is serial on hardware (one kernel at a time)
- per-layer weights are "dynamic": staged into input IOSurface before each call

---

## IOKit / driver architecture (from probe)

the 7-level probe (`cargo run --bin ane_probe`) reveals the full stack:

```
LEVEL 1: IOKit finds H11ANEIn driver in IORegistry
LEVEL 2: IOServiceOpen(type=1) succeeds — but all selectors are blocked
LEVEL 3: 64 selectors all return kIOReturnNotPermitted
         → process lacks com.apple.ane.iokit-user-access entitlement
         → only /usr/libexec/aned has this entitlement
LEVEL 4: IOSurface creation works (public API, no entitlement needed)
LEVEL 5: dlopen loads 3 private frameworks, finds 26 C symbols
LEVEL 6: bootstrap_look_up("com.apple.appleneuralengine") → Mach port to aned
LEVEL 7: compile + load + eval via ObjC runtime → ANE hardware
```

all user code goes through XPC to the aned daemon. aned has the IOKit entitlement and mediates all hardware access. this is why compilation and evaluation are safe — malformed bytecode never reaches hardware without aned validation.

three private frameworks loaded at runtime:
- AppleNeuralEngine.framework — ObjC classes, MIL validation, XPC interface
- ANECompiler.framework — ANECCompile, bytecode generation
- ANEServices.framework — device/program lifecycle (used by aned internally)

---

## C API (partially reverse-engineered)

the `ANECCompile` C function exists in ANECompiler.framework and is what `compileWithQoS:` calls internally. the crate currently uses the ObjC path because:

1. `ANECCreateCompilerInputDictionary()` hangs when called with no args — signature unknown
2. the ObjC path is proven and matches the training code exactly
3. `objc_msgSend` is just a C function — no ObjC compiler needed

symbols found in the frameworks:

| symbol | framework | status |
|---|---|---|
| ANECCompile | ANECompiler | used internally by ObjC path |
| ANECCompileJIT | ANECompiler | untested |
| ANECCompileOnline | ANECompiler | untested |
| ANECCompileOffline | ANECompiler | untested |
| ANECCreateCompilerInputDictionary | ANECompiler | hangs with no args |
| ANECCreateCompilerOptionDictionary | ANECompiler | hangs with no args |
| ANECCreateModelDictionary | ANECompiler | hangs with no args |
| ANECValidate | ANECompiler | untested |
| ANEServicesDeviceOpen | ANEServices | requires aned entitlement |
| ANEServicesProgramCreate | ANEServices | requires aned entitlement |
| ANEValidateMILNetworkOnHost | AppleNeuralEngine | may block on some systems |
| _ANEDaemonInterface | AppleNeuralEngine | returns XPC protocol spec |

future work: reverse-engineer the ANECCompile dictionary format to bypass the ObjC layer entirely.

---

## source references

| file | what it contains |
|---|---|
| rust_ane/src/lib.rs | public API: AneModel, AneSurface, MilProgram, AneError |
| rust_ane/src/ffi.rs | all FFI bindings (IOKit, CF, IOSurface, libobjc, dlopen) |
| rust_ane/src/surface.rs | AneSurface — IOSurface wrapper with lock/unlock |
| rust_ane/src/model.rs | AneModel — compile/load/run/unload via objc_msgSend |
| rust_ane/src/mil.rs | MilProgram builder, matmul generator, weight blob format |
| rust_ane/src/main.rs | 7-level probe (binary target) |
| rust_ane/examples/matmul.rs | clean matmul demo using library API |
| training/training_dynamic/io.h | ObjC compilation pipeline (reference implementation) |
| training/training_dynamic/mil_dynamic.h | MIL generators for all kernel types (Qwen3-0.6B) |
| training/training_dynamic/train.m | training pipeline (10 kernels: 3 fwd + 7 bwd) |
| training/training_dynamic/infer.m | inference pipeline with KV-cache |
