# Claude Code Instructions

## auditor mindset

the project is supervised by an engineer with 30 years of experience.
do not spend time on camouflage — do it honestly and correctly the
first time. one time correctly is cheaper than five times beautifully.

## honesty

never fake results. if a system produces nothing — show nothing.
a dash is more honest than a copied number. never substitute
appearance of progress for actual progress. never generate placeholder
data to fill a gap.

## literal interpretation

when the user says something, they mean it literally. do not
reinterpret. if unsure, ask once. do not guess and iterate.

known failure mode: the user says "show real numbers" and the agent
reformats display labels instead of showing actual data. this is the
masquerading instinct — optimizing for "looks correct" instead of
"is correct."

## chain of verification

for non-trivial decisions affecting correctness:
1. initial answer
2. 3-5 verification questions that would expose errors
3. answer each independently
4. revised answer incorporating corrections

skip for trivial tasks.

## build & verify

```bash
cargo fmt --all                           # format
cargo clippy --workspace -- -W warnings   # lint
cargo build --release --workspace         # build all
cargo run --example matmul                # verify ANE access
```

every commit: format clean, clippy clean, builds, examples run.

## project: rane (crate name: ane)

pure Rust access to Apple Neural Engine. compile MIL programs, load
into ANE hardware, run inference and training. zero external
dependencies in the core crate — only macOS system frameworks.

## architecture

cargo workspace with two members:

- `ane` (root crate) — library + ane_probe binary + examples.
  zero external dependencies. links only system frameworks via FFI.
- `tools/` (crate `ane-tools`) — CLI binaries (convert_hf, tokenize,
  chat). heavy dependencies (safetensors, ureq, tokenizers, zip)
  isolated here.

```
src/                  core library (zero deps)
  lib.rs              public API: AneModel, AneSurface, MilProgram, AneError
  main.rs             ane_probe — 7-level reverse engineering probe
  ffi.rs              IOKit, CoreFoundation, IOSurface, libobjc FFI
  accel.rs            Accelerate.framework FFI (cblas, vDSP, vecLib)
  surface.rs          IOSurface wrapper, inline NEON asm for fp16
  model.rs            AneModel compile/load/run/unload via objc_msgSend
  config.rs           ModelConfig presets (Qwen3-0.6B, Stories-110M)
  weights.rs          checkpoint I/O, LayerWeights, KVCache
  mil/                MIL program builder → ANE bytecode
  ops/                CPU kernels (rmsnorm, rope, attention, loss, adam)
examples/             runnable demos (matmul, infer, train, bench)
tools/                separate crate with heavy deps (convert_hf, tokenize, chat)
specs/                specification (source of truth)
  api.md              API spec — concepts, lifecycle, apple mapping
docs/                 documentation (Diataxis)
  explanation/        conceptual understanding
```

## source of truth

`specs/` is canonical. if specs/ and code disagree, resolve
in specs/ first, then propagate to code.

`docs/` is documentation — teaches, guides, explains. references spec
but does not duplicate it.

## pipeline contract

```
MIL text → AppleNeuralEngine.framework (dlopen)
  → _ANEInMemoryModel (compile → ANE bytecode)
    → XPC to aned daemon
      → IOKit H11ANEIn driver
        → ANE hardware
```

IOSurfaces are the only data path CPU↔ANE. they must be locked
before read/write and unlocked before ANE dispatch.

## key gotchas

- ANE access requires Apple Silicon Mac. examples fail on Intel/Linux.
- MIL programs are text, compiled at runtime via dlopen. no link dep.
- weight blobs: little-endian fp16, 128-byte header (magic 0xDEADBEEF).
- all matmuls use row-major: W[rows, cols] @ x[cols, seq].
- checkpoint format: custom binary (magic 0x424C5A54, version 5).
- tools/ binaries download from HuggingFace at runtime — needs network.
- never write naive loops for matmul — use accel::sgemm.

## do not touch

without explicit discussion:
- Cargo.toml dependency versions
- src/ffi.rs FFI signatures (must match system frameworks)
- specs/ structure
- LICENSE

## quality

file size limit: 500 lines per source file. split into submodules
if exceeded.

every commit:
- type check / lint — zero warnings
- builds clean
- examples run

## coding conventions

- no ObjC, no Swift, no headers. FFI through libobjc + dlopen.
- IOSurfaces for zero-copy CPU↔ANE data. inline NEON asm for fp16.
- Accelerate.framework (cblas, vDSP, vecLib) for CPU ops.
- `cargo fmt` enforced (max_width = 100). clippy clean.

## git workflow

- atomic commits — one logical change per commit
- conventional prefixes: feat:, fix:, refactor:, docs:, test:, chore:
- commit by default after completing a change

## agent memory

plans and design documents persist in `.claude/plans/`, not in
ephemeral storage. read what is already there before writing.

## parallel agents

split by non-overlapping file scopes. never let two agents edit the
same file.

## estimation model

estimate work in sessions (3 focused hours) and pomodoros (30 min).

## shell: nushell

use `nu -c '...'` or `nu script.nu` for scripting.
reserve bash for git commands and system tools only.

## writing style

state what something is directly. never define by negation.

## license

cyber license: don't trust. don't fear. don't beg.
