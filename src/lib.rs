//! `ane` — Pure Rust access to Apple Neural Engine
//!
//! Compile MIL programs, load them into ANE hardware, and run inference
//! with zero external dependencies. Only requires macOS with Apple Silicon.
//!
//! ```no_run
//! use ane::{MilProgram, AneSurface, AneModel};
//!
//! let program = MilProgram::matmul(64, 64, 64);
//! let mut model = AneModel::compile(&program, &[])?;
//! model.load()?;
//!
//! let input = AneSurface::new(program.input_bytes())?;
//! let output = AneSurface::new(program.output_bytes())?;
//! model.run(&input, &output)?;
//! # Ok::<(), ane::AneError>(())
//! ```

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::missing_transmute_annotations
)]

pub mod accel;
pub mod config;
pub mod ffi;
pub mod mil;
pub mod model;
pub mod ops;
pub mod staging;
pub mod surface;
pub mod weights;

pub use config::ModelConfig;
pub use mil::{build_weight_blob, MilProgram};
pub use model::AneModel;
pub use surface::{f32_to_fp16, fp16_to_f32, AneSurface};
pub use weights::{load_checkpoint, CkptHeader, KVCache, LayerWeights};

#[derive(Debug)]
#[non_exhaustive]
pub enum AneError {
    SurfaceCreationFailed(String),
    ClassNotFound(&'static str),
    DescriptorCreationFailed,
    ModelCreationFailed,
    CompilationFailed(String),
    LoadFailed(String),
    EvalFailed(String),
    UnloadFailed(String),
    Io(std::io::Error),
}

impl std::fmt::Display for AneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AneError::SurfaceCreationFailed(msg) => {
                write!(f, "IOSurface creation failed: {}", msg)
            }
            AneError::ClassNotFound(name) => write!(f, "ObjC class not found: {}", name),
            AneError::DescriptorCreationFailed => write!(f, "Model descriptor creation failed"),
            AneError::ModelCreationFailed => write!(f, "Model creation failed"),
            AneError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            AneError::LoadFailed(msg) => write!(f, "Load failed: {}", msg),
            AneError::EvalFailed(msg) => write!(f, "Evaluation failed: {}", msg),
            AneError::UnloadFailed(msg) => write!(f, "Unload failed: {}", msg),
            AneError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for AneError {}

impl From<std::io::Error> for AneError {
    fn from(e: std::io::Error) -> Self {
        AneError::Io(e)
    }
}
