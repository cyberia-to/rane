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

#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

pub mod ffi;
pub mod accel;
pub mod surface;
pub mod model;
pub mod mil;
pub mod ops;
pub mod config;
pub mod weights;

pub use surface::{AneSurface, fp16_to_f32, f32_to_fp16};
pub use model::AneModel;
pub use mil::{MilProgram, build_weight_blob};
pub use config::ModelConfig;
pub use weights::{LayerWeights, KVCache, CkptHeader, load_checkpoint};

#[derive(Debug)]
pub enum AneError {
    SurfaceCreationFailed,
    ClassNotFound(&'static str),
    DescriptorCreationFailed,
    ModelCreationFailed,
    CompilationFailed(String),
    LoadFailed(String),
    EvalFailed(String),
    UnloadFailed(String),
}

impl std::fmt::Display for AneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AneError::SurfaceCreationFailed => write!(f, "IOSurface creation failed"),
            AneError::ClassNotFound(name) => write!(f, "ObjC class not found: {}", name),
            AneError::DescriptorCreationFailed => write!(f, "Model descriptor creation failed"),
            AneError::ModelCreationFailed => write!(f, "Model creation failed"),
            AneError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            AneError::LoadFailed(msg) => write!(f, "Load failed: {}", msg),
            AneError::EvalFailed(msg) => write!(f, "Evaluation failed: {}", msg),
            AneError::UnloadFailed(msg) => write!(f, "Unload failed: {}", msg),
        }
    }
}

impl std::error::Error for AneError {}
