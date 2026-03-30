//! Hardware tests — require macOS + Apple Silicon with ANE
//! Run with: cargo test --test hardware -- --test-threads=1
//! Skipped on CI (Intel runners have no ANE).

use rane::{f32_to_fp16, fp16_to_f32, AneModel, AneSurface};

#[test]
fn compile_load_unload() {
    let p = rane::mil::matmul(64, 64, 64);
    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();
    model.unload().unwrap();
}

#[test]
fn compile_load_run_identity() {
    let ic = 64;
    let oc = 64;
    let seq = 64;
    let p = rane::mil::matmul(ic, oc, seq);

    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();

    let input = AneSurface::new(p.input_bytes()).unwrap();
    let output = AneSurface::new(p.output_bytes()).unwrap();

    input.with_data_mut(|d| {
        let sp = seq + oc;
        for ch in 0..ic {
            for s in 0..seq {
                d[ch * sp + s] = f32_to_fp16(1.0);
            }
            for o in 0..oc {
                d[ch * sp + seq + o] = if ch == o { f32_to_fp16(1.0) } else { 0 };
            }
        }
    });

    model.run(&input, &output).unwrap();

    output.with_data(|d| {
        let mut max_err: f32 = 0.0;
        for i in 0..oc * seq {
            let val = fp16_to_f32(d[i]);
            max_err = max_err.max((val - 1.0).abs());
        }
        assert!(max_err < 0.01, "identity matmul max_err = {}", max_err);
    });
}

#[test]
fn double_unload_is_safe() {
    let p = rane::mil::matmul(32, 32, 32);
    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();
    model.unload().unwrap();
    model.unload().unwrap();
}

#[test]
fn run_without_load_fails() {
    let p = rane::mil::matmul(32, 32, 32);
    let model = AneModel::compile(&p, &[]).unwrap();
    let input = AneSurface::new(p.input_bytes()).unwrap();
    let output = AneSurface::new(p.output_bytes()).unwrap();
    assert!(model.run(&input, &output).is_err());
}

#[test]
fn multiple_runs_same_model() {
    let p = rane::mil::matmul(32, 32, 32);
    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();

    let input = AneSurface::new(p.input_bytes()).unwrap();
    let output = AneSurface::new(p.output_bytes()).unwrap();

    for _ in 0..10 {
        model.run(&input, &output).unwrap();
    }
}

#[test]
fn matmul_various_sizes() {
    for (ic, oc, seq) in [(64, 64, 64), (64, 128, 64), (128, 64, 64)] {
        let p = rane::mil::matmul(ic, oc, seq);
        let mut model = AneModel::compile(&p, &[]).unwrap();
        model.load().unwrap();

        let input = AneSurface::new(p.input_bytes()).unwrap();
        let output = AneSurface::new(p.output_bytes()).unwrap();
        model.run(&input, &output).unwrap();
    }
}
