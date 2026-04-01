//! Hardware tests — require macOS + Apple Silicon with ANE.
//! ANE sandbox extension is not re-entrant — tests serialize via ANE_LOCK.
//! Skipped on CI (Intel runners have no ANE).

use rane::{f32_to_fp16, fp16_to_f32, Buffer, Program};
use std::sync::Mutex;

static ANE_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn compile_load_unload() {
    let _g = ANE_LOCK.lock().unwrap();
    let p = rane::mil::matmul(64, 64, 64);
    let mut model = Program::compile(&p, &[]).unwrap();
    model.load().unwrap();
    model.unload().unwrap();
}

#[test]
fn compile_load_run_identity() {
    let _g = ANE_LOCK.lock().unwrap();
    let ic = 64;
    let oc = 64;
    let seq = 64;
    let p = rane::mil::matmul(ic, oc, seq);

    let mut model = Program::compile(&p, &[]).unwrap();
    model.load().unwrap();

    let input = Buffer::new(p.input_size()).unwrap();
    let output = Buffer::new(p.output_size()).unwrap();

    input.write(|d| {
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

    output.read(|d| {
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
    let _g = ANE_LOCK.lock().unwrap();
    let p = rane::mil::matmul(32, 32, 32);
    let mut model = Program::compile(&p, &[]).unwrap();
    model.load().unwrap();
    model.unload().unwrap();
    model.unload().unwrap();
}

#[test]
fn run_without_load_fails() {
    let _g = ANE_LOCK.lock().unwrap();
    let p = rane::mil::matmul(32, 32, 32);
    let model = Program::compile(&p, &[]).unwrap();
    let input = Buffer::new(p.input_size()).unwrap();
    let output = Buffer::new(p.output_size()).unwrap();
    assert!(model.run(&input, &output).is_err());
}

#[test]
fn multiple_runs_same_model() {
    let _g = ANE_LOCK.lock().unwrap();
    let p = rane::mil::matmul(32, 32, 32);
    let mut model = Program::compile(&p, &[]).unwrap();
    model.load().unwrap();

    let input = Buffer::new(p.input_size()).unwrap();
    let output = Buffer::new(p.output_size()).unwrap();

    for _ in 0..10 {
        model.run(&input, &output).unwrap();
    }
}

#[test]
fn matmul_various_sizes() {
    let _g = ANE_LOCK.lock().unwrap();
    for (ic, oc, seq) in [(64, 64, 64), (64, 128, 64), (128, 64, 64)] {
        let p = rane::mil::matmul(ic, oc, seq);
        let mut model = Program::compile(&p, &[]).unwrap();
        model.load().unwrap();

        let input = Buffer::new(p.input_size()).unwrap();
        let output = Buffer::new(p.output_size()).unwrap();
        model.run(&input, &output).unwrap();
    }
}
