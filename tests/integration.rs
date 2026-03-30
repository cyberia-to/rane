//! Integration tests for the ane driver
//! Requires macOS + Apple Silicon with ANE access

use ane::{f32_to_fp16, fp16_to_f32, AneModel, AneSurface};

// ── Surface tests ──

#[test]
fn surface_create_and_size() {
    let s = AneSurface::new(4096).unwrap();
    assert!(s.size() >= 4096);
    assert!(s.id() > 0);
}

#[test]
fn surface_with_shape() {
    let s = AneSurface::with_shape(64, 128).unwrap();
    assert!(s.size() >= 64 * 128 * 2); // fp16
}

#[test]
fn surface_write_read_roundtrip() {
    let s = AneSurface::new(256 * 2).unwrap();
    s.with_data_mut(|d| {
        for i in 0..256 {
            d[i] = f32_to_fp16(i as f32);
        }
    });
    s.with_data(|d| {
        for i in 0..256 {
            let val = fp16_to_f32(d[i]);
            assert!((val - i as f32).abs() < 0.5, "mismatch at {}: {}", i, val);
        }
    });
}

// ── fp16 tests ──

#[test]
fn fp16_roundtrip_exact() {
    for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0] {
        let h = f32_to_fp16(v);
        let back = fp16_to_f32(h);
        assert_eq!(back, v, "roundtrip failed for {}", v);
    }
}

#[test]
fn fp16_bulk_conversion() {
    let n = 1024;
    let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut fp16 = vec![0u16; n];
    let mut dst = vec![0.0f32; n];

    ane::cvt_f32_f16(&mut fp16, &src);
    ane::cvt_f16_f32(&mut dst, &fp16);

    for i in 0..n {
        let err = (dst[i] - src[i]).abs();
        assert!(err < 0.1, "bulk conversion error at {}: {} vs {}", i, dst[i], src[i]);
    }
}

// ── MIL program tests ──

#[test]
fn mil_matmul_shape() {
    let p = ane::mil::matmul(64, 64, 64);
    assert_eq!(p.input_shape(), (64, 128)); // ic=64, sp=seq+oc=128
    assert_eq!(p.output_shape(), (64, 64));
    assert_eq!(p.input_bytes(), 64 * 128 * 2);
    assert_eq!(p.output_bytes(), 64 * 64 * 2);
}

#[test]
fn mil_matmul_text_is_valid() {
    let p = ane::mil::matmul(32, 16, 8);
    let text = p.as_str();
    assert!(text.contains("program(1.3)"));
    assert!(text.contains("func main"));
    assert!(text.contains("matmul"));
}

// ── Model lifecycle tests ──

#[test]
fn compile_load_unload() {
    let p = ane::mil::matmul(64, 64, 64);
    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();
    model.unload().unwrap();
}

#[test]
fn compile_load_run_identity() {
    let ic = 64;
    let oc = 64;
    let seq = 64;
    let p = ane::mil::matmul(ic, oc, seq);

    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();

    let input = AneSurface::new(p.input_bytes()).unwrap();
    let output = AneSurface::new(p.output_bytes()).unwrap();

    // activations = 1.0, weights = identity
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
    let p = ane::mil::matmul(32, 32, 32);
    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();
    model.unload().unwrap();
    model.unload().unwrap(); // second unload should be no-op
}

#[test]
fn run_without_load_fails() {
    let p = ane::mil::matmul(32, 32, 32);
    let model = AneModel::compile(&p, &[]).unwrap();
    let input = AneSurface::new(p.input_bytes()).unwrap();
    let output = AneSurface::new(p.output_bytes()).unwrap();
    assert!(model.run(&input, &output).is_err());
}

#[test]
fn multiple_runs_same_model() {
    let ic = 32;
    let oc = 32;
    let seq = 32;
    let p = ane::mil::matmul(ic, oc, seq);
    let mut model = AneModel::compile(&p, &[]).unwrap();
    model.load().unwrap();

    let input = AneSurface::new(p.input_bytes()).unwrap();
    let output = AneSurface::new(p.output_bytes()).unwrap();

    for _ in 0..10 {
        model.run(&input, &output).unwrap();
    }
}

// ── Different matmul sizes ──

#[test]
fn matmul_various_sizes() {
    // ANE has size constraints — shapes must meet hardware alignment requirements
    for (ic, oc, seq) in [(64, 64, 64), (64, 128, 64), (128, 64, 64)] {
        let p = ane::mil::matmul(ic, oc, seq);
        let mut model = AneModel::compile(&p, &[]).unwrap();
        model.load().unwrap();

        let input = AneSurface::new(p.input_bytes()).unwrap();
        let output = AneSurface::new(p.output_bytes()).unwrap();
        model.run(&input, &output).unwrap();
    }
}
