//! Integration tests — no ANE hardware required
//! These tests verify fp16 conversion, MIL generation, and IOSurface creation.
//! Run on any macOS (including Intel CI runners).

use rane::{f32_to_fp16, fp16_to_f32, AneSurface};

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
    assert!(s.size() >= 64 * 128 * 2);
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

#[test]
fn surface_bounds_check() {
    assert!(AneSurface::new(0).is_err());
    assert!(AneSurface::new(512 * 1024 * 1024).is_err()); // > 256 MB limit
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

    rane::cvt_f32_f16(&mut fp16, &src);
    rane::cvt_f16_f32(&mut dst, &fp16);

    for i in 0..n {
        let err = (dst[i] - src[i]).abs();
        assert!(
            err < 0.1,
            "bulk conversion error at {}: {} vs {}",
            i,
            dst[i],
            src[i]
        );
    }
}

// ── MIL program tests ──

#[test]
fn mil_matmul_shape() {
    let p = rane::mil::matmul(64, 64, 64);
    assert_eq!(p.input_shape(), (64, 128));
    assert_eq!(p.output_shape(), (64, 64));
    assert_eq!(p.input_bytes(), 64 * 128 * 2);
    assert_eq!(p.output_bytes(), 64 * 64 * 2);
}

#[test]
fn mil_matmul_text_is_valid() {
    let p = rane::mil::matmul(32, 16, 8);
    let text = p.as_str();
    assert!(text.contains("program(1.3)"));
    assert!(text.contains("func main"));
    assert!(text.contains("matmul"));
}
