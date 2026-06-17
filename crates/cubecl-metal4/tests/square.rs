//! M1 walking-skeleton verify: `square_array` end-to-end on Metal 4 —
//! alloc → compile MSL → dispatch → read back, asserting `out == in²`.
#![cfg(target_os = "macos")]

use cubecl_metal4::Metal4;

const SQUARE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void square(device const float* in  [[buffer(0)]],
                   device float*       out [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid] * in[gid];
}
"#;

#[test]
fn square_array() {
    let m4 = match Metal4::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping square_array — no Metal 4 here: {e}");
            return;
        }
    };

    let n = 1024usize; // multiple of threadgroup size → exact grid, no OOB
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 100.0).collect();

    let in_buf = m4.buffer_from(&input);
    let out_buf = m4.alloc(n * core::mem::size_of::<f32>());
    let pipe = m4.compile(SQUARE_MSL, "square").expect("compile square");

    let threads = 256u32;
    let groups = (n as u32) / threads;
    m4.dispatch(&pipe, &[&in_buf, &out_buf], (groups, 1, 1), (threads, 1, 1))
        .expect("dispatch");

    let out = unsafe { out_buf.as_slice::<f32>() };
    for (i, &x) in input.iter().enumerate() {
        let expect = x * x;
        assert!(
            (out[i] - expect).abs() <= 1e-3 * (1.0 + expect.abs()),
            "out[{i}] = {} expected {expect}",
            out[i]
        );
    }
    println!("square_array OK ({n} elems) on {}", m4.name());
}

/// M2 verify: per-dispatch GPU timestamps return a sane, non-zero duration.
#[test]
fn square_array_timed() {
    let m4 = match Metal4::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skipping square_array_timed — no Metal 4 here: {e}");
            return;
        }
    };

    let n = 1usize << 20; // 1M elems — enough to be measurable
    let input: Vec<f32> = (0..n).map(|i| (i % 97) as f32).collect();
    let in_buf = m4.buffer_from(&input);
    let out_buf = m4.alloc(n * core::mem::size_of::<f32>());
    let pipe = m4.compile(SQUARE_MSL, "square").expect("compile square");

    let threads = 256u32;
    let groups = (n as u32) / threads;
    let ns = m4
        .dispatch_timed(&pipe, &[&in_buf, &out_buf], (groups, 1, 1), (threads, 1, 1))
        .expect("dispatch_timed");

    // correctness still holds on the timed path
    let out = unsafe { out_buf.as_slice::<f32>() };
    assert_eq!(out[12345], input[12345] * input[12345]);

    // sane: non-zero and under a second for a 1M-elem elementwise kernel
    assert!(ns > 0, "GPU duration was zero");
    assert!(ns < 1_000_000_000, "GPU duration absurdly high: {ns} ns");
    println!(
        "square_array_timed: {n} elems in {ns} ns ({:.3} ms) on {}  (freq-derived GPU time)",
        ns as f64 / 1e6,
        m4.name()
    );
}
