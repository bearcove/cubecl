//! M3 verify: real `#[cube]` kernels run through the full cubecl launch path on
//! `Metal4Runtime` (compile via MslCompiler → batched dispatch → readback), and
//! the batching invariant holds (commits ≪ dispatches).
#![cfg(target_os = "macos")]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_metal4::{Metal4Runtime, global_commit_count};

#[cube(launch_unchecked)]
fn square_array<F: Float>(input: &[F], output: &mut [F]) {
    if ABSOLUTE_POS < input.len() {
        let i = ABSOLUTE_POS as usize;
        output[i] = input[i] * input[i];
    }
}

#[cube(launch_unchecked)]
fn add_one<F: Float>(data: &mut [F]) {
    if ABSOLUTE_POS < data.len() {
        let i = ABSOLUTE_POS as usize;
        data[i] += F::new(1.0);
    }
}

/// The full cubecl launch path on Metal 4: out == in².
#[test]
fn cube_square_array() {
    let client = Metal4Runtime::client(&Default::default());
    let n = 1024usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 50.0).collect();
    let ih = client.create_from_slice(f32::as_bytes(&input));
    let oh = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        square_array::launch_unchecked::<f32, Metal4Runtime>(
            &client,
            CubeCount::Static((n / 256) as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(ih, n),
            BufferArg::from_raw_parts(oh.clone(), n),
        );
    }
    let out = f32::from_bytes(&client.read_one(oh).unwrap()).to_vec();
    for (i, &x) in input.iter().enumerate() {
        let expect = x * x;
        assert!(
            (out[i] - expect).abs() <= 1e-3 * (1.0 + expect.abs()),
            "out[{i}] = {} expected {expect}",
            out[i]
        );
    }
    println!("cube square_array OK ({n} elems) on metal4");
}

/// Batching proof: 100 dependent `add_one` dispatches (each reads the previous
/// result) on the same buffer, read ONCE. Asserts (a) correctness — every
/// element advanced by exactly 100 (so the intra-encoder barrier is real), and
/// (b) the queue committed far fewer than 100 times (batched, not one-per-dispatch).
#[test]
fn cube_batching_proof() {
    let client = Metal4Runtime::client(&Default::default());
    let n = 256usize;
    let base: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let h = client.create_from_slice(f32::as_bytes(&base));

    let iters = 100u32;
    let commits_before = global_commit_count();
    for _ in 0..iters {
        unsafe {
            add_one::launch_unchecked::<f32, Metal4Runtime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n as u32),
                BufferArg::from_raw_parts(h.clone(), n),
            );
        }
    }
    let out = f32::from_bytes(&client.read_one(h).unwrap()).to_vec();
    let commits = global_commit_count() - commits_before;

    for (i, &b) in base.iter().enumerate() {
        assert_eq!(out[i], b + iters as f32, "elem {i}: dependency/barrier broken");
    }
    assert!(
        commits < iters as u64,
        "not batched: {commits} commits for {iters} dispatches"
    );
    println!("cube batching proof OK: {iters} dependent dispatches in {commits} commit(s)");
}
