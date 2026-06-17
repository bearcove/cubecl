//! M3 parity: the cubek `qa_gemv` shape (warp-per-column W4A16 GEMV — shared
//! codebook LUT, dense 4-bit dequant-on-read, `plane_sum` reduction) run through
//! `Metal4Runtime`, checked against a CPU oracle. This exercises exactly the
//! Metal-4 capabilities cubek needs: simdgroup plane reduction, threadgroup
//! shared memory, and dynamic indexing — proving cubek kernels run on this
//! backend. (The cubek crate itself pins a different cubecl rev; helix-burn's
//! dep graph reconciles that in M4. The kernel logic here mirrors
//! `cubek/qa_matmul.rs`'s plain panel path.)
#![cfg(target_os = "macos")]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_metal4::Metal4Runtime;

/// Warp = one output column. Lane strides over K by the plane width; each lane
/// dequantizes its weight codes on read (4-bit, 8 codes/u32) against a shared
/// 16-entry LUT, multiplies by the activation, and the warp `plane_sum`s.
#[cube(launch_unchecked)]
fn qa_gemv<F: Float>(
    a: &[F],          // [K] activation (M = 1)
    w_codes: &[u32],  // [N, K] dense 4-bit codes (8 per u32, code j at bit 4j)
    w_scales: &[F],   // [N, K/16] per-16 scale
    lut: &[F],        // 16 centroids
    out: &mut [F],    // [N]
    #[comptime] k: u32,
) {
    let lane = UNIT_POS_PLANE;
    let warp = UNIT_POS / PLANE_DIM;
    let n_warps = CUBE_DIM / PLANE_DIM;
    let col = (CUBE_POS_X * n_warps + warp) as usize;
    let kk = k as usize;
    let units = comptime!((k / 16) as usize);

    // Stage the codebook in shared once per cube.
    let mut lut_sh = Shared::<[F]>::new_slice(16usize);
    if UNIT_POS < 16 {
        lut_sh[UNIT_POS as usize] = lut[UNIT_POS as usize];
    }
    sync_cube();

    if col < out.len() {
        let mut partial = F::new(0.0);
        let mut p = lane as usize;
        while p < kk {
            let idx = col * kk + p; // global 4-bit code index
            let code = (w_codes[idx / 8] >> (((idx % 8) * 4) as u32)) & 0xfu32;
            let scale = w_scales[col * units + p / 16];
            partial += a[p] * lut_sh[code as usize] * scale;
            p += PLANE_DIM as usize;
        }
        let total = plane_sum(partial);
        if lane == 0 {
            out[col] = total;
        }
    }
}

const LUT16: [f32; 16] = [
    -2.732, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128, 0.128, 0.388, 0.657, 0.942,
    1.256, 1.618, 2.069, 2.732,
];

#[test]
fn qa_gemv_parity() {
    let client = Metal4Runtime::client(&Default::default());
    let (n, k) = (64usize, 64usize); // K multiple of 32 (warp) and 8 (pack)
    let units = k / 16;

    // Deterministic pseudo-random inputs.
    let mut s = 0xBEEF_1234u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a: Vec<f32> = (0..k).map(|_| (next() % 1000) as f32 / 500.0 - 1.0).collect();
    let codes_u8: Vec<u8> = (0..n * k).map(|_| (next() % 16) as u8).collect();
    let scales: Vec<f32> = (0..n * units).map(|_| 0.2 + (next() % 800) as f32 / 1000.0).collect();

    // Pack codes: 8 nibbles per u32, code j at bit 4j (per cubek dense layout).
    let mut codes = vec![0u32; n * k / 8];
    for (j, &c) in codes_u8.iter().enumerate() {
        codes[j / 8] |= ((c & 0xf) as u32) << ((j % 8) * 4);
    }

    // CPU oracle.
    let mut oracle = vec![0f32; n];
    for col in 0..n {
        let mut acc = 0f32;
        for p in 0..k {
            acc += a[p] * LUT16[codes_u8[col * k + p] as usize] * scales[col * units + p / 16];
        }
        oracle[col] = acc;
    }

    let ah = client.create_from_slice(f32::as_bytes(&a));
    let ch = client.create_from_slice(u32::as_bytes(&codes));
    let sh = client.create_from_slice(f32::as_bytes(&scales));
    let lh = client.create_from_slice(f32::as_bytes(&LUT16));
    let oh = client.empty(n * core::mem::size_of::<f32>());

    let warps_per_cube = 8u32; // CubeDim 256 / plane 32
    let cubes = (n as u32).div_ceil(warps_per_cube);
    unsafe {
        qa_gemv::launch_unchecked::<f32, Metal4Runtime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(ah, k),
            BufferArg::from_raw_parts(ch, n * k / 8),
            BufferArg::from_raw_parts(sh, n * units),
            BufferArg::from_raw_parts(lh, 16),
            BufferArg::from_raw_parts(oh.clone(), n),
            k as u32,
        );
    }
    let got = f32::from_bytes(&client.read_one(oh).unwrap()).to_vec();

    let cmax = oracle.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let mut max_rel = 0f32;
    for col in 0..n {
        let rel = (got[col] - oracle[col]).abs() / (1.0 + cmax);
        max_rel = max_rel.max(rel);
    }
    assert!(max_rel < 1e-4, "qa_gemv parity: max_rel {max_rel} too high");
    println!("cubek qa_gemv parity OK on metal4: N={n} K={k}, max_rel {max_rel:.2e}");
}
