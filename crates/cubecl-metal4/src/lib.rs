//! Native **Metal 4** runtime for CubeCL — no wgpu, no Metal 3.
//!
//! This is the walking skeleton (milestone M1): a [`Metal4`] context that owns
//! the `MTL4CommandQueue` + allocator pool + shared-event fence + residency set,
//! allocates `MTLBuffer` storage, compiles MSL into a compute pipeline, and
//! dispatches it — all through the Metal 4 command-buffer API that bee's
//! `helix-metal4` proved out. The cubecl `Runtime`/`ComputeServer` trait impls
//! (so `#[cube]` kernels and Burn run on this) land in later milestones; this
//! layer is the device plumbing they'll sit on.
//!
//! Everything is Apple-only; on other targets the crate is intentionally empty
//! so the cubecl workspace (`members = ["crates/*"]`) still builds on Linux/CUDA.

#[cfg(target_vendor = "apple")]
mod device;
#[cfg(target_vendor = "apple")]
mod imp;
#[cfg(target_vendor = "apple")]
mod runtime;
#[cfg(target_vendor = "apple")]
mod server;
#[cfg(target_vendor = "apple")]
mod stax_lane;
#[cfg(target_vendor = "apple")]
mod storage;

#[cfg(target_vendor = "apple")]
pub use device::*;
#[cfg(target_vendor = "apple")]
pub use imp::*;
#[cfg(target_vendor = "apple")]
pub use runtime::*;
#[cfg(target_vendor = "apple")]
pub use server::*;
#[cfg(target_vendor = "apple")]
pub use storage::*;

// Golden-vector kernel suite from cubecl-core/std, run against the native Metal 4
// runtime. Same wiring as cubecl-cuda/cubecl-wgpu — `testgen_all!` brings reduce,
// softmax, matmul, normalization, etc. to bear so a divergence/NaN is localized to
// a single op (the audio-encoder NaN bisect).
#[cfg(all(test, target_vendor = "apple"))]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::Metal4Runtime;

    pub use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
}
