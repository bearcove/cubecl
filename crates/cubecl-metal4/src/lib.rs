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
