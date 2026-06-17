//! `Runtime` + `DeviceService` for the native Metal 4 backend.

use std::sync::Arc;

use cubecl_common::{device::DeviceService, profile::TimingMethod};
use cubecl_core::{
    MemoryConfiguration, Runtime,
    client::ComputeClient,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        DeviceProperties, ElemType, FloatKind, HardwareProperties, MemoryDeviceProperties,
        TargetProperties, VectorSize, features::Plane,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::{
    DialectWmmaCompiler, register_supported_types,
    metal::{MslDialect, arch::MetalArchitecture},
    shared::{CompilationOptions, register_wmma_features},
};
use cubecl_runtime::{allocator::ContiguousMemoryLayoutPolicy, logging::ServerLogger};

use crate::device::Metal4Device;
use crate::imp::Metal4;
use crate::server::{Metal4Compiler, Metal4Server};

#[derive(Debug, Clone)]
pub struct Metal4Runtime;

/// Apple simdgroup width (the cubecl "plane" size) is 32 across Apple GPUs.
const PLANE_SIZE: u32 = 32;

impl DeviceService for Metal4Server {
    fn init(_device_id: DeviceId) -> Self {
        let ctx = Arc::new(Metal4::new().expect("Metal 4 device (needs macOS 26+ on Apple GPU)"));
        let logger = Arc::new(ServerLogger::default());

        let (cd_x, cd_y, cd_z) = ctx.max_threads_per_threadgroup();
        let max_units_per_cube = cd_x.max(cd_y).max(cd_z);
        // Real per-cube threadgroup memory (Apple M-series expose well above the
        // old 32 KiB floor). Hardcoding 32 KiB starved the matmul autotune of its
        // larger-tile candidates, forcing the scalar fallbacks that miscompute the
        // codebook lhs. Query the device for the true limit.
        let max_shared_memory_size = ctx.max_threadgroup_memory();
        let working_set = ctx.recommended_working_set_size().max(1 << 30);

        // Apple GPUs have simdgroup-matrix (cooperative-matrix / cmma) units —
        // the same ones cubecl-cpp's MSL dialect targets for cubecl-wgpu's cmma
        // matmul. Advertise them so the matmul autotune gets tile/cmma candidates
        // (like CUDA's WMMA) instead of only the scalar unit/vecmat/naive ones,
        // which mishandle the packed-u32 codebook lhs under fusion (cast.rs:28).
        let wmma_combinations = MslDialect::supported_wmma_combinations(&MetalArchitecture::Metal3);
        let min_tensor_cores_dim = if wmma_combinations.is_empty() { None } else { Some(8) };

        let topology = HardwareProperties {
            load_width: 128,
            plane_size_min: PLANE_SIZE,
            plane_size_max: PLANE_SIZE,
            max_bindings: crate::device::METAL4_MAX_BINDINGS,
            max_shared_memory_size,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube,
            max_cube_dim: (cd_x, cd_y, cd_z),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim,
            num_cpu_cores: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };

        const ALIGNMENT: u64 = 256;
        let mem_properties = MemoryDeviceProperties {
            max_page_size: (working_set / 4).max(1 << 28),
            alignment: ALIGNMENT,
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            topology,
            // Real per-kernel GPU timing comes from the MTL4 counter heap.
            TimingMethod::Device,
        );
        register_supported_types(&mut device_props);
        // Register the simdgroup-matrix combinations into `features.matmul.cmma`
        // so the matmul autotune can pick cmma tile candidates.
        register_wmma_features(wmma_combinations, &mut device_props);
        device_props.register_type_usage(
            ElemType::Float(FloatKind::F16),
            cubecl_core::ir::features::TypeUsage::all(),
        );
        // Apple simdgroups support the plane (subgroup) shuffle/reduction ops the
        // cubek kernels use.
        device_props.features.plane.insert(Plane::Ops);

        let mut comp_opts = CompilationOptions::default();
        comp_opts.warp_size = PLANE_SIZE;

        let utilities = ServerUtilities::new(
            device_props,
            logger,
            (),
            ContiguousMemoryLayoutPolicy::new(ALIGNMENT as usize),
        );

        Metal4Server::new(
            ctx,
            mem_properties,
            MemoryConfiguration::default(),
            comp_opts,
            Arc::new(utilities),
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}

impl Runtime for Metal4Runtime {
    type Compiler = Metal4Compiler;
    type Server = Metal4Server;
    type Device = Metal4Device;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "metal4"
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u32::MAX, u32::MAX, u32::MAX)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as cubecl_core::server::ComputeServer>::Info,
    ) -> Vec<DeviceId> {
        vec![DeviceId {
            type_id: 0,
            index_id: 0,
        }]
    }
}
