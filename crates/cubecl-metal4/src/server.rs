//! The cubecl `ComputeServer` for Metal 4 — async + batched, mirroring the
//! cubecl-cpu/wgpu shape: a `SchedulerMultiStream` over a per-stream
//! `MemoryManagement<Metal4Storage>` and an **open batch** (one MTL4 command
//! buffer + compute encoder) that many dispatches append to before a single
//! commit on flush/sync/read. Kernels compile through `MslCompiler` (MSL source)
//! and run via the [`Metal4`] primitives.

use std::collections::HashMap;
use std::sync::Arc;

use cubecl_common::{bytes::Bytes, profile::ProfileDuration, stream_id::StreamId};
use cubecl_core::{
    CubeCount, CubeDim, ExecutionMode, MemoryConfiguration, MemoryUsage,
    backtrace::BackTrace,
    future::DynFut,
    ir::MemoryDeviceProperties,
    server::{
        Binding, ComputeServer, CopyDescriptor, IoError, KernelArguments, MetadataBindingInfo,
        ProfileError, ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
        StreamErrorMode,
    },
    zspace::{Shape, Strides, strides},
};
use cubecl_cpp::shared::CompilationOptions;
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    id::KernelId,
    kernel::CompiledKernel,
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryBinding, ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement,
        MemoryManagementOptions,
    },
    storage::{ComputeStorage, ManagedResource},
    stream::{
        StreamFactory,
        scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    },
    timestamp_profiler::TimestampProfiler,
};

use crate::imp::{Batch, Buffer, Metal4, Pipeline};
use crate::storage::{Metal4Resource, Metal4Storage};

pub type Metal4Compiler = cubecl_cpp::MslCompiler;

/// One scheduled task on a Metal 4 stream.
pub enum ScheduleTask {
    /// Host → unified-memory upload into an already-resolved resource.
    Write { resource: Metal4Resource, data: Bytes },
    /// A compiled kernel ready to encode into the open batch.
    Execute {
        pipeline: Arc<Pipeline>,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
        bindings: BindingsResource,
    },
}

impl core::fmt::Debug for ScheduleTask {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Write { .. } => f.debug_struct("Write").finish(),
            Self::Execute {
                cube_dim,
                cube_count,
                ..
            } => f
                .debug_struct("Execute")
                .field("cube_dim", cube_dim)
                .field("cube_count", cube_count)
                .finish(),
        }
    }
}

/// Resolved resources for a kernel launch: the buffer bindings (in slot order)
/// plus the packed metadata/scalars (`info`) that the CppCompiler appends as the
/// last buffer.
#[derive(Debug)]
pub struct BindingsResource {
    pub resources: Vec<Metal4Resource>,
    /// The memory-pool bindings backing `resources`. We bind by raw `gpu_address`
    /// into the MTL4 argument table, so nothing else holds a refcount on these
    /// pool slices — they MUST be retained until the batch commits, otherwise a
    /// dropped tensor's slice is recycled and overwritten while the queued (not
    /// yet committed) kernel still reads it (observed as audio-encoder NaN).
    pub handles: Vec<ManagedMemoryBinding>,
    pub info: MetadataBindingInfo,
}

// ---------------------------------------------------------------------------
// Stream: memory management + the open batch.
// ---------------------------------------------------------------------------

pub struct Metal4Stream {
    ctx: Arc<Metal4>,
    pub(crate) memory_management: MemoryManagement<Metal4Storage>,
    pub(crate) timestamps: TimestampProfiler,
    /// The currently-open batch (lazily opened on the first dispatch, committed
    /// on flush). `None` means no work is pending.
    batch: Option<Batch>,
    /// `info`/metadata buffers created for in-flight dispatches; kept alive until
    /// the batch commits, then cleared.
    transient: Vec<Buffer>,
    /// Memory-pool bindings referenced by dispatches in the open batch. Held so
    /// the pool can't recycle a slice the uncommitted GPU work still reads; freed
    /// on commit.
    in_flight: Vec<ManagedMemoryBinding>,
    errors: Vec<ServerError>,
}

impl core::fmt::Debug for Metal4Stream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Metal4Stream").finish()
    }
}

impl Metal4Stream {
    fn enqueue_task(&mut self, task: ScheduleTask) {
        if !self.is_healthy() {
            return;
        }
        let result = match task {
            ScheduleTask::Write { resource, data } => self.do_write(resource, data),
            ScheduleTask::Execute {
                pipeline,
                cube_dim,
                cube_count,
                bindings,
            } => self.do_execute(pipeline, cube_dim, cube_count, bindings),
        };
        if let Err(e) = result {
            self.errors.push(ServerError::Io(IoError::Unknown {
                description: e,
                backtrace: BackTrace::capture(),
            }));
        }
    }

    fn do_write(&mut self, resource: Metal4Resource, data: Bytes) -> Result<(), String> {
        // Unified memory: write straight into the resource's host pointer. The
        // batch hasn't committed yet, so the GPU sees these bytes when it runs.
        let n = data.len().min(resource.size);
        // SAFETY: `resource.ptr` is valid for `resource.size` bytes (resident
        // unified-memory allocation); `n <= size`.
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), resource.ptr, n) };
        Ok(())
    }

    fn do_execute(
        &mut self,
        pipeline: Arc<Pipeline>,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
        bindings: BindingsResource,
    ) -> Result<(), String> {
        // Buffer slots: each binding's GPU address in order, then the appended
        // metadata/scalars buffer (CppCompiler's `info`).
        let mut addresses: Vec<u64> = bindings.resources.iter().map(|r| r.gpu_address).collect();
        if !bindings.info.data.is_empty() {
            let info_buf =
                self.ctx.buffer_from(bytemuck::cast_slice::<u64, u8>(&bindings.info.data));
            addresses.push(info_buf.gpu_address());
            self.transient.push(info_buf);
        }
        // Pin the bound pool slices until this batch commits (see BindingsResource).
        self.in_flight.extend(bindings.handles);

        if self.batch.is_none() {
            self.batch = Some(self.ctx.open_batch()?);
        }
        let batch = self.batch.as_mut().unwrap();
        self.ctx.batch_dispatch(
            batch,
            &pipeline,
            &addresses,
            (cube_count[0], cube_count[1], cube_count[2]),
            (cube_dim.x, cube_dim.y, cube_dim.z),
            None,
        )
    }

    /// Commit the open batch (if any), blocking until the GPU retires it, and
    /// drop the transient info buffers.
    fn commit(&mut self) -> Result<(), String> {
        if let Some(batch) = self.batch.take() {
            if batch.dispatches() > 0 {
                self.ctx.commit_batch(batch)?;
            }
        }
        self.transient.clear();
        // GPU work retired (commit blocks); the bound slices can now be recycled.
        self.in_flight.clear();
        Ok(())
    }

    pub fn flush(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if let Err(e) = self.commit() {
            self.errors.push(ServerError::Io(IoError::Unknown {
                description: e,
                backtrace: BackTrace::capture(),
            }));
        }
        if mode.flush || !mode.ignore {
            if !mode.ignore && !self.errors.is_empty() {
                return Err(ServerError::ServerUnhealthy {
                    errors: self.errors.clone(),
                    backtrace: BackTrace::capture(),
                });
            }
        }
        Ok(())
    }

    fn is_healthy(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error(&mut self, error: ServerError) {
        self.errors.push(error);
    }

    pub fn empty(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.memory_management.reserve(size)
    }

    pub fn bind(&mut self, reserved: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        self.memory_management.bind(reserved, new, 0).unwrap();
    }

    pub fn get_resource(&mut self, binding: Binding) -> Result<Metal4Resource, IoError> {
        self.memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Synchronous read: commit pending work, then view the resolved resource's
    /// unified memory as bytes.
    pub fn read_sync(&mut self, descriptor: CopyDescriptor) -> Result<Bytes, IoError> {
        self.commit().map_err(|e| IoError::Unknown {
            description: e,
            backtrace: BackTrace::capture(),
        })?;
        let resource = self.get_resource(descriptor.handle)?;
        // SAFETY: dispatch retired (commit blocked); unified memory is coherent.
        let bytes = unsafe { resource.as_bytes() };
        Ok(Bytes::from_bytes_vec(bytes.to_vec()))
    }

    pub fn sync(&mut self) -> Result<(), ServerError> {
        self.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    pub fn start_profile(&mut self) -> Result<ProfilingToken, ServerError> {
        self.sync()?;
        Ok(self.timestamps.start())
    }

    pub fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = self.sync() {
            self.timestamps.error(ProfileError::Server(Box::new(err)));
        }
        self.timestamps.stop(token)
    }

    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.memory_management.mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Scheduler backend + factory.
// ---------------------------------------------------------------------------

pub struct Metal4StreamFactory {
    ctx: Arc<Metal4>,
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
}

impl StreamFactory for Metal4StreamFactory {
    type Stream = Metal4Stream;

    fn create(&mut self) -> Self::Stream {
        let memory_management = MemoryManagement::from_configuration(
            Metal4Storage::new(self.ctx.clone(), self.memory_properties.alignment as usize),
            &self.memory_properties,
            self.memory_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Main Metal4"),
        );
        Metal4Stream {
            ctx: self.ctx.clone(),
            memory_management,
            timestamps: TimestampProfiler::default(),
            batch: None,
            transient: Vec::new(),
            in_flight: Vec::new(),
            errors: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct ScheduledMetal4Backend {
    factory: Metal4StreamFactory,
}

impl core::fmt::Debug for Metal4StreamFactory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Metal4StreamFactory").finish()
    }
}

impl cubecl_runtime::stream::scheduler::SchedulerStreamBackend for ScheduledMetal4Backend {
    type Task = ScheduleTask;
    type Stream = Metal4Stream;
    type Factory = Metal4StreamFactory;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream) {
        stream.enqueue_task(task);
    }

    fn flush(stream: &mut Self::Stream) {
        let _ = stream.flush(StreamErrorMode {
            ignore: true,
            flush: false,
        });
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}

// ---------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------

pub struct Metal4Server {
    ctx: Arc<Metal4>,
    scheduler: SchedulerMultiStream<ScheduledMetal4Backend>,
    utilities: Arc<ServerUtilities<Metal4Server>>,
    pipelines: HashMap<KernelId, (Arc<Pipeline>, CubeDim)>,
    compilation_options: CompilationOptions,
    streams_pool: Vec<StreamId>,
}

impl core::fmt::Debug for Metal4Server {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Metal4Server").finish()
    }
}

impl Metal4Server {
    pub fn new(
        ctx: Arc<Metal4>,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: CompilationOptions,
        utilities: Arc<ServerUtilities<Metal4Server>>,
    ) -> Self {
        let factory = Metal4StreamFactory {
            ctx: ctx.clone(),
            memory_properties,
            memory_config,
            logger: utilities.logger.clone(),
        };
        let backend = ScheduledMetal4Backend { factory };
        let config = CubeClRuntimeConfig::get();
        let scheduler = SchedulerMultiStream::new(
            utilities.logger.clone(),
            backend,
            SchedulerMultiStreamOptions {
                max_streams: config.streaming.max_streams,
                // Generous cap so a decode token (~200 launches) batches into a
                // handful of commits, not one per dispatch.
                max_tasks: 64,
                strategy: SchedulerStrategy::Interleave,
            },
        );
        Self {
            ctx,
            scheduler,
            utilities,
            pipelines: HashMap::new(),
            compilation_options,
            streams_pool: Vec::new(),
        }
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    /// Total queue commits so far (the batching invariant: ≪ dispatch count).
    pub fn commit_count(&self) -> u64 {
        self.ctx.commit_count()
    }

    /// Compile (or fetch from cache) the pipeline for `kernel`.
    fn pipeline(
        &mut self,
        kernel: Box<dyn CubeTask<Metal4Compiler>>,
        mode: ExecutionMode,
    ) -> Result<(Arc<Pipeline>, CubeDim), String> {
        let mut id = kernel.id();
        id.mode(mode);
        if let Some(hit) = self.pipelines.get(&id) {
            return Ok(hit.clone());
        }
        let mut compiler = Metal4Compiler::default();
        let addr = kernel.address_type();
        let compiled: CompiledKernel<Metal4Compiler> = kernel
            .compile(&mut compiler, &self.compilation_options, mode, addr)
            .map_err(|e| format!("MSL codegen failed: {e}"))?;
        let pipeline = self
            .ctx
            .compile(&compiled.source, &compiled.entrypoint_name)?;
        let entry = (Arc::new(pipeline), compiled.cube_dim);
        self.pipelines.insert(id, entry.clone());
        Ok(entry)
    }

    fn prepare_bindings(&mut self, bindings: KernelArguments) -> BindingsResource {
        let mut handles = Vec::with_capacity(bindings.buffers.len());
        let resources = bindings
            .buffers
            .into_iter()
            .map(|binding| {
                // Clone the pool binding so the slice stays reserved until the
                // batch commits (we bind by raw gpu_address, nothing else pins it).
                handles.push(binding.memory.clone());
                let stream = self.scheduler.stream(&binding.stream);
                stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        BindingsResource {
            resources,
            handles,
            info: bindings.info,
        }
    }
}

impl ServerCommunication for Metal4Server {
    const SERVER_COMM_ENABLED: bool = false;
}

impl ComputeServer for Metal4Server {
    type Kernel = Box<dyn CubeTask<Metal4Compiler>>;
    type Storage = Metal4Storage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, _sizes: &[usize], _stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        let reserved = stream.empty(size).unwrap();
        stream.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mut streams = vec![stream_id];
        for d in &descriptors {
            if !streams.contains(&d.handle.stream) {
                streams.push(d.handle.stream);
            }
        }
        self.scheduler.execute_streams(streams);
        let mut out = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            let stream = self.scheduler.stream(&stream_id);
            match stream.read_sync(desc) {
                Ok(b) => out.push(b),
                Err(e) => return Box::pin(async move { Err(e.into()) }),
            }
        }
        Box::pin(async move { Ok(out) })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);
            if contiguous_strides(&desc.shape) != desc.strides {
                stream.error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }
            let resource = match stream.get_resource(desc.handle.clone()) {
                Ok(r) => r,
                Err(e) => {
                    stream.error(ServerError::Io(e));
                    return;
                }
            };
            self.scheduler
                .register(stream_id, ScheduleTask::Write { resource, data }, &[]);
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        self.streams_pool.clear();
        bindings
            .buffers
            .iter()
            .for_each(|b| self.streams_pool.push(b.stream));

        let cube_count = match count {
            CubeCount::Static(x, y, z) => [x, y, z],
            CubeCount::Dynamic(_) => {
                todo!("cubecl-metal4: CubeCount::Dynamic (indirect dispatch) not wired yet")
            }
        };

        let (pipeline, cube_dim) = match self.pipeline(kernel, mode) {
            Ok(v) => v,
            Err(e) => {
                let stream = self.scheduler.stream(&stream_id);
                stream.error(ServerError::Io(IoError::Unknown {
                    description: e,
                    backtrace: BackTrace::capture(),
                }));
                return;
            }
        };

        let bindings = self.prepare_bindings(bindings);
        let task = ScheduleTask::Execute {
            pipeline,
            cube_dim,
            cube_count,
            bindings,
        };
        self.scheduler.register(stream_id, task, &self.streams_pool);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        let result = stream.sync();
        Box::pin(async move { result })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.get_resource(binding)?;
        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.memory_management.memory_usage())
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.scheduler.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.memory_management.cleanup(true);
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.end_profile(token)
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        stream.allocation_mode(mode);
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
