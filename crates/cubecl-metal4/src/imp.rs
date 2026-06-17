//! Metal 4 device plumbing: queue + allocator pool + shared-event fence +
//! residency set (ported/trimmed from bee's `helix-metal4`), plus MTLBuffer
//! storage, MSL pipeline compilation, and one-shot compute dispatch.

use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4CounterHeap,
    MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTL4TimestampGranularity, MTL4VisibilityOptions,
    MTLAllocation, MTLBuffer, MTLCompileOptions, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLSize, MTLStages,
};

/// Host-side wait budget for a dispatch to retire (generous; turns a wedged
/// queue into an error instead of a hang).
const WAIT_TIMEOUT_MS: u64 = 30_000;

/// A native Metal 4 runtime context on one device.
pub struct Metal4 {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Pool of command allocators; one is borrowed per command buffer and reset
    /// back into the pool after the GPU retires the work.
    allocators: Mutex<Vec<Retained<ProtocolObject<dyn MTL4CommandAllocator>>>>,
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Strictly-increasing signal value (never 0 — a shared event starts at 0).
    next_signal: AtomicU64,
    /// GPU timestamp ticks per second (for `dispatch_timed` ns conversion).
    frequency_hz: u64,
    /// Byte stride of one timestamp counter-heap entry on this device.
    ts_entry_size: usize,
    /// Total number of `commit` calls on the queue. Exposed so tests can assert
    /// the batched server commits far fewer than once-per-dispatch.
    commit_count: AtomicU64,
}

/// An **open** command buffer + compute encoder accumulating many dispatches
/// before a single commit. This is the batched-launch lifecycle the server
/// drives: [`Metal4::open_batch`] begins it once, [`Batch::encode_dispatch`]
/// appends dispatches (each gets its own argument table but no commit), and
/// [`Metal4::commit_batch`] ends + commits + signals exactly once.
pub struct Batch {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    cb: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    /// Number of dispatches encoded into this batch so far.
    dispatches: usize,
}

impl Batch {
    /// Number of dispatches encoded into this batch so far.
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }
}

/// An `MTLBuffer` allocation. On Apple Silicon `StorageModeShared` is unified
/// memory, so [`Buffer::as_slice`]/[`as_mut_slice`] read/write it directly.
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

/// A compiled MSL compute pipeline.
pub struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// SAFETY: `Buffer`/`Pipeline` wrap reference-counted Metal objects that are
// thread-safe to retain/release; all *use* is serialized under the cubecl
// channel mutex. These impls let them live in the (Send) storage/pipeline cache.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Buffer {
    /// GPU virtual address, for binding into an `MTL4ArgumentTable`.
    pub(crate) fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
    }

    /// Raw pointer to the shared (unified-memory) buffer contents, for host
    /// read/write by the `ComputeStorage` layer. Valid for `len()` bytes.
    pub(crate) fn contents_ptr(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }

    /// Byte length of the allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// View the buffer contents as `&[T]` (shared storage; host-coherent on
    /// Apple Silicon after a completed dispatch).
    ///
    /// # Safety
    /// The caller asserts the buffer holds a valid, initialized `[T]` of the
    /// returned length and that no GPU work is concurrently writing it.
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let ptr = self.raw.contents().as_ptr() as *const T;
        unsafe { core::slice::from_raw_parts(ptr, self.len / core::mem::size_of::<T>()) }
    }

    /// Write `data` into the buffer (host → unified memory).
    pub fn write<T: Copy>(&self, data: &[T]) {
        let bytes = core::mem::size_of_val(data);
        assert!(bytes <= self.len, "write {bytes} > buffer {}", self.len);
        let dst = self.raw.contents().as_ptr() as *mut u8;
        let src = data.as_ptr() as *const u8;
        unsafe { core::ptr::copy_nonoverlapping(src, dst, bytes) };
    }
}

impl Metal4 {
    /// Create a Metal 4 context on the system-default device, or report why it
    /// can't (no Metal device, or the OS predates Metal 4 — needs macOS 26+).
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no system-default MTLDevice")?;
        let queue = device
            .newMTL4CommandQueue()
            .ok_or("device did not create an MTL4CommandQueue (needs macOS 26+)")?;
        let first_allocator = device
            .newCommandAllocator()
            .ok_or("device did not create an MTL4CommandAllocator")?;
        let shared_event = device
            .newSharedEvent()
            .ok_or("device did not create an MTLSharedEvent")?;

        let desc = MTLResidencySetDescriptor::new();
        desc.setLabel(Some(&NSString::from_str("cubecl.metal4.residency")));
        unsafe { desc.setInitialCapacity(128) };
        let residency_set = device
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| format!("device did not create an MTLResidencySet: {e}"))?;
        residency_set.requestResidency();
        queue.addResidencySet(&residency_set);

        let frequency_hz = device.queryTimestampFrequency();
        let ts_entry_size = device.sizeOfCounterHeapEntry(MTL4CounterHeapType::Timestamp);

        Ok(Self {
            device,
            queue,
            allocators: Mutex::new(vec![first_allocator]),
            shared_event,
            residency_set,
            next_signal: AtomicU64::new(1),
            frequency_hz,
            ts_entry_size,
            commit_count: AtomicU64::new(0),
        })
    }

    /// Total queue commits so far. The batched server commits once per flush
    /// (many dispatches), so `commit_count() ≪ dispatch_count` — the batching
    /// invariant the M3 proof test asserts.
    pub fn commit_count(&self) -> u64 {
        self.commit_count.load(Ordering::Relaxed)
    }

    /// The Metal device name (for logging / device enumeration).
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Max threads per threadgroup `(x, y, z)` for this device.
    pub fn max_threads_per_threadgroup(&self) -> (u32, u32, u32) {
        let s = self.device.maxThreadsPerThreadgroup();
        (s.width as u32, s.height as u32, s.depth as u32)
    }

    /// Largest single `MTLBuffer` this device can allocate, in bytes.
    pub fn max_buffer_length(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    /// Recommended working-set size (≈ usable GPU memory) in bytes.
    pub fn recommended_working_set_size(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// Allocate a shared-storage buffer of `bytes` and register it resident.
    pub fn alloc(&self, bytes: usize) -> Buffer {
        let raw = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .expect("MTLBuffer allocation failed");
        // Argument tables bind raw GPU addresses with no implicit residency, so
        // every buffer the GPU may touch must be registered in the queue's set.
        let alloc: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(&*raw);
        self.residency_set.addAllocation(alloc);
        self.residency_set.commit();
        Buffer { raw, len: bytes }
    }

    /// Allocate a buffer initialized from `data`.
    pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Buffer {
        let buf = self.alloc(core::mem::size_of_val(data));
        buf.write(data);
        buf
    }

    /// Compile MSL `source` and build a compute pipeline for entry point `name`.
    pub fn compile(&self, source: &str, name: &str) -> Result<Pipeline, String> {
        let opts = MTLCompileOptions::new();
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
            .map_err(|e| format!("MSL compile failed: {e}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| format!("entry point `{name}` not found in compiled library"))?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("pipeline creation failed: {e}"))?;
        Ok(Pipeline { state })
    }

    /// Dispatch `pipeline` with `bindings` bound to argument-table slots
    /// `0..bindings.len()`, `groups` threadgroups of `threads` each, and block
    /// until the GPU retires the work.
    pub fn dispatch(
        &self,
        pipeline: &Pipeline,
        bindings: &[&Buffer],
        groups: (u32, u32, u32),
        threads: (u32, u32, u32),
    ) -> Result<(), String> {
        let addrs: Vec<u64> = bindings.iter().map(|b| b.gpu_address()).collect();
        self.dispatch_inner(pipeline, &addrs, groups, threads, None)
    }

    /// Dispatch `pipeline` binding the given raw GPU `addresses` to argument-table
    /// slots `0..addresses.len()` and block until the GPU retires the work.
    ///
    /// This is the binding model the cubecl launch path needs: the server gathers
    /// each storage resource's `gpu_address + offset` (and an optional packed
    /// scalar/metadata buffer) and hands them over in slot order. The buffers
    /// backing those addresses must already be resident (every [`Buffer`] from
    /// [`alloc`](Self::alloc) is).
    pub fn dispatch_addresses(
        &self,
        pipeline: &Pipeline,
        addresses: &[u64],
        groups: (u32, u32, u32),
        threads: (u32, u32, u32),
    ) -> Result<(), String> {
        self.dispatch_inner(pipeline, addresses, groups, threads, None)
    }

    /// Like [`dispatch`](Self::dispatch) but brackets the dispatch with a
    /// **Metal 4 counter-heap timestamp pair** (`Precise` granularity) and
    /// returns the real on-GPU kernel duration in nanoseconds.
    pub fn dispatch_timed(
        &self,
        pipeline: &Pipeline,
        bindings: &[&Buffer],
        groups: (u32, u32, u32),
        threads: (u32, u32, u32),
    ) -> Result<u64, String> {
        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        unsafe { desc.setCount(2) };
        let heap = self
            .device
            .newCounterHeapWithDescriptor_error(&desc)
            .map_err(|e| format!("counter heap creation failed: {e}"))?;
        unsafe { heap.invalidateCounterRange(NSRange { location: 0, length: 2 }) };

        let addrs: Vec<u64> = bindings.iter().map(|b| b.gpu_address()).collect();
        self.dispatch_inner(pipeline, &addrs, groups, threads, Some(&heap))?;

        // Resolve the two timestamps and convert ticks → ns.
        let data = unsafe { heap.resolveCounterRange(NSRange { location: 0, length: 2 }) }
            .ok_or("timestamp resolve returned no data")?;
        let mut bytes = vec![0u8; data.length()];
        if !bytes.is_empty() {
            unsafe {
                data.getBytes_length(
                    NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).unwrap(),
                    bytes.len(),
                );
            }
        }
        let read = |i: usize| -> Result<u64, String> {
            let start = i * self.ts_entry_size;
            let raw = bytes
                .get(start..start + 8)
                .ok_or("timestamp entry out of range")?;
            Ok(u64::from_ne_bytes(raw.try_into().unwrap()))
        };
        let begin = read(0)?;
        let end = read(1)?;
        let ticks = end.saturating_sub(begin);
        Ok(if self.frequency_hz > 0 {
            ((ticks as u128) * 1_000_000_000 / self.frequency_hz as u128) as u64
        } else {
            ticks
        })
    }

    // ---- batched submission: open a command buffer, append many dispatches,
    // commit once. This is the competitive path the cubecl server drives; the
    // one-shot `dispatch*` above are thin wrappers over it (open→encode→commit).

    /// Begin a batch: borrow an allocator, open a command buffer + one compute
    /// encoder that subsequent `batch_dispatch` calls append to.
    pub fn open_batch(&self) -> Result<Batch, String> {
        let allocator = {
            let pooled = self
                .allocators
                .lock()
                .map_err(|_| "allocator pool poisoned")?
                .pop();
            match pooled {
                Some(a) => a,
                None => self
                    .device
                    .newCommandAllocator()
                    .ok_or("device did not create an MTL4CommandAllocator")?,
            }
        };
        let cb = self
            .device
            .newCommandBuffer()
            .ok_or("device did not create an MTL4CommandBuffer")?;
        cb.setLabel(Some(&NSString::from_str("cubecl.metal4.batch")));
        cb.beginCommandBufferWithAllocator(&allocator);
        let encoder = cb
            .computeCommandEncoder()
            .ok_or("command buffer did not create a compute encoder")?;
        Ok(Batch {
            allocator,
            cb,
            encoder,
            dispatches: 0,
        })
    }

    /// Append one dispatch to an open `batch`. A conservative compute→compute
    /// barrier precedes every dispatch after the first, so a kernel that reads a
    /// previous kernel's output in the same batch sees its writes (MTL4 does no
    /// automatic hazard tracking). `timestamps` writes a begin/end pair around
    /// the dispatch into the heap at `(begin, end)` when provided.
    pub fn batch_dispatch(
        &self,
        batch: &mut Batch,
        pipeline: &Pipeline,
        addresses: &[u64],
        groups: (u32, u32, u32),
        threads: (u32, u32, u32),
        timestamps: Option<(&ProtocolObject<dyn MTL4CounterHeap>, usize, usize)>,
    ) -> Result<(), String> {
        let table_desc = MTL4ArgumentTableDescriptor::new();
        table_desc.setMaxBufferBindCount(addresses.len().max(1));
        table_desc.setInitializeBindings(true);
        let table = self
            .device
            .newArgumentTableWithDescriptor_error(&table_desc)
            .map_err(|e| format!("argument table creation failed: {e}"))?;
        for (i, &addr) in addresses.iter().enumerate() {
            unsafe { table.setAddress_atIndex(addr, i) };
        }

        let enc = &batch.encoder;
        if batch.dispatches > 0 {
            // Intra-pass RAW/WAR guard between dispatches sharing one encoder:
            // subsequent Dispatch-stage work waits for prior Dispatch-stage work
            // in THIS encoder, with device-visible cache flush. (MTL4 does no
            // automatic hazard tracking; this is the documented intra-pass form.)
            enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
        }
        enc.setComputePipelineState(&pipeline.state);
        enc.setArgumentTable(Some(&table));
        if let Some((heap, begin, _)) = timestamps {
            unsafe {
                enc.writeTimestampWithGranularity_intoHeap_atIndex(
                    MTL4TimestampGranularity::Precise,
                    heap,
                    begin,
                )
            };
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.0 as usize,
                height: groups.1 as usize,
                depth: groups.2 as usize,
            },
            MTLSize {
                width: threads.0 as usize,
                height: threads.1 as usize,
                depth: threads.2 as usize,
            },
        );
        if let Some((heap, _, end)) = timestamps {
            unsafe {
                enc.writeTimestampWithGranularity_intoHeap_atIndex(
                    MTL4TimestampGranularity::Precise,
                    heap,
                    end,
                )
            };
        }
        batch.dispatches += 1;
        Ok(())
    }

    /// End the batch's encoder + command buffer, commit ONCE, signal, and host-
    /// wait for the GPU to retire it. Bumps `commit_count` (one per batch).
    pub fn commit_batch(&self, batch: Batch) -> Result<(), String> {
        batch.encoder.endEncoding();
        batch.cb.endCommandBuffer();

        let mut command_ptr = NonNull::from(&*batch.cb);
        let command_ptrs = NonNull::from(&mut command_ptr);
        unsafe { self.queue.commit_count(command_ptrs, 1) };
        self.commit_count.fetch_add(1, Ordering::Relaxed);
        let signal = self.next_signal.fetch_add(1, Ordering::Relaxed);
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.shared_event);
        self.queue.signalEvent_value(event, signal);
        let completed = self
            .shared_event
            .waitUntilSignaledValue_timeoutMS(signal, WAIT_TIMEOUT_MS);
        if !completed {
            // GPU may still consume the allocator's encoded commands → leak it.
            return Err(format!(
                "batch commit timed out after {WAIT_TIMEOUT_MS} ms (event at {})",
                self.shared_event.signaledValue()
            ));
        }
        batch.allocator.reset();
        if let Ok(mut pool) = self.allocators.lock() {
            pool.push(batch.allocator);
        }
        Ok(())
    }

    fn dispatch_inner(
        &self,
        pipeline: &Pipeline,
        addresses: &[u64],
        groups: (u32, u32, u32),
        threads: (u32, u32, u32),
        timestamps: Option<&ProtocolObject<dyn MTL4CounterHeap>>,
    ) -> Result<(), String> {
        let mut batch = self.open_batch()?;
        let ts = timestamps.map(|h| (h, 0usize, 1usize));
        self.batch_dispatch(&mut batch, pipeline, addresses, groups, threads, ts)?;
        self.commit_batch(batch)
    }
}

// SAFETY: `Metal4` is accessed single-threaded per context (the cubecl channel
// serializes all server access under a mutex); the Retained Metal objects are
// internally reference-counted and thread-safe to retain/release. `Sync` lets it
// sit behind the `Arc` shared between the storage and the server.
unsafe impl Send for Metal4 {}
unsafe impl Sync for Metal4 {}
