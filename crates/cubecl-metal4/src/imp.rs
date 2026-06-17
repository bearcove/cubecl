//! Metal 4 device plumbing: queue + allocator pool + shared-event fence +
//! residency set (ported/trimmed from bee's `helix-metal4`), plus MTLBuffer
//! storage, MSL pipeline compilation, and one-shot compute dispatch.

use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLAllocation, MTLBuffer,
    MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent,
    MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent,
    MTLSize,
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

impl Buffer {
    /// GPU virtual address, for binding into an `MTL4ArgumentTable`.
    fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
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

        Ok(Self {
            device,
            queue,
            allocators: Mutex::new(vec![first_allocator]),
            shared_event,
            residency_set,
            next_signal: AtomicU64::new(1),
        })
    }

    /// The Metal device name (for logging / device enumeration).
    pub fn name(&self) -> String {
        self.device.name().to_string()
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
        cb.setLabel(Some(&NSString::from_str("cubecl.metal4.dispatch")));
        cb.beginCommandBufferWithAllocator(&allocator);

        let table_desc = MTL4ArgumentTableDescriptor::new();
        table_desc.setMaxBufferBindCount(bindings.len());
        table_desc.setInitializeBindings(true);
        let table = self
            .device
            .newArgumentTableWithDescriptor_error(&table_desc)
            .map_err(|e| format!("argument table creation failed: {e}"))?;
        for (i, buf) in bindings.iter().enumerate() {
            unsafe { table.setAddress_atIndex(buf.gpu_address(), i) };
        }

        let encoder = cb
            .computeCommandEncoder()
            .ok_or("command buffer did not create a compute encoder")?;
        encoder.setComputePipelineState(&pipeline.state);
        encoder.setArgumentTable(Some(&table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        encoder.endEncoding();
        cb.endCommandBuffer();

        // Commit + signal a fresh value + host-wait on the shared event.
        let mut command_ptr = NonNull::from(&*cb);
        let command_ptrs = NonNull::from(&mut command_ptr);
        unsafe { self.queue.commit_count(command_ptrs, 1) };
        let signal = self.next_signal.fetch_add(1, Ordering::Relaxed);
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.shared_event);
        self.queue.signalEvent_value(event, signal);
        let completed = self
            .shared_event
            .waitUntilSignaledValue_timeoutMS(signal, WAIT_TIMEOUT_MS);

        if !completed {
            // GPU may still consume the allocator's encoded commands → leak it.
            return Err(format!(
                "dispatch timed out after {WAIT_TIMEOUT_MS} ms (event at {})",
                self.shared_event.signaledValue()
            ));
        }
        allocator.reset();
        if let Ok(mut pool) = self.allocators.lock() {
            pool.push(allocator);
        }
        Ok(())
    }
}

// SAFETY: `Metal4` is accessed single-threaded per context in M1; the Retained
// Metal objects are internally reference-counted and thread-safe to retain/release.
unsafe impl Send for Metal4 {}
