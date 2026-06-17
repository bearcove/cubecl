//! `ComputeStorage` over Metal 4 `MTLBuffer`s.
//!
//! Apple Silicon is unified memory, so we keep one `StorageModeShared`
//! `MTLBuffer` per [`StorageId`] (allocated through the shared [`Metal4`]
//! context, which also registers it resident) and expose sub-ranges via
//! [`StorageHandle`] offsets. A [`Metal4Resource`] carries the buffer's GPU
//! address (for argument-table binding on the launch path) plus a host pointer
//! into the same unified allocation (for `read`/`write`).

use std::collections::HashMap;
use std::sync::Arc;

use cubecl_core::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};

use crate::imp::{Buffer, Metal4};

/// Buffer storage backed by Metal 4 `MTLBuffer`s on one device.
pub struct Metal4Storage {
    ctx: Arc<Metal4>,
    memory: HashMap<StorageId, Buffer>,
    deallocations: Vec<StorageId>,
    mem_alignment: usize,
}

/// A resolved Metal 4 resource: a GPU address + host pointer into a sub-range
/// of a unified-memory `MTLBuffer`.
#[derive(Debug)]
pub struct Metal4Resource {
    /// GPU virtual address of the start of the sub-range (base address + offset),
    /// ready to bind into an `MTL4ArgumentTable`.
    pub gpu_address: u64,
    /// Host pointer to the same sub-range (unified memory).
    pub ptr: *mut u8,
    /// Size of the sub-range in bytes.
    pub size: usize,
}

// SAFETY: the underlying memory is owned by `Metal4Storage`, which is only
// touched under the cubecl channel mutex; the raw pointer is just a view into a
// resident unified-memory allocation that outlives any single server call.
unsafe impl Send for Metal4Resource {}
unsafe impl Sync for Metal4Resource {}

impl Metal4Resource {
    /// View the resource as `&[u8]` (host-coherent on Apple Silicon after the
    /// owning dispatch has retired).
    ///
    /// # Safety
    /// The caller must ensure no GPU work is concurrently writing this range.
    pub unsafe fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// View the resource as `&mut [u8]`.
    ///
    /// # Safety
    /// Same as [`as_bytes`](Self::as_bytes), plus exclusive access.
    pub unsafe fn as_bytes_mut(&self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Metal4Storage {
    /// Create storage on the shared Metal 4 context.
    pub fn new(ctx: Arc<Metal4>, mem_alignment: usize) -> Self {
        Self {
            ctx,
            memory: HashMap::new(),
            deallocations: Vec::new(),
            mem_alignment,
        }
    }

    fn perform_deallocations(&mut self) {
        for id in self.deallocations.drain(..) {
            // Dropping the `Buffer` releases the `Retained<MTLBuffer>`. The
            // residency-set registration is dropped with it on Metal's side.
            self.memory.remove(&id);
        }
    }

    /// Register **external, caller-owned** memory as a no-copy buffer and return a
    /// [`StorageHandle`] over it (zero-copy weight load; see [`Metal4::alloc_no_copy`]).
    ///
    /// # Safety
    /// `ptr` must be page-aligned, valid for `len` bytes, and outlive this storage
    /// entry (the caller keeps the backing `Mmap` alive). Dropping the entry frees
    /// only the `MTLBuffer` wrapper, never `ptr`.
    pub unsafe fn register_external(&mut self, ptr: *mut u8, len: usize) -> StorageHandle {
        let id = StorageId::new();
        let buffer = unsafe { self.ctx.alloc_no_copy(ptr, len) };
        self.memory.insert(id, buffer);
        StorageHandle::new(id, StorageUtilization { offset: 0, size: len as u64 })
    }
}

impl core::fmt::Debug for Metal4Storage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Metal4Storage").finish()
    }
}

impl ComputeStorage for Metal4Storage {
    type Resource = Metal4Resource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let buffer = self
            .memory
            .get(&handle.id)
            .expect("Metal4 storage handle not found");
        let offset = handle.offset();
        let size = handle.size();
        Metal4Resource {
            gpu_address: buffer.gpu_address() + offset,
            // SAFETY: `offset + size <= buffer.len()` is guaranteed by the
            // memory manager that produced this handle.
            ptr: unsafe { buffer.contents_ptr().add(offset as usize) },
            size: size as usize,
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();
        let buffer = self.ctx.alloc(size as usize);
        self.memory.insert(id, buffer);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.deallocations.push(id);
    }

    fn flush(&mut self) {
        self.perform_deallocations();
    }
}
