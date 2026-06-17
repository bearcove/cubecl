use cubecl_common::device::{Device, DeviceId};

/// Max buffer bindings Burn may pack into one (fused) kernel. Metal allows 31
/// buffer slots in the argument table; reserve one for the per-kernel `info`
/// buffer → 30. The fusion engine reads this to bound how many input tensors it
/// folds into a single kernel; advertising more than Metal supports lets it
/// over-fuse, the excess bindings silently never bind, and the kernel reads
/// zeros (observed as near-zero fused-decode output / no-op layers, 2026-06-17).
/// Matches the value the upstream Metal-3 backend (cubecl-metal) uses.
pub const METAL4_MAX_BINDINGS: u32 = 30;

/// A Metal 4 device handle, addressed by index (Apple systems expose one GPU,
/// so index 0 is the system-default device).
#[derive(Clone, PartialEq, Eq, Default, Hash)]
pub struct Metal4Device {
    pub index: usize,
}

impl Metal4Device {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

impl core::fmt::Debug for Metal4Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Metal4({})", self.index)
    }
}

impl Device for Metal4Device {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index as u16,
        }
    }
}
