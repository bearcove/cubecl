use cubecl_common::device::{Device, DeviceId};

/// Upper bound on live argument-table buffer bindings tracked per dispatch.
pub const METAL4_MAX_BINDINGS: u32 = 1024;

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
