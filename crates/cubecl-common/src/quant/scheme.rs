use alloc::vec;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use core::{default::Default, ops::Deref};
use serde::{Deserialize, Serialize};

/// A codebook (centroid table) for [`QuantMode::Codebook`], supplied by the
/// caller as a **comptime constant** baked into the shader — never a runtime
/// buffer, and the values are never hardcoded in this fork (the caller, e.g.
/// bee, owns them and passes `Codebook(&ITS_TABLE)`).
///
/// `f32` is neither `Hash` nor `Eq` (NaN), so we key on the bit pattern — fine
/// for a fixed table — which lets `Codebook` be a `#[comptime]` kernel arg.
#[derive(Clone, Copy, Debug)]
pub struct Codebook(pub &'static [f32]);

impl PartialEq for Codebook {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
impl Eq for Codebook {}
impl Hash for Codebook {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in self.0 {
            v.to_bits().hash(state);
        }
    }
}

/// TQ2 (`Q2F`) codebook: 4 Lloyd-Max levels for a unit-variance Gaussian.
pub const Q2F: [f32; 4] = [-1.510_400, -0.452_800, 0.452_800, 1.510_400];

/// TQ4 (`Q4F`) codebook: 16 Lloyd-Max levels for a unit-variance Gaussian.
pub const Q4F: [f32; 16] = [
    -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
    0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
];

/// TQ6 (`Q6F`) codebook: 64 Lloyd-Max levels for a unit-variance Gaussian.
pub const Q6F: [f32; 64] = [
    -3.73971331,
    -3.23553866,
    -2.91215583,
    -2.66675206,
    -2.46556925,
    -2.29307792,
    -2.14077946,
    -2.00348979,
    -1.87780041,
    -1.76134301,
    -1.65240050,
    -1.54968499,
    -1.45220328,
    -1.35917132,
    -1.26995767,
    -1.18404491,
    -1.10100239,
    -1.02046671,
    -0.94212725,
    -0.86571539,
    -0.79099622,
    -0.71776211,
    -0.64582771,
    -0.57502585,
    -0.50520434,
    -0.43622321,
    -0.36795256,
    -0.30027058,
    -0.23306199,
    -0.16621658,
    -0.09962796,
    -0.03319237,
    0.03319237,
    0.09962796,
    0.16621658,
    0.23306199,
    0.30027058,
    0.36795256,
    0.43622321,
    0.50520434,
    0.57502585,
    0.64582771,
    0.71776211,
    0.79099622,
    0.86571539,
    0.94212725,
    1.02046671,
    1.10100239,
    1.18404491,
    1.26995767,
    1.35917132,
    1.45220328,
    1.54968499,
    1.65240050,
    1.76134301,
    1.87780041,
    2.00348979,
    2.14077946,
    2.29307792,
    2.46556925,
    2.66675206,
    2.91215583,
    3.23553866,
    3.73971331,
];

/// Centroid table for a table-codebook `value` — the canonical TQ codebooks,
/// shared by every consumer (burn-cubecl, burn-cubecl-fusion, cubek) so the
/// table lives in ONE place. Linear (`Q8F`) / symmetric values read no centroid
/// table, so they get an empty placeholder (the codebook branch is comptime-guarded
/// off for them).
pub fn codebook_for(value: QuantValue) -> Codebook {
    match value {
        QuantValue::Q2F => Codebook(&Q2F),
        QuantValue::Q4F => Codebook(&Q4F),
        QuantValue::Q6F => Codebook(&Q6F),
        _ => Codebook(&[]),
    }
}

/// Describes a quantization scheme/configuration.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantScheme {
    /// The logical data type of quantized input values (e.g., `QInt8`).
    ///
    /// This defines how values are interpreted during computation, independent of how they're stored.
    pub value: QuantValue,
    /// Precision used for quantization parameters (e.g., scale and biases).
    pub param: QuantParam,
    /// Data type used for storing quantized values.
    pub store: QuantStore,
    /// Granularity level of quantization (e.g., per-tensor).
    pub level: QuantLevel,
    /// Quantization mode (e.g., symmetric).
    pub mode: QuantMode,
    /// Activation-side rotation applied before the contraction. The weights are
    /// stored pre-rotated; the forward transform is folded onto the activation at
    /// matmul time (so the quant-matmul stays a single fusable op rather than a
    /// separate rotation op + a custom barrier). Defaults to [`Rotation::None`],
    /// preserving every existing scheme's behavior.
    pub rotation: Rotation,
}

impl Default for QuantScheme {
    fn default() -> Self {
        Self {
            value: QuantValue::Q8F,
            param: QuantParam::F32,
            store: QuantStore::PackedU32(0),
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            rotation: Rotation::None,
        }
    }
}

/// Activation-side rotation folded into the quant-matmul. The sign table is
/// canonical/deterministic (resolved at codegen, like [`codebook_for`]), so the
/// scheme only carries WHICH rotation and the block size — no per-scheme data.
#[derive(
    Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum Rotation {
    /// No rotation (the default — plain dequantize-and-matmul).
    #[default]
    None,
    /// Forward random-Hadamard transform (RHT) applied to each `block`-sized span
    /// of the activation before the contraction. Mirrors bee's `matvec_prerot`:
    /// weights are stored rotated, the activation is rotated in-kernel.
    Rht {
        /// Hadamard block size (e.g. 32).
        block: u32,
    },
}

impl QuantScheme {
    /// Set the quantization level.
    pub fn with_level(mut self, level: QuantLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the quantization mode.
    pub fn with_mode(mut self, mode: QuantMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the data type used for quantized values.
    pub fn with_value(mut self, value: QuantValue) -> Self {
        self.value = value;
        self
    }

    /// Set the data type used to store quantized values.
    pub fn with_store(mut self, store: QuantStore) -> Self {
        self.store = store;
        self
    }

    /// Set the precision used for quantization parameters
    pub fn with_param(mut self, param: QuantParam) -> Self {
        self.param = param;
        self
    }

    /// Set the activation-side rotation folded into the quant-matmul.
    pub fn with_rotation(mut self, rotation: Rotation) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns the size of the quantization storage type in bits.
    pub fn size_bits_stored(&self) -> usize {
        self.store.size_bits(&self.value)
    }

    /// Returns the size of the quantization storage type in bits.
    pub fn size_bits_value(&self) -> usize {
        self.value.size_bits()
    }

    /// Returns the number of quantized values stored in a single element.
    pub fn num_quants(&self) -> usize {
        self.size_bits_stored() / self.value.size_bits()
    }

    /// Number of `u32` storage words holding one dense unit (one `num_quants`
    /// group). For [`QuantStore::PackedU32`] this is always 1; for
    /// [`QuantStore::PackedU32Dense`] it is `lcm(value_bits, 32) / 32` (e.g. 3
    /// for 6-bit codes, where 16 codes straddle 3 words). Used to size the `NQ`
    /// of the stored `Vector<u32, NQ>` so a single load covers a whole unit —
    /// `num_quants / num_quants` (= 1 unit) does NOT give the word count when
    /// codes don't divide 32 evenly.
    pub fn storage_words_per_unit(&self) -> usize {
        match self.store {
            QuantStore::PackedU32(_) | QuantStore::PackedU32Dense(_) => {
                self.size_bits_stored() / 32
            }
            // Non-u32 stores don't pack into u32 words; one element per quant unit.
            QuantStore::Native | QuantStore::PackedNative(_) => 1,
        }
    }

    /// Returns the native packing factor for the values. When native packing > 1, the packed
    /// representation stores `num_quants` elements grouped into packs of `native_packing` size.
    pub fn native_packing(&self) -> usize {
        self.value.native_packing()
    }

    /// Returns the packing dim for the store.
    pub fn packing_dim(&self) -> Option<usize> {
        self.store.packing_dim()
    }

    /// Swaps the packing dim if it's either of `dim0` or `dim1`.
    /// Executes the corresponding update to `shape.swap(dim0, dim1)`.
    pub fn swap_packing_dim(&mut self, dim0: usize, dim1: usize) {
        if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
            &mut self.store
        {
            if *packed_dim == dim0 {
                *packed_dim = dim1;
            } else if *packed_dim == dim1 {
                *packed_dim = dim0;
            }
        }
    }
}

/// Level or granularity of quantization.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantLevel {
    /// Quantize the whole tensor using a single tensor.
    Tensor,
    /// Quantize a tensor using multiple blocks.
    Block(BlockSize),
}

impl QuantLevel {
    /// Converting constructor for [`QuantLevel::Block`]
    pub fn block(values: impl AsRef<[u8]>) -> Self {
        QuantLevel::Block(BlockSize::new(values))
    }
}

/// Data type used to represent quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantValue {
    /// 8-bit quantization with full range.
    Q8F,
    /// 8-bit floating point, e5m2 format.
    E5M2,
    /// 8-bit floating point, e4m3 format.
    E4M3,
    /// 6-bit quantization with full range. Codes pack densely (no even u32
    /// division), so this requires a dense store ([`QuantStore::PackedU32Dense`]).
    Q6F,
    /// 4-bit quantization with full range.
    Q4F,
    /// 4-bit floating point, e2m1 format.
    E2M1,
    /// 2-bit quantization with full range.
    Q2F,
    /// 8-bit quantization with symmetric range.
    Q8S,
    /// 4-bit quantization with symmetric range.
    Q4S,
    /// 2-bit quantization with symmetric range.
    Q2S,
}

impl QuantValue {
    /// Returns the size of the quantization input type in bits.
    pub fn size_bits(&self) -> usize {
        match self {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => 8,
            QuantValue::Q6F => 6,
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::E2M1 => 4,
            QuantValue::Q2F | QuantValue::Q2S => 2,
        }
    }

    /// Packing factor for the native representation used for intermediate values. If > 1, values
    /// should always be processed in `native_packing` sized chunks.
    pub fn native_packing(&self) -> usize {
        match self {
            QuantValue::E2M1 => 2,
            _ => 1,
        }
    }

    /// The possible range of values allowed by the quant value.
    pub fn range(&self) -> (f32, f32) {
        match self {
            QuantValue::Q8F => (i8::MIN as f32, i8::MAX as f32),
            QuantValue::Q6F => (-32.0, 31.0),
            QuantValue::Q4F => (-8.0, 7.0),
            QuantValue::Q2F => (-2.0, 1.0),
            QuantValue::Q8S => (-i8::MAX as f32, i8::MAX as f32),
            QuantValue::Q4S => (-7.0, 7.0),
            QuantValue::Q2S => (-1.0, 1.0),
            QuantValue::E4M3 => (-448.0, 448.0),
            QuantValue::E5M2 => (-57344.0, 57344.0),
            QuantValue::E2M1 => (-6.0, 6.0), // Hardcoded because of no-std
        }
    }

    /// If the range of values is symmetric around zero.
    pub fn is_symmetric(&self) -> bool {
        match self {
            Self::Q8F
            | Self::Q6F
            | Self::Q4F
            | Self::Q2F
            | Self::E4M3
            | Self::E5M2
            | Self::E2M1 => false,
            Self::Q8S | Self::Q4S | Self::Q2S => true,
        }
    }
}

impl QuantStore {
    /// Returns the size of the quantization input type in bits.
    pub fn size_bits(&self, value: &QuantValue) -> usize {
        match self {
            QuantStore::Native => value.size_bits(),
            QuantStore::PackedNative(_) => value.size_bits() * value.native_packing(),
            QuantStore::PackedU32(_) => 32,
            // A dense unit is the smallest run of u32 words holding a whole number
            // of codes: lcm(code bits, 32).
            QuantStore::PackedU32Dense(_) => lcm(value.size_bits(), 32),
        }
    }

    fn packing_dim(&self) -> Option<usize> {
        match self {
            QuantStore::Native => None,
            QuantStore::PackedNative(packing_dim)
            | QuantStore::PackedU32(packing_dim)
            | QuantStore::PackedU32Dense(packing_dim) => Some(*packing_dim),
        }
    }
}

const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

const fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// Data type used to stored quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantStore {
    /// Native quantization doesn't require packing and unpacking.
    Native,
    /// Store packed quantized values in a natively supported packing format (i.e. e2m1x2).
    /// Argument is the dimension the tensor is packed on, starting from the innermost dimension.
    PackedNative(usize),
    /// Store packed quantized values in a 4-byte unsigned integer.
    /// Argument is the dimension the tensor is packed on, starting from the innermost dimension.
    PackedU32(usize),
    /// Densely bit-pack codes into a u32 stream: code `j` occupies bits
    /// `[j*size_bits, (j+1)*size_bits)`. For widths that don't divide 32 evenly
    /// (e.g. 6-bit), codes straddle word boundaries. One "unit" is
    /// `lcm(size_bits, 32)` bits = `lcm/32` u32 words holding `lcm/size_bits`
    /// codes with no straddle across the unit boundary.
    /// Argument is the dimension the tensor is packed on, starting from the innermost dimension.
    PackedU32Dense(usize),
    // /// Store packed quantized values in a 8-bit unsigned integer.
    // U8,
}

/// Strategy used to quantize values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantMode {
    /// Symmetric or scale quantization.
    Symmetric,
    /// Codebook (lookup-table) quantization: the stored value is an index into a
    /// per-format centroid table, and dequant is `centroid[index] * scale`
    /// (e.g. Lloyd-Max codebooks). Unlike [`Symmetric`](Self::Symmetric) the
    /// stored bits are an unsigned index, not a signed integer.
    Codebook,
}

/// Quantization floating-point precision.
///
/// This is used to represent the floating-point precision of quantization parameters like the scale(s)
/// or the accumulation precision used during operations like matrix multiplication.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantParam {
    /// Full precision.
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
    /// unsigned floating point, e8m0 format.
    UE8M0,
    /// unsigned floating point, e4m3 format.
    UE4M3,
}

const MAX_DIMS: usize = 5;

/// Copyable block size, specialized version of `SmallVec`.
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockSize {
    storage: [u8; MAX_DIMS],
    len: u8,
}

impl core::fmt::Debug for BlockSize {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "BlockSize({:?})", self.as_slice())
    }
}

impl BlockSize {
    /// Max number of dimensions for block size
    pub const MAX_DIMS: usize = MAX_DIMS;

    /// Create a new blocksize from a set of values. The number of values must be `<= MAX_DIMS`.
    pub fn new(values: impl AsRef<[u8]>) -> Self {
        let values = values.as_ref();
        debug_assert!(
            values.len() <= MAX_DIMS,
            "Tried creating a block size larger than the cap"
        );
        let len = values.len().min(MAX_DIMS);
        let mut storage = [1; MAX_DIMS];
        storage[..len].copy_from_slice(&values[..len]);
        Self {
            storage,
            len: len as u8,
        }
    }

    /// Create a new blocksize from a set of values. The number of values must be `<= MAX_DIMS`.
    /// Trims any leading zeros.
    pub fn new_trim(values: impl AsRef<[u8]>) -> Self {
        let values = values.as_ref();
        let first_value = values.iter().position(|s| *s != 1).unwrap_or(0);
        Self::new(&values[first_value..])
    }

    /// Return a slice of only the initialized values
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[..self.len as usize]
    }

    /// Return a vec of only the initialized values
    pub fn to_vec(&self) -> Vec<u8> {
        self.storage[..self.len as usize].to_vec()
    }

    /// Returns `N` dimensions, unsqueezing if necessary.
    pub fn as_dim<const N: usize>(&self) -> [u8; N] {
        let data_len = N.min(self.len as usize);
        let data_start = N - data_len;
        let mut out = [1; N];
        out[data_start..].copy_from_slice(&self.storage[..data_len]);
        out
    }

    /// Returns a vector of `len` dimensions, unsqueezing if necessary.
    pub fn to_dim_vec(&self, len: usize) -> Vec<u8> {
        let data_len = len.min(self.len as usize);
        let data_start = len - data_len;
        let mut out = vec![1; len];
        out[data_start..].copy_from_slice(&self.storage[..data_len]);
        out
    }

    /// Create an iterator over all stored dimensions
    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.as_slice().iter()
    }

    /// Returns the total number of elements in each block
    pub fn num_elements(&self) -> usize {
        self.iter().map(|it| *it as usize).product()
    }
}

impl Deref for BlockSize {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: AsRef<[u8]>> From<T> for BlockSize {
    fn from(value: T) -> Self {
        BlockSize::new(value)
    }
}
