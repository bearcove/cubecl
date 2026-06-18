#[cfg(std_io)]
use super::cache::CacheConfig;
use super::logger::{LogLevel, LoggerConfig};

/// Configuration for autotuning in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AutotuneConfig {
    /// Logger configuration for autotune logs, using autotune-specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<AutotuneLogLevel>,

    /// Autotune level, controlling the intensity of autotuning.
    #[serde(default)]
    pub level: AutotuneLevel,

    /// Frozen mode: never benchmark. Read the (shipped) cache, pick the cached
    /// fastest index, and HARD-ERROR on any key that is missing or whose checksum
    /// no longer matches the compiled kernels.
    ///
    /// This exists for shipped applications — notably iOS, where the sandbox makes
    /// the persistent autotune cache effectively non-writable, so a normal
    /// (autotuning) build re-benchmarks every kernel on every cold launch (~20 s).
    /// A frozen build ships a captured table and refuses to benchmark, so the first
    /// dictation is fast and any table/binary skew fails loudly instead of silently
    /// re-tuning.
    #[serde(default)]
    pub frozen: bool,

    /// Cache location for storing autotune results.
    #[serde(default)]
    #[cfg(std_io)]
    pub cache: CacheConfig,
}

/// Log levels for autotune logging in `CubeCL`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLogLevel {
    /// Autotune logging is disabled.
    #[serde(rename = "disabled")]
    Disabled,

    /// Minimal autotune information is logged such as the fastest kernel selected and a few
    /// statistics (default).
    #[default]
    #[serde(rename = "minimal")]
    Minimal,

    /// Full autotune details are logged.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for AutotuneLogLevel {}

/// Autotune levels controlling the intensity of autotuning.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLevel {
    /// Minimal autotuning effort.
    #[serde(rename = "minimal")]
    Minimal,

    /// Balanced autotuning effort (default).
    #[default]
    #[serde(rename = "balanced")]
    Balanced,

    /// Increased autotuning effort.
    #[serde(rename = "extensive")]
    Extensive,

    /// Maximum autotuning effort.
    #[serde(rename = "full")]
    Full,
}
