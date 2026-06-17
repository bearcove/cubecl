//! Optional bridge from per-dispatch Metal 4 counter-heap timestamps to a stax
//! target lane (`stax-target`), mirroring bee's `helix-metal4::stax`. Spans show
//! up in stax as a synthetic "GPU" thread with kernels as named frames, rendered
//! under the CPU stack that dispatched them (via the captured origin).
//!
//! Behind the `stax` cargo feature. When the feature is off this is a zero-cost
//! no-op (the public `cubecl` build never pulls `stax-target`); helix-burn turns
//! it on for profiling. Capture is additionally gated at runtime on
//! [`reporting_active`] — free unless a stax recording of this process is live.

#[cfg(feature = "stax")]
mod backend {
    use std::sync::OnceLock;

    pub use stax_target::CapturedOrigin as Origin;

    fn lane() -> &'static stax_target::Lane {
        static LANE: OnceLock<stax_target::Lane> = OnceLock::new();
        LANE.get_or_init(|| stax_target::Lane::new("GPU metal4"))
    }

    /// True while a stax recording of this process is active (one relaxed load).
    pub fn reporting_active() -> bool {
        lane().reporting_active()
    }

    /// Capture the CPU origin (tid + timestamp) at dispatch-encoding time so stax
    /// can render the GPU kernel under the sampled CPU stack that queued it.
    pub fn capture_origin() -> Origin {
        lane().capture_origin()
    }

    /// Report resolved per-dispatch spans. `ticks` are raw device timestamps;
    /// `freq_hz` converts them to the nanosecond clock CPU profilers record in.
    pub fn report(freq_hz: u64, spans: Vec<(String, u64, u64, Origin)>) {
        let to_ns = |t: u64| -> u64 {
            if freq_hz == 0 {
                t
            } else {
                ((t as u128) * 1_000_000_000 / freq_hz as u128) as u64
            }
        };
        let lane = lane();
        let built: Vec<_> = spans
            .into_iter()
            .filter_map(|(name, begin, end, origin)| {
                lane.span_with_captured_origin(name, to_ns(begin), to_ns(end), origin)
            })
            .collect();
        if !built.is_empty() {
            let _ = lane.report_if_active(built);
        }
    }
}

#[cfg(not(feature = "stax"))]
mod backend {
    /// Placeholder origin (the real one lives in `stax-target`).
    #[derive(Clone, Default)]
    pub struct Origin;

    #[inline]
    pub fn reporting_active() -> bool {
        false
    }

    #[inline]
    pub fn capture_origin() -> Origin {
        Origin
    }

    #[inline]
    pub fn report(_freq_hz: u64, _spans: Vec<(String, u64, u64, Origin)>) {}
}

pub use backend::*;
