// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::SignalUid;
use laboneq_units::duration::{Duration, Second};

/// Trait for calculating feedback latencies.
pub trait FeedbackCalculator {
    type Error: std::error::Error;

    /// Computes the feedback latency for a given acquisition signal and its associated signals.
    ///
    /// Parameters:
    /// - `absolute_start`: The absolute start time of the acquisition.
    /// - `acquisition_length`: The length of the acquisition.
    /// - `local_feedback`: A boolean indicating if the feedback is local.
    /// - `acquisition_signal`: The UID of the acquisition signal.
    /// - `associated_signals`: An iterator over the UIDs of associated signals used to react to the acquisition.
    fn compute_feedback_latency(
        &self,
        absolute_start: Duration<Second>,
        acquisition_length: Duration<Second>,
        local_feedback: bool,
        acquisition_signal: SignalUid,
        associated_signals: impl Iterator<Item = SignalUid>,
    ) -> Result<Duration<Second>, Self::Error>;
}
