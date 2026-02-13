// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::ir::{PlayPulse, Samples, compilation_job::Signal};

/// Create a delay PlayPulse with the given signal and length.
pub(crate) fn delay(signal: Arc<Signal>, length: Samples) -> PlayPulse {
    PlayPulse {
        length,
        signal,
        amplitude: None,
        amp_param_name: None,
        phase: 0.0,
        set_oscillator_phase: None,
        incr_phase_param_name: None,
        increment_oscillator_phase: None,
        id_pulse_params: None,
        markers: vec![],
        pulse_def: None,
    }
}
