// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{error::Result, scheduler::experiment::Experiment};

mod validate_pulses;

use validate_pulses::validate_play_pulse_operations;

/// Validates an [`Experiment`].
pub(super) fn validate_experiment(experiment: &Experiment) -> Result<()> {
    validate_play_pulse_operations(experiment)?;
    Ok(())
}
