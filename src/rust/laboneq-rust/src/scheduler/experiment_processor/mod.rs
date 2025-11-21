// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{error::Result, scheduler::experiment::Experiment};

mod resolve_averaging;
mod resolve_phase_reset;
mod resolve_timing_boundary;

use resolve_averaging::resolve_averaging;
use resolve_phase_reset::resolve_phase_reset;
use resolve_timing_boundary::resolve_timing_boundary;

/// Processes an [`Experiment`] ready for scheduling.
pub fn process_experiment(experiment: &mut Experiment) -> Result<()> {
    for section in experiment.sections.iter_mut() {
        resolve_timing_boundary(section)?;
        resolve_averaging(section)?;
        resolve_phase_reset(section, &experiment.signals)?;
    }
    Ok(())
}
