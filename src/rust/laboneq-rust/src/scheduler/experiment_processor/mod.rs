// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{error::Result, scheduler::experiment::Experiment};

mod resolve_averaging;
mod resolve_phase_reset;
mod resolve_pulses;
mod resolve_section_triggers;
mod resolve_timing_boundary;

use resolve_averaging::resolve_averaging;
use resolve_phase_reset::resolve_phase_reset;
use resolve_pulses::resolve_pulses;
use resolve_section_triggers::resolve_effective_triggers;
use resolve_timing_boundary::resolve_timing_boundary;

/// Processes an [`Experiment`] ready for scheduling.
pub(super) fn process_experiment(experiment: &mut Experiment) -> Result<()> {
    for section in experiment.sections.iter_mut() {
        resolve_timing_boundary(section)?;
        resolve_averaging(section)?;
        resolve_phase_reset(section, &experiment.signals)?;
        resolve_effective_triggers(section)?;
    }
    resolve_pulses(experiment)?;
    Ok(())
}
