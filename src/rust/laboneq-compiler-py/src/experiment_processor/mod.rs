// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::error::Result;
use crate::experiment::Experiment;
use crate::experiment_context::ExperimentContext;
use crate::signal_view::signal_views;

mod resolve_averaging;
mod resolve_match;
mod resolve_phase_reset;
mod resolve_pulses;
mod resolve_section_triggers;
mod resolve_timing_boundary;

use laboneq_ir::system::DeviceSetup;
use resolve_averaging::resolve_averaging;
use resolve_match::resolve_match;
use resolve_phase_reset::resolve_phase_reset;
use resolve_pulses::resolve_pulses;
use resolve_section_triggers::resolve_effective_triggers;
use resolve_timing_boundary::resolve_timing_boundary;

/// Processes an [`Experiment`] ready for scheduling.
pub(super) fn process_experiment(
    experiment: &mut Experiment,
    device_setup: &DeviceSetup,
    context: &ExperimentContext,
) -> Result<()> {
    let signal_views = &signal_views(device_setup);

    resolve_timing_boundary(&mut experiment.root)?;
    resolve_averaging(&mut experiment.root)?;
    resolve_phase_reset(
        &mut experiment.root,
        &device_setup.signals().map(|s| (s.uid, s)).collect(),
    )?;
    resolve_effective_triggers(&mut experiment.root)?;
    resolve_pulses(experiment, signal_views)?;
    resolve_match(&mut experiment.root, signal_views, context)?;
    Ok(())
}
