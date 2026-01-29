// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{OscillatorKind, SectionUid, SignalUid};
use laboneq_log::diagnostic;
use laboneq_units::tinysample::{
    TinySamples, seconds_to_tinysamples, tiny_samples, tinysamples_to_seconds,
};
use num_traits::Inv;

use crate::error::Result;
use crate::experiment_context::ExperimentContext;
use crate::ir::IrKind;
use crate::schedule_info::ScheduleInfoBuilder;
use crate::utils::{ceil_to_grid, lcm};
use crate::{ScheduleInfo, ScheduledNode, SignalInfo};

/// Creates a [ScheduledNode] that represents a [IrKind::ResetOscillatorPhase] for the given signals.
///
/// The timing properties `length` and `grid` are determined based on the device
/// traits `oscillator_reset_duration` and `lo_frequency_granularity` of the hardware modulated
/// signals.
pub(super) fn handle_reset_oscillator_phase<T: SignalInfo>(
    signals: &[&T],
    ctx: &ExperimentContext<T>,
    system_grid: TinySamples,
    section_uid: SectionUid,
) -> Result<ScheduledNode> {
    let ir = create_ir(signals.iter().map(|s| s.uid()));
    let schedule = create_schedule(signals, ctx, system_grid, section_uid)?;
    let node = ScheduledNode::new(ir, schedule);
    Ok(node)
}

fn create_ir(signals: impl IntoIterator<Item = SignalUid>) -> IrKind {
    IrKind::ResetOscillatorPhase {
        signals: signals.into_iter().collect(),
    }
}

fn create_schedule<T: SignalInfo>(
    signals: &[&T],
    ctx: &ExperimentContext<T>,
    system_grid: TinySamples,
    section_uid: SectionUid,
) -> Result<ScheduleInfo> {
    let mut hw_reset_signals = Vec::new();
    for signal in signals {
        if let Some(osc) = signal.oscillator()
            && osc.kind == OscillatorKind::Hardware
        {
            hw_reset_signals.push(signal);
        }
    }
    let mut grid = 1;
    let mut length = 0;
    for signal in hw_reset_signals.iter() {
        let duration = seconds_to_tinysamples(signal.device_traits().oscillator_reset_duration);
        length = length.max(duration.value());
        grid = lcm(grid, system_grid.value());
        if let Some(lo_freq_granularity) = signal.device_traits().lo_frequency_granularity {
            // Align the grid the LO frequency granularity to ensure phase consistency after the reset of the NCO.
            let duration = seconds_to_tinysamples(lo_freq_granularity.inv());
            let grid_adjusted = lcm(grid, duration.value());
            // TODO: Returns a dedicated result type with logging information instead of logging directly here?
            if grid != grid_adjusted {
                diagnostic!(
                    "Phase reset in section '{}' has extended the section's timing grid to {}, so to be commensurate with the local oscillator.",
                    ctx.resolve_uid(section_uid)?,
                    tinysamples_to_seconds(tiny_samples(grid_adjusted)),
                );
            }
            grid = grid_adjusted;
        }
        diagnostic!(
            "An additional delay of {} has been added on signal '{}' to wait for the phase reset.",
            signal.device_traits().oscillator_reset_duration,
            ctx.resolve_uid(signal.uid())?,
        );
    }
    let schedule = ScheduleInfoBuilder::new()
        .grid(grid)
        .length(ceil_to_grid(length, grid))
        .build();
    Ok(schedule)
}
