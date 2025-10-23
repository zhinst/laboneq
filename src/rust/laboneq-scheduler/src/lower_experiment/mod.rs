// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::Result;
use crate::experiment::ExperimentNode;
use crate::experiment::types::SignalUid;

use crate::ir::IrKind;
use crate::schedule_info::ScheduleInfoBuilder;
use crate::{ParameterStore, ScheduledNode, SignalInfo, TinySample};

mod oscillators;
use oscillators::{handle_initial_local_oscillator_frequency, handle_initial_oscillator_frequency};
mod voltage_offset;
use voltage_offset::handle_initial_voltage_offset;

/// Lowering of Experiment nodes to scheduled IR nodes.
///
/// The pass will apply the given near-time parameters where applicable.
pub fn lower_to_ir<T: SignalInfo + Sized>(
    _node: &ExperimentNode, // Not actually needed just yet
    signals: &HashMap<SignalUid, T>,
    nt_parameters: &ParameterStore,
    system_grid: TinySample,
) -> Result<ScheduledNode> {
    let mut root = ScheduledNode::new(
        IrKind::NotYetImplemented,
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    let initial_oscillator_frequency =
        handle_initial_oscillator_frequency(signals, nt_parameters, system_grid)?;
    let initial_local_oscillator_frequency =
        handle_initial_local_oscillator_frequency(signals, nt_parameters, system_grid)?;
    root.add_child(0, initial_oscillator_frequency);
    initial_local_oscillator_frequency
        .into_iter()
        .for_each(|child| {
            root.add_child(0, child);
        });
    let initial_voltage_offset =
        handle_initial_voltage_offset(signals, nt_parameters, system_grid)?;
    initial_voltage_offset.into_iter().for_each(|child| {
        root.add_child(0, child);
    });
    Ok(root)
}
