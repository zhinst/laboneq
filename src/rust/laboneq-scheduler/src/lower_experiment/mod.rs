// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::Result;
use crate::experiment::types::{Operation, SignalUid};
use crate::experiment::{ExperimentNode, NodeChild};

use crate::ir::{IrKind, Loop};
use crate::schedule_info::ScheduleInfoBuilder;
use crate::{ParameterStore, ScheduledNode, SignalInfo, TinySample};

mod oscillators;
use oscillators::{handle_initial_local_oscillator_frequency, handle_initial_oscillator_frequency};
mod voltage_offset;
use voltage_offset::handle_initial_voltage_offset;

/// Lowering of [ExperimentNode] to [ScheduledNode].
///
/// The pass will apply the given near-time parameters where applicable.
pub fn lower_to_ir<T: SignalInfo + Sized>(
    node: &ExperimentNode,
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
    lower_to_ir_impl(node, signals, system_grid)?
        .into_iter()
        .for_each(|child| {
            root.add_child(0, child);
        });
    Ok(root)
}

fn lower_to_ir_impl<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    signals: &HashMap<SignalUid, T>,
    system_grid: TinySample,
) -> Result<Vec<ScheduledNode>> {
    match &node.kind {
        Operation::PrngLoop(_) => lower_sweep(node, signals, system_grid),
        Operation::Sweep(_) => lower_sweep(node, signals, system_grid),
        Operation::AveragingLoop(_) => lower_sweep(node, signals, system_grid),
        _ => {
            let mut ir_node = ScheduledNode::new(
                IrKind::NotYetImplemented,
                ScheduleInfoBuilder::new().grid(1).build(),
            );
            for child in node.children.iter() {
                for child_node in lower_to_ir_impl(child, signals, system_grid)? {
                    ir_node.add_child(0, child_node);
                }
            }
            Ok(vec![ir_node])
        }
    }
}

/// Lower Experiment loop nodes into IR nodes.
pub fn lower_sweep<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    signals: &HashMap<SignalUid, T>,
    system_grid: TinySample,
) -> Result<Vec<ScheduledNode>> {
    let loop_info = node.kind.loop_info().expect("Expected a loop");
    let children = lower_children(&node.children, signals, system_grid)?;

    // Loop iteration preamble
    let preamble = ScheduledNode::new(
        IrKind::LoopIterationPreamble,
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    // Loop iteration
    let mut iteration = ScheduledNode::new(
        IrKind::LoopIteration,
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    iteration.add_child(0, preamble);
    children.into_iter().for_each(|child| {
        iteration.add_child(0, child);
    });

    // Root loop
    let mut root = ScheduledNode::new(
        IrKind::Loop(Loop {
            uid: *node.kind.section_info().expect("Expected a section").uid,
            parameters: loop_info.parameters.to_vec(),
            iterations: loop_info.count as usize,
        }),
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    root.add_child(0, iteration);
    Ok(vec![root])
}

fn lower_children(
    children: &[NodeChild],
    signals: &HashMap<SignalUid, impl SignalInfo + Sized>,
    system_grid: TinySample,
) -> Result<Vec<ScheduledNode>> {
    let mut children_ir = Vec::new();
    for child in children {
        let child_nodes = lower_to_ir_impl(child, signals, system_grid)?;
        children_ir.extend(child_nodes);
    }
    Ok(children_ir)
}
