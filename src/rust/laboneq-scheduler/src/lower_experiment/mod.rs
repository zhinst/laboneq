// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::experiment::types::{self as experiment_types, NumericLiteral};
use crate::experiment::types::{Operation, SignalUid};
use crate::experiment::{ExperimentNode, NodeChild};

use crate::experiment_context::ExperimentContext;
use crate::ir::{Acquire, Case, IrKind, Loop, Match, PlayPulse, Section, Trigger};
use crate::schedule_info::ScheduleInfoBuilder;

use crate::utils::lcm;
use crate::{ParameterStore, ScheduledNode, SignalInfo};
use laboneq_units::tinysample::{TinySamples, tiny_samples};

mod oscillators;
use oscillators::{
    handle_initial_local_oscillator_frequency, handle_initial_oscillator_frequency,
    handle_set_oscillator_frequency,
};
mod voltage_offset;
use voltage_offset::handle_initial_voltage_offset;
mod reset_phase;
use reset_phase::handle_reset_oscillator_phase;
mod ppc_sweep_steps;
use ppc_sweep_steps::handle_ppc_sweep_steps;

/// Lowering of [ExperimentNode] to [ScheduledNode].
///
/// The pass will apply the given near-time parameters where applicable.
pub fn lower_to_ir<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    nt_parameters: &ParameterStore,
    system_grid: TinySamples,
) -> Result<ScheduledNode> {
    let mut root = ScheduledNode::new(
        IrKind::NotYetImplemented,
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    let concrete_signals = ctx.signals().collect::<Vec<_>>();
    let initial_oscillator_frequency =
        handle_initial_oscillator_frequency(&concrete_signals, nt_parameters, system_grid)?;
    let initial_local_oscillator_frequency =
        handle_initial_local_oscillator_frequency(&concrete_signals, nt_parameters, system_grid)?;
    root.add_child(tiny_samples(0), initial_oscillator_frequency);
    initial_local_oscillator_frequency
        .into_iter()
        .for_each(|child| {
            root.add_child(tiny_samples(0), child);
        });
    let initial_voltage_offset =
        handle_initial_voltage_offset(&concrete_signals, nt_parameters, system_grid)?;
    initial_voltage_offset.into_iter().for_each(|child| {
        root.add_child(tiny_samples(0), child);
    });
    lower_to_ir_impl(node, ctx, system_grid)?
        .into_iter()
        .for_each(|child| {
            root.add_child(tiny_samples(0), child);
        });
    Ok(root)
}

fn lower_to_ir_impl<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    system_grid: TinySamples,
) -> Result<Vec<ScheduledNode>> {
    match &node.kind {
        Operation::Section(section) => Ok(vec![lower_section(
            section,
            &node.children,
            ctx,
            system_grid,
        )?]),
        Operation::PrngLoop(_) => lower_sweep(node, ctx, system_grid),
        Operation::Sweep(_) => lower_sweep(node, ctx, system_grid),
        Operation::AveragingLoop(_) => lower_sweep(node, ctx, system_grid),
        Operation::Reserve(reserve) => Ok(vec![lower_reserve(reserve)]),
        Operation::PlayPulse(play_pulse) => Ok(vec![lower_play_pulse(play_pulse)]),
        Operation::Acquire(acquire) => Ok(vec![lower_acquire(acquire)]),
        Operation::Delay(delay) => Ok(vec![lower_delay(delay)]),
        Operation::ResetOscillatorPhase(reset) => Ok(lower_reset_oscillator_phase(reset)
            .map(|node| vec![node])
            .unwrap_or_default()),
        Operation::Case(_) => Err(Error::new("Case must be used within a match block.")),
        Operation::Match(match_) => {
            Ok(vec![lower_match(match_, &node.children, ctx, system_grid)?])
        }
        Operation::PrngSetup(_) | Operation::Root | Operation::RealTimeBoundary => {
            let children = lower_children(&node.children, ctx, system_grid)?;
            let mut ir_node = ScheduledNode::new(
                IrKind::NotYetImplemented,
                ScheduleInfoBuilder::new().grid(1).build(),
            );
            children.into_iter().for_each(|child_node| {
                ir_node.add_child(tiny_samples(0), child_node);
            });
            Ok(vec![ir_node])
        }
        Operation::NearTimeCallback => {
            panic!("Near-time callbacks cannot exist in real-time.")
        }
        Operation::SetNode => panic!("Set node cannot exist in real-time."),
    }
}

fn lower_children(
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo + Sized>,
    system_grid: TinySamples,
) -> Result<Vec<ScheduledNode>> {
    let mut result = Vec::new();
    for child in children {
        let nodes = lower_to_ir_impl(child, ctx, system_grid)?;
        result.extend(nodes);
    }
    Ok(result)
}

/// Lower Experiment loop nodes into IR nodes.
pub fn lower_sweep<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    system_grid: TinySamples,
) -> Result<Vec<ScheduledNode>> {
    let loop_info = node
        .kind
        .loop_info()
        .ok_or_else(|| Error::new("Expected a loop"))?;
    let children = lower_children(&node.children, ctx, system_grid)?;
    let this_signals: HashSet<&SignalUid> = children
        .iter()
        .flat_map(|child| child.schedule.signals.iter())
        .collect();

    // Loop iteration preamble
    let mut preamble = ScheduledNode::new(
        IrKind::LoopIterationPreamble,
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    let signal_vec = this_signals
        .iter()
        .map(|s| ctx.get_signal(s))
        .collect::<Result<Vec<&T>>>()?;
    if let Some(set_osc_freq) = handle_set_oscillator_frequency(
        &signal_vec,
        loop_info.parameters.iter().collect(),
        system_grid,
    )? {
        preamble.add_child(tiny_samples(0), set_osc_freq);
    }
    let mut grid = 1;
    if loop_info.reset_oscillator_phase && !signal_vec.is_empty() {
        let reset_osc_node =
            handle_reset_oscillator_phase(&signal_vec, ctx, system_grid, loop_info.uid)?;
        if reset_osc_node.schedule.grid.value() != grid {
            // On SHFxx, we align the phase reset with the LO granularity (100 MHz)
            grid = lcm(grid, reset_osc_node.schedule.grid.value());
        }
        preamble.add_child(tiny_samples(0), reset_osc_node);
    }
    for ppc_step in handle_ppc_sweep_steps(
        &ctx.signals().collect::<Vec<_>>(), // Use all the signals present in the experiment
        loop_info.parameters,
        tiny_samples(lcm(grid, system_grid.value())),
    )? {
        preamble.add_child(tiny_samples(0), ppc_step);
    }

    // Loop iteration
    let mut iteration = ScheduledNode::new(
        IrKind::LoopIteration,
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    iteration.add_child(tiny_samples(0), preamble);
    children.into_iter().for_each(|child| {
        iteration.add_child(tiny_samples(0), child);
    });

    // Root loop
    let mut root = ScheduledNode::new(
        IrKind::Loop(Loop {
            uid: *loop_info.uid,
            parameters: loop_info.parameters.to_vec(),
            iterations: loop_info.count as usize,
        }),
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    root.add_child(tiny_samples(0), iteration);
    Ok(vec![root])
}

fn lower_section(
    section: &experiment_types::Section,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo>,
    system_grid: TinySamples,
) -> Result<ScheduledNode> {
    let children = lower_children(children, ctx, system_grid)?;
    let mut root = ScheduledNode::new(
        IrKind::Section(Section {
            uid: section.uid,
            triggers: section
                .triggers
                .iter()
                .map(|trig| Trigger {
                    signal: trig.signal,
                    state: trig.state,
                })
                .collect(),
        }),
        ScheduleInfoBuilder::new().grid(1).build(),
    );
    children.into_iter().for_each(|child| {
        root.add_child(tiny_samples(0), child);
    });
    Ok(root)
}

fn lower_reserve(obj: &experiment_types::Reserve) -> ScheduledNode {
    ScheduledNode::new(
        IrKind::Reserve { signal: obj.signal },
        ScheduleInfoBuilder::new().grid(1).build(),
    )
}

fn lower_play_pulse(obj: &experiment_types::PlayPulse) -> ScheduledNode {
    ScheduledNode::new(
        IrKind::PlayPulse(PlayPulse { signal: obj.signal }),
        ScheduleInfoBuilder::new().grid(1).build(),
    )
}

fn lower_acquire(obj: &experiment_types::Acquire) -> ScheduledNode {
    ScheduledNode::new(
        IrKind::Acquire(Acquire { signal: obj.signal }),
        ScheduleInfoBuilder::new().grid(1).build(),
    )
}

fn lower_delay(obj: &experiment_types::Delay) -> ScheduledNode {
    ScheduledNode::new(
        IrKind::PlayPulse(PlayPulse { signal: obj.signal }),
        ScheduleInfoBuilder::new().grid(1).build(),
    )
}

fn lower_reset_oscillator_phase(
    obj: &experiment_types::ResetOscillatorPhase,
) -> Option<ScheduledNode> {
    if obj.signals.is_empty() {
        panic!("Expected signals to be defined for ResetOscillatorPhase.");
    } else {
        ScheduledNode::new(
            IrKind::ResetOscillatorPhase {
                signals: obj.signals.clone(),
            },
            ScheduleInfoBuilder::new().grid(1).build(),
        )
        .into()
    }
}

fn cast_case(kind: &Operation) -> Option<&experiment_types::Case> {
    if let Operation::Case(case) = kind {
        Some(case)
    } else {
        None
    }
}

fn lower_match(
    section: &experiment_types::Match,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo>,
    system_grid: TinySamples,
) -> Result<ScheduledNode> {
    let mut root = ScheduledNode::new(
        IrKind::Match(Match {
            uid: section.uid,
            target: section.target.clone(),
            // TODO: Resolve in experiment processor!
            // At this point the local flag should be resolved to concrete value.
            local: section.local.unwrap_or(false),
            play_after: section.play_after.clone(),
        }),
        ScheduleInfoBuilder::new().build(),
    );
    // Matching a sweep parameter requires special handling as each case
    // maps to a specific iteration of a loop.
    if let experiment_types::MatchTarget::SweepParameter(param_uid) = &section.target {
        let parameter = ctx.sweep_parameter(param_uid)?;
        // Map the iteration number of a loop to a specific case.
        for idx in 0..parameter.len() {
            for child_node in children {
                let target_value: NumericLiteral = parameter
                    .value_numeric_at_index(idx)
                    .unwrap_or_else(|| panic!("Expected value to exist"));
                let case = cast_case(&child_node.kind)
                    .ok_or_else(|| Error::new("Expected a case operation."))?;
                if case.state == target_value {
                    let kind = IrKind::Case(Case {
                        uid: case.uid,
                        state: idx,
                    });
                    let mut case_node =
                        ScheduledNode::new(kind, ScheduleInfoBuilder::new().build());
                    lower_children(&child_node.children, ctx, system_grid)?
                        .into_iter()
                        .for_each(|child| {
                            case_node.add_child(tiny_samples(0), child);
                        });
                    root.add_child(tiny_samples(0), case_node);
                    break;
                }
            }
        }
        // All parameters must be covered by a case,
        // but not all cases need to be used.
        if parameter.len() > children.len() {
            let msg = format!(
                "Using a match statement for sweep parameter must cover all values.
            Match statement for parameter '{}' has {} cases, but parameter has {} values.",
                parameter.uid.0,
                children.len(),
                parameter.len(),
            );
            return Err(Error::new(msg));
        }
    } else {
        for child in children {
            let case =
                cast_case(&child.kind).ok_or_else(|| Error::new("Expected a case operation."))?;
            let kind = IrKind::Case(Case {
                uid: case.uid,
                state: case.state.try_into().map_err(|e| {
                    Error::new(format!(
                        "Invalid case state value '{}' for case uid '{}': {}",
                        case.state, case.uid.0, e
                    ))
                })?,
            });
            let mut case_node = ScheduledNode::new(kind, ScheduleInfoBuilder::new().build());
            lower_children(&child.children, ctx, system_grid)?
                .into_iter()
                .for_each(|child| {
                    case_node.add_child(tiny_samples(0), child);
                });
            root.add_child(tiny_samples(0), case_node);
        }
    }
    Ok(root)
}
