// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::experiment::types::{self as experiment_types};
use crate::experiment::types::{Operation, SignalUid};
use crate::experiment::{ExperimentNode, NodeChild};

use crate::experiment_context::ExperimentContext;
use crate::ir::{Acquire, IrKind, Loop, PlayPulse, Section, Trigger};
use crate::lower_experiment::local_context::LocalContext;
use crate::schedule_info::ScheduleInfoBuilder;

use crate::utils::{lcm, round_to_grid, signal_grid};
use crate::{ParameterStore, ScheduledNode, SignalInfo};
use laboneq_units::tinysample::{TinySamples, seconds_to_tinysamples, tiny_samples};

mod local_context;
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
mod match_case;
use match_case::lower_match;

/// Lowering of [ExperimentNode] to [ScheduledNode].
///
/// The pass will apply the given near-time parameters where applicable.
pub(crate) fn lower_to_ir<T: SignalInfo + Sized>(
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

    let mut local_ctx = LocalContext::new(&ctx.parameters, nt_parameters, system_grid);

    lower_to_ir_impl(node, ctx, &mut local_ctx)?
        .into_iter()
        .for_each(|child| {
            root.add_child(tiny_samples(0), child);
        });
    Ok(root)
}

fn lower_to_ir_impl<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    local_ctx: &mut LocalContext,
) -> Result<Vec<ScheduledNode>> {
    match &node.kind {
        Operation::Section(section) => Ok(vec![lower_section(
            section,
            &node.children,
            ctx,
            local_ctx,
        )?]),
        Operation::PrngLoop(_) => lower_sweep(node, ctx, local_ctx),
        Operation::Sweep(_) => lower_sweep(node, ctx, local_ctx),
        Operation::AveragingLoop(_) => lower_sweep(node, ctx, local_ctx),
        Operation::Reserve(reserve) => Ok(vec![lower_reserve(reserve)]),
        Operation::PlayPulse(play_pulse) => Ok(vec![lower_play_pulse(play_pulse)]),
        Operation::Acquire(acquire) => Ok(vec![lower_acquire(acquire, ctx)]),
        Operation::Delay(delay) => Ok(vec![lower_delay(delay, ctx)]),
        Operation::ResetOscillatorPhase(reset) => {
            let signals = reset
                .signals
                .iter()
                .map(|s| ctx.get_signal(s))
                .collect::<Result<Vec<&T>>>()?;
            Ok(vec![handle_reset_oscillator_phase(
                &signals,
                ctx,
                local_ctx.system_grid,
                local_ctx.section_uid.expect("Phase reset not in a section"),
            )?])
        }
        Operation::Case(_) => Err(Error::new("Case must be used within a match block.")),
        Operation::Match(_) => Ok(vec![lower_match(node, ctx, local_ctx)?]),
        Operation::PrngSetup(obj) => local_ctx.with_section(obj.uid, |local_ctx| {
            let children = lower_children(&node.children, ctx, local_ctx)?;
            let mut ir_node = ScheduledNode::new(
                IrKind::NotYetImplemented,
                ScheduleInfoBuilder::new().grid(1).build(),
            );
            children.into_iter().for_each(|child_node| {
                ir_node.add_child(tiny_samples(0), child_node);
            });
            Ok(vec![ir_node])
        }),
        Operation::Root | Operation::RealTimeBoundary => {
            let children = lower_children(&node.children, ctx, local_ctx)?;
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
    local_ctx: &mut LocalContext,
) -> Result<Vec<ScheduledNode>> {
    let mut result = Vec::new();
    for child in children {
        let nodes = lower_to_ir_impl(child, ctx, local_ctx)?;
        result.extend(nodes);
    }
    Ok(result)
}

/// Lower Experiment loop nodes into IR nodes.
fn lower_sweep<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    local_ctx: &mut LocalContext,
) -> Result<Vec<ScheduledNode>> {
    let loop_info = node
        .kind
        .loop_info()
        .ok_or_else(|| Error::new("Expected a loop"))?;

    let children = local_ctx
        .with_loop(*loop_info.uid, loop_info.parameters, |new_ctx| {
            lower_children(&node.children, ctx, new_ctx)
        })
        .flatten()?;
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
        local_ctx.system_grid,
    )? {
        preamble.add_child(tiny_samples(0), set_osc_freq);
    }
    let mut grid = 1;
    if loop_info.reset_oscillator_phase && !signal_vec.is_empty() {
        let reset_osc_node =
            handle_reset_oscillator_phase(&signal_vec, ctx, local_ctx.system_grid, *loop_info.uid)?;
        if reset_osc_node.schedule.grid.value() != grid {
            // On SHFxx, we align the phase reset with the LO granularity (100 MHz)
            grid = lcm(grid, reset_osc_node.schedule.grid.value());
        }
        preamble.add_child(tiny_samples(0), reset_osc_node);
    }
    for ppc_step in handle_ppc_sweep_steps(
        &ctx.signals().collect::<Vec<_>>(), // Use all the signals present in the experiment
        loop_info.parameters,
        tiny_samples(lcm(grid, local_ctx.system_grid.value())),
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
    local_ctx: &mut LocalContext,
) -> Result<ScheduledNode> {
    let children = local_ctx.with_section(section.uid, |local_ctx| {
        lower_children(children, ctx, local_ctx)
    })?;
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

fn lower_acquire(
    obj: &experiment_types::Acquire,
    ctx: &ExperimentContext<impl SignalInfo>,
) -> ScheduledNode {
    let integration_length =
        seconds_to_tinysamples(obj.length.expect("Expected Acquire to have length"));
    let grid = signal_grid(ctx.get_signal(&obj.signal).unwrap());
    let integration_length = round_to_grid(integration_length.value(), grid.value());
    ScheduledNode::new(
        IrKind::Acquire(Acquire {
            signal: obj.signal,
            handle: obj.handle,
            integration_length: integration_length.into(),
            kernels: obj.kernel.clone(),
            parameters: obj.parameters.clone(),
            pulse_parameters: obj.pulse_parameters.clone(),
        }),
        ScheduleInfoBuilder::new()
            .grid(grid)
            .length(integration_length)
            .build(),
    )
}

fn lower_delay(
    obj: &experiment_types::Delay,
    ctx: &ExperimentContext<impl SignalInfo>,
) -> ScheduledNode {
    let ir = IrKind::Delay { signal: obj.signal };
    let grid = signal_grid(ctx.get_signal(&obj.signal).unwrap());
    let mut schedule = ScheduleInfoBuilder::new();
    match obj.time {
        experiment_types::ValueOrParameter::Value(v) => {
            let length = seconds_to_tinysamples(v);
            let length_tinysample = round_to_grid(length.value(), grid.value());
            schedule = schedule.length(length_tinysample);
        }
        experiment_types::ValueOrParameter::Parameter(p) => {
            schedule = schedule.length_param(p);
        }
        _ => {}
    }
    let mut node = ScheduledNode::new(ir, schedule.grid(grid).build());
    // Add precompensation clear as the child of the delay to ensure it is scheduled
    // at the same time as the delay.
    if obj.precompensation_clear {
        let precomp_node = ScheduledNode::new(
            IrKind::ClearPrecompensation { signal: obj.signal },
            ScheduleInfoBuilder::new().grid(grid).length(0).build(),
        );
        node.add_child(tiny_samples(0), precomp_node);
    }
    node
}
