// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::experiment::types::{self as experiment_types};
use crate::experiment::types::{Operation, SignalUid};
use crate::experiment::{ExperimentNode, NodeChild};

use crate::experiment_context::ExperimentContext;
use crate::ir::builders::SectionBuilder;
use crate::ir::{Acquire, ChangeOscillatorPhase, IrKind, Loop, LoopKind, PlayPulse};
use crate::lower_experiment::local_context::LocalContext;
use crate::schedule_info::{RepetitionMode, ScheduleInfoBuilder};

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
    let mut root = ScheduledNode::new(IrKind::Root, ScheduleInfoBuilder::new().grid(1).build());
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
        Operation::PlayPulse(play_pulse) => Ok(vec![lower_play_pulse(play_pulse, ctx)]),
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
                local_ctx
                    .section_uid
                    .expect("Internal error: Phase reset not in a section"),
            )?])
        }
        Operation::Case(_) => Err(Error::new(
            "Internal error: Case must be used within a match block.",
        )),
        Operation::Match(_) => Ok(vec![lower_match(node, ctx, local_ctx)?]),
        Operation::PrngSetup(obj) => {
            Ok(vec![lower_prng_setup(obj, &node.children, ctx, local_ctx)?])
        }
        Operation::RealTimeBoundary => {
            // Inline the children of the real-time boundary since scheduling is only done
            // for the real-time part of the experiment.
            lower_children(&node.children, ctx, local_ctx)
        }
        Operation::Root => {
            panic!(
                "Internal error: Scheduling is supported only for real-time part of an experiment."
            )
        }
        Operation::NearTimeCallback => {
            panic!("Internal error: Near-time callbacks cannot exist in real-time.")
        }
        Operation::SetNode => panic!("Internal error: Set node cannot exist in real-time."),
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
    let mut schedule_builder = ScheduleInfoBuilder::new()
        .grid(1)
        .alignment_mode(*loop_info.alignment);
    if let Some(repetition_mode) = &loop_info.repetition_mode {
        schedule_builder =
            schedule_builder.repetition_mode(transform_repetition_mode(repetition_mode));
    }
    let schedule = schedule_builder.build();
    let kind = match &node.kind {
        Operation::AveragingLoop(obj) => LoopKind::Averaging {
            mode: obj.averaging_mode,
        },
        Operation::Sweep(_) => LoopKind::Sweeping {
            parameters: loop_info.parameters.to_vec(),
        },
        Operation::PrngLoop(obj) => LoopKind::Prng {
            sample_uid: obj.sample_uid,
        },
        _ => return Err(Error::new("Internal error: Expected a loop operation.")),
    };
    let mut root = ScheduledNode::new(
        IrKind::Loop(Loop {
            uid: *loop_info.uid,
            iterations: loop_info.count as usize,
            kind,
        }),
        schedule,
    );
    root.add_child(tiny_samples(0), iteration);
    Ok(vec![root])
}

fn transform_repetition_mode(mode: &experiment_types::RepetitionMode) -> RepetitionMode {
    match mode {
        experiment_types::RepetitionMode::Auto => RepetitionMode::Auto,
        experiment_types::RepetitionMode::Constant { time } => RepetitionMode::Constant {
            time: seconds_to_tinysamples(*time),
        },
        experiment_types::RepetitionMode::Fastest => RepetitionMode::Fastest,
    }
}

fn lower_section(
    section: &experiment_types::Section,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo>,
    local_ctx: &mut LocalContext,
) -> Result<ScheduledNode> {
    let mut ir_section = SectionBuilder::new(section.uid);
    for trig in &section.triggers {
        ir_section = ir_section.add_trigger(trig.signal, trig.state);
    }
    let mut root = ScheduledNode::new(
        IrKind::Section(ir_section.build()),
        ScheduleInfoBuilder::new()
            .grid(1)
            .alignment_mode(section.alignment)
            .build(),
    );

    let children = local_ctx.with_section(section.uid, |local_ctx| {
        lower_children(children, ctx, local_ctx)
    })?;
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

fn lower_play_pulse(
    obj: &experiment_types::PlayPulse,
    ctx: &ExperimentContext<impl SignalInfo>,
) -> ScheduledNode {
    if let Some(pulse) = obj.pulse {
        let ir = IrKind::PlayPulse(PlayPulse {
            signal: obj.signal,
            pulse,
            amplitude: obj.amplitude,
            phase: obj.phase,
            increment_oscillator_phase: obj.increment_oscillator_phase,
            set_oscillator_phase: obj.set_oscillator_phase,
            parameters: obj.parameters.clone(),
            pulse_parameters: obj.pulse_parameters.clone(),
            markers: obj.markers.clone(),
        });
        let mut schedule = ScheduleInfoBuilder::new();
        let grid = signal_grid(ctx.get_signal(&obj.signal).unwrap());
        schedule = schedule.grid(grid);
        if let Some(length) = obj.length {
            match length {
                experiment_types::ValueOrParameter::Value(v) => {
                    let length = seconds_to_tinysamples(v);
                    schedule = schedule.length(round_to_grid(length.value(), grid.value()));
                }
                experiment_types::ValueOrParameter::Parameter(p) => {
                    schedule = schedule.length_param(p);
                }
                _ => {}
            }
        }
        ScheduledNode::new(ir, schedule.build())
    } else {
        let ir = IrKind::ChangeOscillatorPhase(ChangeOscillatorPhase {
            signal: obj.signal,
            increment: obj.increment_oscillator_phase,
            set: obj.set_oscillator_phase,
        });
        let grid = signal_grid(ctx.get_signal(&obj.signal).unwrap());
        ScheduledNode::new(ir, ScheduleInfoBuilder::new().grid(grid).length(0).build())
    }
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

fn lower_prng_setup(
    section: &experiment_types::PrngSetup,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo>,
    local_ctx: &mut LocalContext,
) -> Result<ScheduledNode> {
    let ir_section = SectionBuilder::new(section.uid)
        .prng_setup(section.range, section.seed)
        .build();
    let mut root = ScheduledNode::new(
        IrKind::Section(ir_section),
        ScheduleInfoBuilder::new()
            .grid(local_ctx.system_grid)
            .build(),
    );
    let children = local_ctx.with_section(section.uid, |local_ctx| {
        lower_children(children, ctx, local_ctx)
    })?;
    children.into_iter().for_each(|child| {
        root.add_child(tiny_samples(0), child);
    });
    Ok(root)
}
