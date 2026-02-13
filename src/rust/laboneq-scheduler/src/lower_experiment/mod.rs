// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::error::{Error, Result};

use crate::experiment_context::ExperimentContext;
use crate::lower_experiment::local_context::LocalContext;
use crate::schedule_info::{RepetitionMode, ScheduleInfoBuilder};
use laboneq_ir::builders::SectionBuilder;
use laboneq_ir::{Acquire, ChangeOscillatorPhase, IrKind, Loop, LoopKind, PlayPulse};

use crate::utils::{compute_grid, lcm, round_to_grid};
use crate::{ParameterStore, ScheduledNode, SignalInfo};
use laboneq_dsl::operation::{
    Acquire as AcquireDsl, Delay as DelayDsl, LoopInfo, Operation, PlayPulse as PlayPulseDsl,
    PrngSetup as PrngSetupDsl, Section as SectionDsl,
};
use laboneq_dsl::types::{RepetitionMode as RepetitionModeDsl, SignalUid, ValueOrParameter};
use laboneq_dsl::{ExperimentNode, NodeChild};
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
) -> Result<ScheduledNode> {
    let system_grid = compute_grid(ctx.signals()).1;
    let mut local_ctx =
        LocalContext::new(&ctx.parameters, nt_parameters, system_grid, ctx.signals());

    let mut root = ScheduledNode::new(IrKind::Root, ScheduleInfoBuilder::new().grid(1).build());

    // Collect and sort signals by name to ensure deterministic ordering
    let mut concrete_signals: Vec<_> = ctx.signals().collect();
    concrete_signals.sort_by_key(|sig| {
        ctx.id_store
            .resolve(sig.uid().0)
            .unwrap_or("unknown")
            .to_string()
    });
    setup_initial_conditions(&mut root, &concrete_signals, nt_parameters, system_grid)?;

    let (children, reserved_signals) = lower_children(&node.children, ctx, &mut local_ctx)?;
    root.schedule.signals.extend(reserved_signals);

    for child in children {
        root.add_child(tiny_samples(0), child);
    }

    adjust_node_grids(&mut root);
    Ok(root)
}

/// Setup initial oscillator frequencies and voltage offsets
fn setup_initial_conditions<T: SignalInfo>(
    root: &mut ScheduledNode,
    signals: &[&T],
    nt_parameters: &ParameterStore,
    system_grid: TinySamples,
) -> Result<()> {
    let initial_osc_freq =
        handle_initial_oscillator_frequency(signals, nt_parameters, system_grid)?;
    let initial_local_osc_freq =
        handle_initial_local_oscillator_frequency(signals, nt_parameters, system_grid)?;
    let initial_voltage_offset =
        handle_initial_voltage_offset(signals, nt_parameters, system_grid)?;

    root.add_child(tiny_samples(0), initial_osc_freq);
    for child in initial_local_osc_freq {
        root.add_child(tiny_samples(0), child);
    }
    for child in initial_voltage_offset {
        root.add_child(tiny_samples(0), child);
    }
    Ok(())
}

/// Lower a single Experiment node into IR nodes.
///
/// Returns a vector of IR nodes and a set of reserved signals within those
/// nodes.
fn lower_to_ir_impl<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    local_ctx: &mut LocalContext,
) -> Result<(Vec<ScheduledNode>, HashSet<SignalUid>)> {
    match &node.kind {
        Operation::Section(section) => {
            let section_node = lower_section(section, &node.children, ctx, local_ctx)?;
            Ok((vec![section_node], HashSet::new()))
        }
        Operation::PrngLoop(_) | Operation::Sweep(_) | Operation::AveragingLoop(_) => {
            Ok((lower_sweep(node, ctx, local_ctx)?, HashSet::new()))
        }
        Operation::Reserve(reserve) => {
            let mut reserved = HashSet::new();
            reserved.insert(reserve.signal);
            Ok((vec![], reserved))
        }
        Operation::PlayPulse(play_pulse) => Ok((
            vec![lower_play_pulse(play_pulse, local_ctx)],
            HashSet::new(),
        )),
        Operation::Acquire(acquire) => {
            Ok((vec![lower_acquire(acquire, local_ctx)], HashSet::new()))
        }
        Operation::Delay(delay) => {
            let delay_node = lower_delay(delay, local_ctx);
            if delay.precompensation_clear {
                let precomp_node = create_precompensation_node(delay.signal, local_ctx);
                Ok((vec![precomp_node, delay_node], HashSet::new()))
            } else {
                Ok((vec![delay_node], HashSet::new()))
            }
        }
        Operation::ResetOscillatorPhase(reset) => {
            let signals = collect_signals_from_uids(&reset.signals, ctx)?;
            let reset_node = handle_reset_oscillator_phase(
                &signals,
                ctx,
                local_ctx.system_grid,
                local_ctx
                    .section_uid
                    .expect("Internal error: Phase reset not in a section"),
            )?;
            Ok((vec![reset_node], HashSet::new()))
        }
        Operation::Case(_) => Err(Error::new(
            "Internal error: Case must be used within a match block.",
        )),
        Operation::Match(_) => Ok((vec![lower_match(node, ctx, local_ctx)?], HashSet::new())),
        Operation::PrngSetup(obj) => {
            let setup_node = lower_prng_setup(obj, &node.children, ctx, local_ctx)?;
            Ok((vec![setup_node], HashSet::new()))
        }
        Operation::RealTimeBoundary => lower_children(&node.children, ctx, local_ctx),
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

/// Helper to collect signal references from UIDs
fn collect_signals_from_uids<'a, T: SignalInfo>(
    signal_uids: &[SignalUid],
    ctx: &'a ExperimentContext<T>,
) -> Result<Vec<&'a T>> {
    signal_uids.iter().map(|s| ctx.get_signal(s)).collect()
}

/// Lower a list of Experiment nodes into IR nodes.
///
/// Returns a vector of IR nodes and a set of reserved signals within those
/// nodes.
fn lower_children(
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo + Sized>,
    local_ctx: &mut LocalContext,
) -> Result<(Vec<ScheduledNode>, HashSet<SignalUid>)> {
    let mut all_nodes = Vec::with_capacity(children.len() * 2);
    let mut all_reserved = HashSet::new();

    for child in children {
        let (nodes, reserved) = lower_to_ir_impl(child, ctx, local_ctx)?;
        all_nodes.extend(nodes);
        all_reserved.extend(reserved);
    }

    Ok((all_nodes, all_reserved))
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

    let (children, reserved_signals) = local_ctx
        .with_loop(*loop_info.uid, loop_info.parameters, |new_ctx| {
            lower_children(&node.children, ctx, new_ctx)
        })
        .flatten()?;

    let mut loop_signals: HashSet<SignalUid> = children
        .iter()
        .flat_map(|child| &child.schedule.signals)
        .cloned()
        .collect();

    let preamble =
        create_loop_preamble(&loop_signals, &reserved_signals, &loop_info, ctx, local_ctx)?;

    loop_signals.extend(reserved_signals.iter().cloned());
    let iteration =
        create_loop_iteration(preamble, children, &loop_info, local_ctx, &loop_signals)?;

    let root_loop = create_root_loop(node, iteration, &loop_info)?;
    Ok(vec![root_loop])
}

/// Create the loop iteration preamble with oscillator and PPC setup
fn create_loop_preamble<T: SignalInfo>(
    loop_signals: &HashSet<SignalUid>,
    reserved_signals: &HashSet<SignalUid>,
    loop_info: &LoopInfo,
    ctx: &ExperimentContext<T>,
    local_ctx: &LocalContext,
) -> Result<ScheduledNode> {
    let mut preamble = ScheduledNode::new(
        IrKind::LoopIterationPreamble,
        ScheduleInfoBuilder::new()
            .signals(loop_signals.iter().cloned().collect())
            .build(),
    );

    let signal_refs =
        collect_signals_from_uids(&loop_signals.iter().cloned().collect::<Vec<_>>(), ctx)?;

    // Add oscillator frequency setup
    if let Some(set_osc_freq) = handle_set_oscillator_frequency(
        &signal_refs,
        loop_info.parameters.iter().collect(),
        local_ctx.system_grid,
    )? {
        preamble.add_child(tiny_samples(0), set_osc_freq);
    }

    // Add phase reset if needed
    if loop_info.reset_oscillator_phase && !signal_refs.is_empty() {
        let reset_node = handle_reset_oscillator_phase(
            &signal_refs,
            ctx,
            local_ctx.system_grid,
            *loop_info.uid,
        )?;
        preamble.add_child(tiny_samples(0), reset_node);
    }

    for child in &preamble.children {
        preamble.schedule.grid = lcm(
            preamble.schedule.grid.value(),
            child.node.schedule.grid.value(),
        )
        .into();
    }
    adjust_node_grids(&mut preamble);

    // Add PPC sweep steps
    let all_signals: Vec<_> = ctx.signals().collect();
    for ppc_step in
        handle_ppc_sweep_steps(&all_signals, loop_info.parameters, preamble.schedule.grid)?
    {
        preamble.add_child(tiny_samples(0), ppc_step);
    }

    // Add reserved signals to preamble to ensure it is scheduled before
    // the loop body
    preamble
        .schedule
        .signals
        .extend(reserved_signals.iter().cloned());
    Ok(preamble)
}

/// Create the loop iteration containing preamble and children
fn create_loop_iteration(
    preamble: ScheduledNode,
    children: Vec<ScheduledNode>,
    loop_info: &LoopInfo,
    local_ctx: &mut LocalContext,
    loop_signals: &HashSet<SignalUid>,
) -> Result<ScheduledNode> {
    let mut iteration = ScheduledNode::new(
        IrKind::LoopIteration,
        ScheduleInfoBuilder::new()
            // Escalate to the system grid.
            // TODO: We might want to relax this in the future.
            .grid(local_ctx.system_grid)
            .alignment_mode(*loop_info.alignment)
            .signals(loop_signals.iter().cloned().collect())
            .build(),
    );

    iteration.add_child(tiny_samples(0), preamble);
    for child in children {
        iteration.add_child(tiny_samples(0), child);
    }

    let (grid, sequencer_grid) =
        local_ctx.calculate_grids(iteration.schedule.signals.iter().cloned(), false, true);

    iteration.schedule.grid = grid;
    iteration.schedule.sequencer_grid = sequencer_grid;
    adjust_node_grids(&mut iteration);

    Ok(iteration)
}

/// Create the root loop node
fn create_root_loop(
    node: &ExperimentNode,
    iteration: ScheduledNode,
    loop_info: &LoopInfo,
) -> Result<ScheduledNode> {
    let loop_kind = match &node.kind {
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

    let loop_ir = Loop {
        uid: *loop_info.uid,
        iterations: loop_info.count,
        kind: loop_kind,
    };

    let mut schedule_builder = ScheduleInfoBuilder::new()
        .alignment_mode(*loop_info.alignment)
        .signals(iteration.schedule.signals.clone())
        .grid(iteration.schedule.grid)
        .sequencer_grid(iteration.schedule.sequencer_grid);

    if let Some(repetition_mode) = &loop_info.repetition_mode {
        schedule_builder =
            schedule_builder.repetition_mode(transform_repetition_mode(repetition_mode));
    }

    let mut root = ScheduledNode::new(IrKind::Loop(loop_ir), schedule_builder.build());
    root.add_child(tiny_samples(0), iteration);
    adjust_node_grids(&mut root);

    Ok(root)
}

fn transform_repetition_mode(mode: &RepetitionModeDsl) -> RepetitionMode {
    match mode {
        RepetitionModeDsl::Auto => RepetitionMode::Auto,
        RepetitionModeDsl::Constant { time } => RepetitionMode::Constant {
            time: seconds_to_tinysamples(*time),
        },
        RepetitionModeDsl::Fastest => RepetitionMode::Fastest,
    }
}

fn lower_section(
    section: &SectionDsl,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo>,
    local_ctx: &mut LocalContext,
) -> Result<ScheduledNode> {
    let mut ir_section = SectionBuilder::new(section.uid);
    for trig in &section.triggers {
        ir_section = ir_section.add_trigger(trig.signal, trig.state);
    }
    let ir_section = ir_section.build();

    let mut schedule_builder = ScheduleInfoBuilder::new()
        .alignment_mode(section.alignment)
        .play_after(section.play_after.clone());

    if let Some(length) = section.length {
        schedule_builder = schedule_builder.length(seconds_to_tinysamples(length));
    }

    let mut root = ScheduledNode::new(IrKind::Section(ir_section), schedule_builder.build());

    let (children, reserved_signals) = local_ctx.with_section(section.uid, |local_ctx| {
        lower_children(children, ctx, local_ctx)
    })?;

    root.schedule.signals.extend(reserved_signals);
    for child in children {
        root.add_child(tiny_samples(0), child);
    }

    let (grid, sequencer_grid) = local_ctx.calculate_grids(
        root.schedule.signals.iter().cloned(),
        !section.triggers.is_empty(),
        section.on_system_grid,
    );

    root.schedule.grid = grid;
    root.schedule.sequencer_grid = sequencer_grid;
    adjust_node_grids(&mut root);
    Ok(root)
}

fn lower_play_pulse(obj: &PlayPulseDsl, ctx: &LocalContext) -> ScheduledNode {
    let (grid, _) = ctx.signal_grids(&obj.signal);
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
        schedule = schedule.grid(grid);
        if let Some(length) = obj.length {
            match length {
                ValueOrParameter::Value(v) => {
                    let length = seconds_to_tinysamples(v);
                    schedule = schedule.length(round_to_grid(length.value(), grid.value()));
                }
                ValueOrParameter::Parameter(p) => {
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
        ScheduledNode::new(ir, ScheduleInfoBuilder::new().grid(grid).length(0).build())
    }
}

fn lower_acquire(obj: &AcquireDsl, ctx: &LocalContext) -> ScheduledNode {
    let integration_length =
        seconds_to_tinysamples(obj.length.expect("Expected Acquire to have length"));
    let (grid, _) = ctx.signal_grids(&obj.signal);
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
            .escalate_to_sequencer_grid(true)
            .build(),
    )
}

fn lower_delay(obj: &DelayDsl, ctx: &LocalContext) -> ScheduledNode {
    let ir = IrKind::Delay { signal: obj.signal };
    let (grid, _) = ctx.signal_grids(&obj.signal);
    let mut schedule = ScheduleInfoBuilder::new();
    match obj.time {
        ValueOrParameter::Value(v) => {
            let length = seconds_to_tinysamples(v);
            let length_tinysample = round_to_grid(length.value(), grid.value());
            schedule = schedule.length(length_tinysample);
        }
        ValueOrParameter::Parameter(p) => {
            schedule = schedule.length_param(p);
        }
        _ => {}
    }
    ScheduledNode::new(ir, schedule.grid(grid).build())
}

fn create_precompensation_node(signal: SignalUid, ctx: &LocalContext) -> ScheduledNode {
    let (grid, _) = ctx.signal_grids(&signal);
    ScheduledNode::new(
        IrKind::ClearPrecompensation { signal },
        ScheduleInfoBuilder::new().grid(grid).length(0).build(),
    )
}

fn lower_prng_setup(
    section: &PrngSetupDsl,
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

    let (children, reserved_signals) = local_ctx.with_section(section.uid, |local_ctx| {
        lower_children(children, ctx, local_ctx)
    })?;

    root.schedule.signals.extend(reserved_signals);
    for child in children {
        root.add_child(tiny_samples(0), child);
    }

    let (grid, sequencer_grid) =
        local_ctx.calculate_grids(root.schedule.signals.iter().cloned(), false, false);
    root.schedule.grid = grid;
    root.schedule.sequencer_grid = sequencer_grid;
    adjust_node_grids(&mut root);
    Ok(root)
}

/// Adjust the grids of the parent node based on its children's grids.
fn adjust_node_grids(parent: &mut ScheduledNode) {
    for child in parent.children.iter_mut() {
        parent.schedule.grid = lcm(
            parent.schedule.grid.value(),
            child.node.schedule.grid.value(),
        )
        .into();
        parent.schedule.sequencer_grid = lcm(
            parent.schedule.sequencer_grid.value(),
            child.node.schedule.sequencer_grid.value(),
        )
        .into();
        parent.schedule.compressed_loop_grid = lcm(
            parent.schedule.compressed_loop_grid.value(),
            child.node.schedule.compressed_loop_grid.value(),
        )
        .into();
        if child.node.schedule.escalate_to_sequencer_grid {
            parent.schedule.grid = lcm(
                parent.schedule.grid.value(),
                parent.schedule.sequencer_grid.value(),
            )
            .into();
        }
    }
}
