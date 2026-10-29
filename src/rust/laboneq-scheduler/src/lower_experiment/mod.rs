// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use std::collections::HashSet;
use std::num::NonZeroU32;

use crate::error::{Error, Result};

use crate::experiment_context::ExperimentContext;
use crate::lower_experiment::local_context::LocalContext;
use crate::schedule_info::{RepetitionMode, ScheduleInfoBuilder};
use crate::utils::{checked_round_to_grid, compute_grid, lcm};
use crate::{ScheduledNode, SignalInfo};
use laboneq_dsl::operation::{
    Acquire as AcquireDsl, Delay as DelayDsl, LoopInfo, Operation, PlayPulse as PlayPulseDsl,
    PrngSetup as PrngSetupDsl, Section as SectionDsl,
};
use laboneq_dsl::types::{
    RepetitionMode as RepetitionModeDsl, SectionTimingMode, SignalUid, ValueOrParameter,
};
use laboneq_dsl::{ExperimentNode, NodeChild};
use laboneq_ir::builders::SectionBuilder;
use laboneq_ir::{
    Acquire, ChangeOscillatorPhase, IrKind, Loop, LoopKind, ParameterLookup, PlayPulse,
};
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
    nt_parameters: &dyn ParameterLookup,
) -> Result<ScheduledNode> {
    let system_grid = compute_grid(ctx.signals())?.1;
    let mut local_ctx =
        LocalContext::new(&ctx.parameters, nt_parameters, system_grid, ctx.signals())?;

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

    let reserved_signals = lower_children(&mut root, &node.children, ctx, &mut local_ctx)?;
    root.schedule.signals.extend(reserved_signals);

    adjust_node_grids(&mut root);
    Ok(root)
}

/// Setup initial oscillator frequencies and voltage offsets
fn setup_initial_conditions<T: SignalInfo>(
    root: &mut ScheduledNode,
    signals: &[&T],
    nt_parameters: &dyn ParameterLookup,
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

/// Lower a single Experiment node, adding the resulting IR node(s) to `parent` as children,
/// and inserting any reserved signals into `reserved`.
fn lower_node<T: SignalInfo + Sized>(
    parent: &mut ScheduledNode,
    reserved: &mut HashSet<SignalUid>,
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    local_ctx: &mut LocalContext,
) -> Result<()> {
    let lowered = match &node.kind {
        Operation::Section(section) => lower_section(section, &node.children, ctx, local_ctx)?,
        Operation::PrngLoop(_) | Operation::Sweep(_) | Operation::AveragingLoop(_) => {
            lower_sweep(node, ctx, local_ctx)?
        }
        Operation::Reserve(reserve) => {
            reserved.insert(reserve.signal);
            return Ok(());
        }
        Operation::PlayPulse(play_pulse) => lower_play_pulse(play_pulse, local_ctx)?,
        Operation::Acquire(acquire) => lower_acquire(acquire, local_ctx)?,
        Operation::Delay(delay) => {
            if delay.precompensation_clear {
                parent.add_child(
                    tiny_samples(0),
                    create_precompensation_node(delay.signal, local_ctx),
                );
            }
            lower_delay(delay, local_ctx)?
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
            if local_ctx.section_timing_mode == SectionTimingMode::Strict
                && reset_node.schedule.try_length() > Some(tiny_samples(0))
            {
                return Err(Error::new(
                    "HW oscillator phase reset is not allowed in strict timing sections.",
                ));
            }
            reset_node
        }
        Operation::Case(_) => {
            return Err(Error::new(
                "Internal error: Case must be used within a match block.",
            ));
        }
        Operation::Match(_) => lower_match(node, ctx, local_ctx)?,
        Operation::PrngSetup(obj) => lower_prng_setup(obj, &node.children, ctx, local_ctx)?,
        Operation::RealTimeBoundary => {
            for child in &node.children {
                lower_node(parent, reserved, child, ctx, local_ctx)?;
            }
            return Ok(());
        }
        Operation::Root => {
            panic!(
                "Internal error: Scheduling is supported only for real-time part of an experiment."
            )
        }
        Operation::NearTimeCallback(_) => {
            panic!("Internal error: Near-time callbacks cannot exist in real-time.")
        }
        Operation::SetNode(_) => panic!("Internal error: Set node cannot exist in real-time."),
        Operation::Send(_)
        | Operation::MarkStale(_)
        | Operation::Ensure(_)
        | Operation::DoUntil(_) => {
            return Err(Error::new(
                "Internal error: HQCS operations must be lowered or refused before scheduling.",
            ));
        }
    };
    parent.add_child(tiny_samples(0), lowered);
    Ok(())
}

/// Helper to collect signal references from UIDs
fn collect_signals_from_uids<'a, T: SignalInfo>(
    signal_uids: &[SignalUid],
    ctx: &'a ExperimentContext<T>,
) -> Result<Vec<&'a T>> {
    signal_uids.iter().map(|s| ctx.get_signal(s)).collect()
}

/// Lower a list of Experiment nodes, attaching each resulting IR node directly
/// as a child of `parent`.
///
/// Returns the set of reserved signals within the lowered children.
fn lower_children(
    parent: &mut ScheduledNode,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo + Sized>,
    local_ctx: &mut LocalContext,
) -> Result<HashSet<SignalUid>> {
    let mut reserved = HashSet::new();
    for child in children {
        lower_node(parent, &mut reserved, child, ctx, local_ctx)?;
    }
    Ok(reserved)
}

/// Lower an Experiment loop node into an IR node.
fn lower_sweep<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    local_ctx: &mut LocalContext,
) -> Result<ScheduledNode> {
    let mut loop_info = node
        .kind
        .loop_info()
        .ok_or_else(|| Error::new("Expected a loop"))?;

    // If this is a parameter sweep, derive the loop's iteration count from the length of the parameter(s),
    // instead of taking `Sweep.count`. This way we account for parameter slicing due to chunking.
    if let Some(param) = loop_info.parameters.first() {
        let len = ctx.sweep_parameter(param)?.len();
        loop_info.count = NonZeroU32::new(len as u32)
            .ok_or_else(|| Error::new("Swept parameter must have at least one value"))?;
    }

    let mut iteration = ScheduledNode::new_with_capacity(
        IrKind::LoopIteration,
        ScheduleInfoBuilder::new()
            // Escalate to the system grid.
            // TODO: We might want to relax this in the future.
            .grid(local_ctx.system_grid)
            .alignment_mode(*loop_info.alignment)
            .section_timing_mode(*loop_info.section_timing_mode)
            .build(),
        node.children.len() + 1, // +1 for preamble
    );
    // Dummy / placeholder preamble
    // The preamble depends on the signals of the loop body,
    // so it will be replaced once the body is lowered.
    iteration.add_child(
        tiny_samples(0),
        ScheduledNode::new(
            IrKind::LoopIterationPreamble,
            ScheduleInfoBuilder::new().build(),
        ),
    );

    let reserved_signals = local_ctx
        .with_loop(
            *loop_info.uid,
            loop_info.parameters,
            *loop_info.section_timing_mode,
            |c| lower_children(&mut iteration, &node.children, ctx, c),
        )
        .flatten()?;

    let preamble = create_loop_preamble(
        &iteration.schedule.signals,
        &reserved_signals,
        &loop_info,
        ctx,
        local_ctx,
    )?;
    iteration.insert_child(0, tiny_samples(0), preamble);

    let (grid, sequencer_grid) =
        local_ctx.calculate_grids(iteration.schedule.signals.iter().cloned(), false, true);
    iteration.schedule.grid = grid;
    iteration.schedule.sequencer_grid = sequencer_grid;
    adjust_node_grids(&mut iteration);

    let root_loop = create_root_loop(node, iteration, &loop_info)?;
    Ok(root_loop)
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

    let signal_refs = loop_signals
        .iter()
        .map(|s| ctx.get_signal(s))
        .collect::<Result<Vec<_>>>()?;

    let strict = *loop_info.section_timing_mode == SectionTimingMode::Strict;

    // Add oscillator frequency setup
    if let Some(set_osc_freq) = handle_set_oscillator_frequency(
        &signal_refs,
        loop_info.parameters.iter().collect(),
        local_ctx.system_grid,
    )? {
        if strict && set_osc_freq.schedule.try_length() > Some(tiny_samples(0)) {
            return Err(Error::new(
                "HW oscillator frequency sweep is not allowed in strict timing sections.",
            ));
        }
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
        if strict && reset_node.schedule.try_length() > Some(tiny_samples(0)) {
            return Err(Error::new(
                "HW oscillator phase reset is not allowed in strict timing sections.",
            ));
        }
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
    let ppc_steps =
        handle_ppc_sweep_steps(&all_signals, loop_info.parameters, preamble.schedule.grid)?;
    if strict && !ppc_steps.is_empty() {
        return Err(Error::new(
            "SHFPPC sweeps are not allowed in strict timing sections.",
        ));
    }
    for ppc_step in ppc_steps {
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
        .sequencer_grid(iteration.schedule.sequencer_grid)
        .section_timing_mode(*loop_info.section_timing_mode);

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
        .play_after(section.play_after.clone())
        .section_timing_mode(section.section_timing_mode);

    if let Some(length) = section.length {
        schedule_builder = schedule_builder.length(seconds_to_tinysamples(length));
    }

    let mut root = ScheduledNode::new_with_capacity(
        IrKind::Section(ir_section),
        schedule_builder.build(),
        children.len(),
    );

    let reserved_signals =
        local_ctx.with_section(section.uid, section.section_timing_mode, |local_ctx| {
            lower_children(&mut root, children, ctx, local_ctx)
        })?;

    root.schedule.signals.extend(reserved_signals);

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

fn lower_play_pulse(obj: &PlayPulseDsl, ctx: &LocalContext) -> Result<ScheduledNode> {
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
                    let tinysample_length = seconds_to_tinysamples(v);
                    let rounded_length = checked_round_to_grid(
                        tinysample_length.value(),
                        grid.value(),
                        ctx.section_timing_mode,
                    )
                    .with_context(|| {
                        format!(
                            "While scheduling pulse of length {} on signal '{}' in section '{}'",
                            v,
                            obj.signal.0,
                            ctx.section_name()
                        )
                    })?;

                    schedule = schedule.length(rounded_length);
                }
                ValueOrParameter::Parameter(p) => {
                    schedule = schedule.length_param(p);
                }
                _ => {}
            }
        }
        Ok(ScheduledNode::new(ir, schedule.build()))
    } else {
        // no pulse id
        let ir = IrKind::ChangeOscillatorPhase(ChangeOscillatorPhase {
            signal: obj.signal,
            increment: obj.increment_oscillator_phase,
            set: obj.set_oscillator_phase,
        });
        Ok(ScheduledNode::new(
            ir,
            ScheduleInfoBuilder::new().grid(grid).length(0).build(),
        ))
    }
}

fn lower_acquire(obj: &AcquireDsl, ctx: &mut LocalContext) -> Result<ScheduledNode> {
    ctx.set_handle_signal(obj.handle, obj.signal)?;
    let raw_length = seconds_to_tinysamples(obj.length.expect("Expected Acquire to have length"));
    let (grid, _) = ctx.signal_grids(&obj.signal);
    let integration_length =
        checked_round_to_grid(raw_length.value(), grid.value(), ctx.section_timing_mode)
            .with_context(|| {
                format!(
                    "While scheduling acquisition on signal '{}' in section '{}'",
                    obj.signal.0,
                    ctx.section_name()
                )
            })?;
    Ok(ScheduledNode::new(
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
    ))
}

fn lower_delay(obj: &DelayDsl, ctx: &LocalContext) -> Result<ScheduledNode> {
    let ir = IrKind::Delay { signal: obj.signal };
    let (grid, _) = ctx.signal_grids(&obj.signal);
    let mut schedule = ScheduleInfoBuilder::new();
    match obj.time {
        ValueOrParameter::Value(v) => {
            let length = seconds_to_tinysamples(v);
            let length_tinysample = checked_round_to_grid(
                length.value(),
                grid.value(),
                ctx.section_timing_mode,
            )
            .map_err(|e| {
                Error::new(format!(
                    "While scheduling delay of length {} on signal '{}' in section '{}': {}",
                    v,
                    obj.signal.0,
                    ctx.section_name(),
                    e
                ))
            })?;
            schedule = schedule.length(length_tinysample);
        }
        ValueOrParameter::Parameter(p) => {
            schedule = schedule.length_param(p);
        }
        _ => {}
    }
    Ok(ScheduledNode::new(ir, schedule.grid(grid).build()))
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

    let mut root = ScheduledNode::new_with_capacity(
        IrKind::Section(ir_section),
        ScheduleInfoBuilder::new()
            .grid(local_ctx.system_grid)
            .build(),
        children.len(),
    );

    let reserved_signals =
        local_ctx.with_section(section.uid, SectionTimingMode::default(), |local_ctx| {
            lower_children(&mut root, children, ctx, local_ctx)
        })?;

    root.schedule.signals.extend(reserved_signals);

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
