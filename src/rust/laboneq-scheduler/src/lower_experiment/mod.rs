// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::num::NonZeroU32;

use anyhow::Context;
use laboneq_dsl::hqcs::Predicate;
use laboneq_dsl::operation::{
    Acquire as AcquireDsl, Delay as DelayDsl, LoopInfo, MarkStaleTarget, Operation,
    PlayPulse as PlayPulseDsl, PrngSetup as PrngSetupDsl, Section as SectionDsl,
};
use laboneq_dsl::types::{
    PulseUid, RepetitionMode as RepetitionModeDsl, SectionTimingMode, SignalUid, ValueOrParameter,
};
use laboneq_dsl::{ExperimentNode, NodeChild};
use laboneq_ir::builders::SectionBuilder;
use laboneq_ir::{
    Acquire, ChangeOscillatorPhase, IrKind, Loop, LoopKind, ParameterLookup, PlayPulse,
};
use laboneq_units::tinysample::{seconds_to_tinysamples, tiny_samples};

use crate::error::{Error, Result};
use crate::experiment_context::ExperimentContext;
use crate::lower_experiment::local_context::LocalContext;
use crate::schedule_info::{Access, CoprocessorValue, RepetitionMode, ScheduleInfoBuilder};
use crate::signal_grids::SignalGrids;
use crate::utils::checked_round_to_grid;
use crate::{ScheduledNode, SignalInfo};

mod local_context;
mod match_case;
mod reset_phase;
use match_case::lower_match;

pub(crate) use reset_phase::handle_reset_oscillator_phase;

/// Lowering of [ExperimentNode] to [ScheduledNode].
///
/// The pass will apply the given near-time parameters where applicable.
pub(crate) fn lower_to_ir<T: SignalInfo + Sized>(
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    nt_parameters: &dyn ParameterLookup,
    grids: &SignalGrids,
) -> Result<ScheduledNode> {
    let mut local_ctx = LocalContext::new(&ctx.parameters, nt_parameters, grids);
    collect_updated_pulses(node, &mut local_ctx.updated_pulses);

    let mut root = ScheduledNode::new(IrKind::Root, ScheduleInfoBuilder::new().build());
    lower_children(&mut root, &node.children, ctx, &mut local_ctx)?;
    Ok(root)
}

fn collect_updated_pulses(node: &ExperimentNode, pulses: &mut HashSet<PulseUid>) {
    if let Operation::MarkStale(mark_stale) = &node.kind
        && let MarkStaleTarget::Pulse(pulse) = mark_stale.target
    {
        pulses.insert(pulse);
    }
    for child in &node.children {
        collect_updated_pulses(child, pulses);
    }
}

/// Lower a single Experiment node, adding the resulting IR node(s) to `parent` as children.
fn lower_node<T: SignalInfo + Sized>(
    parent: &mut ScheduledNode,
    node: &ExperimentNode,
    ctx: &ExperimentContext<T>,
    local_ctx: &mut LocalContext,
) -> Result<()> {
    let lowered = match &node.kind {
        Operation::Section(section) => lower_section(section, &node.children, ctx, local_ctx)?,
        Operation::PrngLoop(_) | Operation::Sweep(_) | Operation::AveragingLoop(_) => {
            lower_sweep(node, ctx, local_ctx)?
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
            handle_reset_oscillator_phase(
                &signals,
                ctx,
                local_ctx.grids.system_grid(),
                local_ctx
                    .section_uid
                    .expect("Internal error: Phase reset not in a section"),
            )?
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
                lower_node(parent, child, ctx, local_ctx)?;
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
        Operation::MarkStale(mark_stale) => ScheduledNode::new(
            IrKind::MarkStale(mark_stale.target),
            ScheduleInfoBuilder::new()
                .length(0)
                .access(mark_stale.target, Access::Write)
                .build(),
        ),
        Operation::Expect(expect) => {
            let value = match expect.condition {
                Predicate::Comparison { variable, .. } => CoprocessorValue::Variable(variable),
                Predicate::IsLive(target) => target.into(),
            };
            ScheduledNode::new(
                IrKind::Expect(expect.condition),
                ScheduleInfoBuilder::new()
                    .length(0)
                    .access(value, Access::Read)
                    .build(),
            )
        }
        Operation::Send(_) | Operation::DoUntil(_) => {
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
fn lower_children(
    parent: &mut ScheduledNode,
    children: &[NodeChild],
    ctx: &ExperimentContext<impl SignalInfo + Sized>,
    local_ctx: &mut LocalContext,
) -> Result<()> {
    for child in children {
        lower_node(parent, child, ctx, local_ctx)?;
    }
    Ok(())
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
            .signals(node.kind.reserved_signals().iter().copied().collect())
            .alignment_mode(*loop_info.alignment)
            .section_timing_mode(*loop_info.section_timing_mode)
            .build(),
        node.children.len() + 1, // +1 for preamble
    );
    local_ctx
        .with_loop(
            *loop_info.uid,
            loop_info.parameters,
            *loop_info.section_timing_mode,
            |c| lower_children(&mut iteration, &node.children, ctx, c),
        )
        .flatten()?;

    let root_loop = create_root_loop(node, iteration, &loop_info)?;
    Ok(root_loop)
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
        .section_timing_mode(*loop_info.section_timing_mode)
        .reset_oscillator_phase(loop_info.reset_oscillator_phase);

    if let Some(repetition_mode) = &loop_info.repetition_mode {
        schedule_builder =
            schedule_builder.repetition_mode(transform_repetition_mode(repetition_mode));
    }

    let mut root = ScheduledNode::new(IrKind::Loop(loop_ir), schedule_builder.build());
    root.add_child(tiny_samples(0), iteration);

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
        .signals(section.reserved_signals.iter().copied().collect())
        .alignment_mode(section.alignment)
        .play_after(section.play_after.clone())
        .section_timing_mode(section.section_timing_mode)
        .on_system_grid(section.on_system_grid);

    if let Some(length) = section.length {
        schedule_builder = schedule_builder.length(seconds_to_tinysamples(length));
    }

    let mut root = ScheduledNode::new_with_capacity(
        IrKind::Section(ir_section),
        schedule_builder.build(),
        children.len(),
    );

    local_ctx.with_section(section.uid, section.section_timing_mode, |local_ctx| {
        lower_children(&mut root, children, ctx, local_ctx)
    })?;
    Ok(root)
}

fn lower_play_pulse(obj: &PlayPulseDsl, ctx: &LocalContext) -> Result<ScheduledNode> {
    let (grid, _) = ctx.grids.of(&obj.signal);
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
        if ctx.updated_pulses.contains(&pulse) {
            schedule = schedule.access(CoprocessorValue::Pulse(pulse), Access::Read);
        }
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

fn lower_acquire(obj: &AcquireDsl, ctx: &LocalContext) -> Result<ScheduledNode> {
    let raw_length = seconds_to_tinysamples(obj.length.expect("Expected Acquire to have length"));
    let (grid, _) = ctx.grids.of(&obj.signal);
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
            acquisition_type: obj
                .acquisition_type
                .expect("Expected Acquire to have acquisition type"),
        }),
        ScheduleInfoBuilder::new()
            .grid(grid)
            .length(integration_length)
            .build(),
    ))
}

fn lower_delay(obj: &DelayDsl, ctx: &LocalContext) -> Result<ScheduledNode> {
    let ir = IrKind::Delay { signal: obj.signal };
    let (grid, _) = ctx.grids.of(&obj.signal);
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
    let (grid, _) = ctx.grids.of(&signal);
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
            .signals(section.reserved_signals.iter().copied().collect())
            .build(),
        children.len(),
    );

    local_ctx.with_section(section.uid, SectionTimingMode::default(), |local_ctx| {
        lower_children(&mut root, children, ctx, local_ctx)
    })?;
    Ok(root)
}
