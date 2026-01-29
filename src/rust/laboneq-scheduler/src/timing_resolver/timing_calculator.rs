// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use itertools::Itertools;
use laboneq_dsl::types::{HandleUid, SectionAlignment, SectionUid, SignalUid};
use std::collections::HashMap;

use laboneq_units::tinysample::{
    TinySamples, seconds_to_tinysamples, tiny_samples, tinysamples_to_seconds,
};
use num_integer::lcm;

use crate::error::{Error, Result};
use crate::ir::{IrKind, MatchTarget};
use crate::scheduled_node::NodeChild;
use crate::timing_resolver::timing_result::TimingWarning;
use crate::utils::{ceil_to_grid, floor_to_grid};
use crate::{RepetitionMode, ScheduledNode};

use super::{FeedbackCalculator, TimingResult};

/// Calculate the timing for the given node.
///
/// This function traverses the experiment tree depth-first, computing start times and durations
/// for each node based on its type and properties.
///
/// The underlying promise is that this function does not remove or add any nodes to the tree,
/// it only updates timing information and node order where applicable.
///
/// If feedback is used in the experiment, a feedback calculator must be provided to compute
/// latencies for match sections that depend on acquisition results.
///
/// Returns a `TimingResult` containing any timing warnings (e.g., sections delayed due to
/// feedback latency constraints) that were encountered during scheduling.
pub(crate) fn calculate_timing(
    node: &mut ScheduledNode,
    feedback_calculator: Option<&impl FeedbackCalculator>,
) -> Result<TimingResult> {
    let mut result = TimingResult::new();
    let mut ctx = Context {
        acquires_by_handle: HashMap::new(),
        feedback_calculator,
        active_section_stack: Vec::new(),
        timing_result: &mut result,
    };
    let absolute_start = tiny_samples(0);
    calculate_node_timing(node, absolute_start, &mut ctx)?;
    Ok(result)
}

struct FeedbackAcquisitionOperation {
    length: TinySamples,
    absolute_start: TinySamples,
    signal: SignalUid,
}

struct Context<'a, T: FeedbackCalculator> {
    acquires_by_handle: HashMap<HandleUid, FeedbackAcquisitionOperation>,
    feedback_calculator: Option<&'a T>,
    active_section_stack: Vec<SectionUid>,
    timing_result: &'a mut TimingResult,
}

impl<T: FeedbackCalculator> Context<'_, T> {
    fn enter_section(&mut self, section_uid: SectionUid) {
        self.active_section_stack.push(section_uid);
    }

    fn exit_section(&mut self) {
        self.active_section_stack.pop();
    }

    fn expect_section(&self) -> SectionUid {
        self.active_section_stack
            .last()
            .cloned()
            .expect("Internal error: No active section")
    }

    /// Register a feedback acquisition event.
    ///
    /// Only the latest acquisition for each handle is stored.
    fn register_feedback_acquisition_event(
        &mut self,
        handle: HandleUid,
        signal: SignalUid,
        length: TinySamples,
        absolute_start: TinySamples,
    ) {
        self.acquires_by_handle.insert(
            handle,
            FeedbackAcquisitionOperation {
                length,
                absolute_start,
                signal,
            },
        );
    }
}

/// Calculate timing for a single node.
fn calculate_node_timing(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    if let Some(s) = node.kind.section_info() {
        ctx.enter_section(*s.uid)
    };

    let absolute_start = match &node.kind {
        IrKind::Root => schedule_root(node, absolute_start, ctx),
        IrKind::LoopIterationPreamble => {
            schedule_loop_iteration_preamble(node)?;
            Ok(absolute_start)
        }
        IrKind::Section(_) => {
            schedule_section(node, absolute_start, ctx)?;
            Ok(absolute_start)
        }
        IrKind::Case(_) => {
            schedule_section(node, absolute_start, ctx)?;
            Ok(absolute_start)
        }
        IrKind::Loop(obj) => schedule_loop(node, obj.compressed(), absolute_start, ctx),
        IrKind::LoopIteration => {
            schedule_section(node, absolute_start, ctx)?;
            Ok(absolute_start)
        }
        IrKind::Match(_) => schedule_match(node, absolute_start, ctx),
        IrKind::Acquire(acquire) => {
            ctx.register_feedback_acquisition_event(
                acquire.handle,
                acquire.signal,
                node.length(),
                absolute_start,
            );
            Ok(absolute_start)
        }
        IrKind::PlayPulse(_) => Ok(absolute_start),
        IrKind::ChangeOscillatorPhase(_) => Ok(absolute_start),
        IrKind::Delay { .. } => Ok(absolute_start),
        IrKind::PpcStep(_) => Ok(absolute_start),
        IrKind::SetOscillatorFrequency(_) => Ok(absolute_start),
        IrKind::ResetOscillatorPhase { .. } => Ok(absolute_start),
        IrKind::InitialLocalOscillatorFrequency(_) => Ok(absolute_start),
        IrKind::InitialOscillatorFrequency(_) => Ok(absolute_start),
        IrKind::InitialVoltageOffset(_) => Ok(absolute_start),
        IrKind::ClearPrecompensation { .. } => Ok(absolute_start),
    }?;
    node.schedule.absolute_start = absolute_start;
    if node.kind.section_info().is_some() {
        ctx.exit_section();
    }
    Ok(absolute_start)
}

/// Update absolute offsets of a child node and its descendants.
fn update_absolute_start(
    child: &mut NodeChild,
    parent_absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) {
    let absolute_start = parent_absolute_start + child.offset;
    if absolute_start == child.node.schedule.absolute_start {
        return;
    }
    if let IrKind::Acquire(obj) = &child.node.kind {
        ctx.register_feedback_acquisition_event(
            obj.handle,
            obj.signal,
            child.node.length(),
            absolute_start,
        );
    }
    let child = child.node.make_mut();
    child.schedule.absolute_start = absolute_start;
    for child in child.children.iter_mut() {
        update_absolute_start(child, absolute_start, ctx);
    }
}

fn schedule_root(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let mut length = tiny_samples(0);
    for child in node.children.iter_mut() {
        calculate_node_timing(child.node.make_mut(), absolute_start, ctx)?;
        length = length.max(child.node.schedule.length());
    }
    length = ceil_to_grid(length.value(), node.schedule.grid.value()).into();
    node.schedule.resolve_length(length);
    Ok(absolute_start)
}

/// Schedule loop iteration preamble.
///
/// The loop iteration preamble's internal schedule obeys slightly different rules than a
/// regular section.
///
/// First, we schedule all PPC sweep steps and oscillator sweep steps. They always
/// can happen in parallel, because they they are not observable (they cannot
/// overlap with any pulses (or even acquisitions).
///
/// Finally, we insert the phase resets. These do have observable side effects:
///
/// 1. We must align them to the LO grid of 200 MHz
/// 2. Their timing w.r.t. to the pulses played in the experiment body is observable
///    as a phase offset.
///
/// For this reason, we place them on the right of the preamble, after all PPC and
/// oscillator frequency steps.
fn schedule_loop_iteration_preamble(node: &mut ScheduledNode) -> Result<()> {
    let mut length = tiny_samples(0);

    // Schedule all PPC and oscillator frequency steps at time 0
    for child in node.children.iter_mut() {
        if matches!(
            &child.node.kind,
            IrKind::PpcStep(_) | IrKind::SetOscillatorFrequency(_)
        ) {
            child.offset = tiny_samples(0);
            length = length.max(child.node.schedule.length());
        }
    }
    let previous_steps_end = ceil_to_grid(length.value(), node.schedule.grid.value());

    // Schedule all phase resets at the end of the previous steps
    for child in node.children.iter_mut() {
        if matches!(&child.node.kind, IrKind::ResetOscillatorPhase { .. }) {
            child.offset =
                ceil_to_grid(previous_steps_end, child.node.schedule.grid.value()).into();
        }
        length = length.max(child.offset + child.node.schedule.length());
    }

    node.schedule.resolve_length(length);
    Ok(())
}

fn schedule_section(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<()> {
    match node.schedule.alignment_mode {
        SectionAlignment::Left => {
            arrange_left_aligned(node, absolute_start, ctx)?;
        }
        SectionAlignment::Right => {
            arrange_right_aligned(node, absolute_start, ctx)?;
        }
    }
    calculate_section_length(node, absolute_start, ctx)?;
    Ok(())
}

/// Arrange children of a left-aligned section.
fn arrange_left_aligned(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<()> {
    let mut signal_start_constraints = HashMap::new();
    let mut play_after_constraints = HashMap::new();

    for child in node.children.iter_mut() {
        // Determine earliest possible start based on signals
        let mut start = tiny_samples(0);
        for signal in child.node.schedule.signals.iter() {
            let signal_start = signal_start_constraints
                .entry(*signal)
                .or_insert(tiny_samples(0));
            start = start.max(*signal_start);
        }

        // Handle play after constraints
        for play_after in child.node.schedule.play_after.iter() {
            let pa_start = play_after_constraints
                .get(play_after)
                .expect("Expected play after constraint");
            start = start.max(*pa_start);
        }

        // Align to grid
        start = ceil_to_grid(start.value(), child.node.schedule.grid.value()).into();

        // Calculate child timing
        let start = calculate_node_timing(child.node.make_mut(), absolute_start + start, ctx)?;

        // Assign calculated start offset relative to parent
        child.offset = start - absolute_start;

        // Update play after constraints
        if let Some(section_info) = &child.node.kind.section_info() {
            play_after_constraints.insert(
                section_info.uid,
                child.offset + child.node.schedule.length(),
            );
        }

        // Update signal constraints
        for signal in child.node.schedule.signals.iter() {
            if let Some(signal_offset) = signal_start_constraints.get_mut(signal) {
                *signal_offset = child.offset + child.node.schedule.length();
            }
        }
    }
    Ok(())
}

/// Arrange children of a right-aligned section.
fn arrange_right_aligned(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<()> {
    let mut signal_end_constraints = HashMap::new();
    let mut play_after_constraints: HashMap<&SectionUid, TinySamples> = HashMap::new();
    let mut relative_offset = tiny_samples(0);

    for child in node.children.iter_mut().rev() {
        let c_offset = calculate_node_timing(child.node.make_mut(), absolute_start, ctx)?;
        assert_eq!(
            (c_offset - absolute_start).value(),
            0,
            "Changing a start time in right-aligned sections is not supported"
        );

        // Determine earliest possible start based on signals
        let mut child_offset = tiny_samples(0);
        for signal in child.node.schedule.signals.iter() {
            let signal_start = signal_end_constraints
                .entry(*signal)
                .or_insert(tiny_samples(0));
            child_offset = child_offset.min(*signal_start);
        }
        child_offset = child_offset - child.node.schedule.length();

        // Handle play after constraints
        if let Some(section_info) = &child.node.kind.section_info()
            && let Some(play_before) = play_after_constraints.get(section_info.uid)
        {
            let pb_start = play_before.value() - child.node.schedule.length().value();
            child_offset = child_offset.value().min(pb_start).into();
        }

        // Align to grid
        child_offset = floor_to_grid(child_offset.value(), child.node.schedule.grid.value()).into();

        // Assign calculated start offset relative to parent
        child.offset = child_offset;
        relative_offset = relative_offset.min(child_offset);

        // Update play after constraints
        child.node.schedule.play_after.iter().for_each(|s| {
            play_after_constraints
                .entry(s)
                .and_modify(|e| {
                    if e.value() > child_offset.value() {
                        *e = child_offset;
                    }
                })
                .or_insert(child_offset);
        });

        // Update signal constraints
        for signal in child.node.schedule.signals.iter() {
            if let Some(end) = signal_end_constraints.get_mut(signal) {
                *end = child_offset;
            }
        }
    }

    // Align section start to grid
    let relative_offset: TinySamples =
        floor_to_grid(relative_offset.value(), node.schedule.grid.value()).into();

    // Adjust children offsets to be relative to section start
    for child in node.children.iter_mut() {
        child.offset = child.offset - relative_offset;
        update_absolute_start(child, absolute_start, ctx);
    }
    Ok(())
}

/// Calculates the length of a section based on its children's lengths and offsets.
///
/// If the section has a forced length, it checks if the children fit within that length.
/// If they do not fit, an error is returned. If they fit, the section length is set to the forced length,
/// and the children's offsets are adjusted if necessary.
fn calculate_section_length(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<()> {
    let mut children_length = tiny_samples(0);
    for child in node.children.iter_mut() {
        children_length = children_length.max(child.offset + child.node.schedule.length());
    }
    children_length = ceil_to_grid(children_length.value(), node.schedule.grid.value()).into();

    // Handle forced section length
    if let Some(max_length) = node.schedule.try_length() {
        let force_length: i64 = ceil_to_grid(max_length.value(), node.schedule.grid.value());
        check_for_exceeded_section_length(
            force_length.into(),
            children_length,
            ctx.expect_section(),
        )?;
        node.schedule.resolve_length(children_length);
        adjust_section_length(node, force_length.into(), absolute_start, ctx)?;
        return Ok(());
    }

    node.schedule.resolve_length(children_length);
    Ok(())
}

fn check_for_exceeded_section_length(
    requested: TinySamples,
    actual: TinySamples,
    section_uid: SectionUid,
) -> Result<()> {
    if actual > requested {
        return Err(Error::new(format!(
            "Content of section '{}' ({}) does not fit into the requested fixed section length ({}).",
            section_uid.0,
            tinysamples_to_seconds(actual),
            tinysamples_to_seconds(requested),
        )));
    }
    Ok(())
}

/// Adjusts the length of a section and shifts its children if necessary.
fn adjust_section_length(
    node: &mut ScheduledNode,
    new_length: TinySamples,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<()> {
    let new_length: TinySamples =
        ceil_to_grid(new_length.value(), node.schedule.grid.value()).into();

    if let Some(current_length) = node.schedule.try_length()
        && new_length == current_length
    {
        return Ok(());
    }

    // Shift children if right-aligned
    if node.schedule.alignment_mode == SectionAlignment::Right {
        let delta = new_length - node.schedule.length();
        for child in node.children.iter_mut() {
            child.offset = child.offset + delta;
            update_absolute_start(child, absolute_start, ctx);
        }
    }
    node.schedule.resolve_length(new_length);
    Ok(())
}

fn create_insufficient_repetition_time_error_message(
    repetition_time: TinySamples,
    current_length: TinySamples,
    section_uid: SectionUid,
    iteration: usize,
) -> String {
    format!(
        "Specified repetition time ({}) is insufficient to fit the contents of '{}', iteration {} ({})",
        tinysamples_to_seconds(repetition_time),
        section_uid.0,
        iteration,
        tinysamples_to_seconds(current_length),
    )
}

fn schedule_compressed_loop(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let IrKind::Loop(loop_obj) = &node.kind else {
        unreachable!()
    };
    let iteration = node
        .children
        .iter_mut()
        .exactly_one()
        .expect("Expected exactly one iteration in compressed loop.");
    calculate_node_timing(iteration.node.make_mut(), absolute_start, ctx)?;
    let mut length = iteration.node.length();
    if let Some(RepetitionMode::Constant { time }) = &node.schedule.repetition_mode {
        let adjusted_time = ceil_to_grid(time.value(), node.schedule.grid.value());
        if adjusted_time < length.value() {
            return Err(Error::new(
                create_insufficient_repetition_time_error_message(*time, length, loop_obj.uid, 0),
            ));
        }
        length = *time;
    }
    let grid = lcm(
        node.schedule.grid.value(),
        node.schedule.compressed_loop_grid.value(),
    );
    length = ceil_to_grid(length.value(), grid).into();
    adjust_section_length(iteration.node.make_mut(), length, absolute_start, ctx)?;
    iteration.offset = tiny_samples(0);
    node.schedule
        .resolve_length(length * loop_obj.iterations as i64);
    Ok(absolute_start)
}

fn schedule_generic_loop(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let mut child_offset = 0;
    for child in node.children.iter_mut() {
        child.offset = child_offset.into();
        calculate_node_timing(
            child.node.make_mut(),
            absolute_start + child_offset.into(),
            ctx,
        )?;
        let child_length = ceil_to_grid(
            child.node.schedule.length().value(),
            node.schedule.grid.value(),
        );
        adjust_section_length(
            child.node.make_mut(),
            child_length.into(),
            absolute_start,
            ctx,
        )?;
        child_offset += child_length;
    }
    Ok(absolute_start)
}

/// Schedule loop with auto repetition mode.
///
/// Each iteration is scheduled to have the same length as the longest iteration.
fn schedule_auto_repetition_mode_loop(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let IrKind::Loop(loop_obj) = &node.kind else {
        unreachable!()
    };
    let mut longest_iteration = 0;
    let mut child_offset = 0;
    for child in node.children.iter_mut() {
        child.offset = child_offset.into();
        calculate_node_timing(
            child.node.make_mut(),
            absolute_start + child_offset.into(),
            ctx,
        )?;
        let length = ceil_to_grid(
            child.node.schedule.length().value(),
            node.schedule.grid.value(),
        );
        child_offset += length;
        longest_iteration = longest_iteration.max(length);
    }

    // Adjust all iterations to the longest length
    for (iteration, child) in node.children.iter_mut().enumerate() {
        adjust_section_length(
            child.node.make_mut(),
            longest_iteration.into(),
            absolute_start,
            ctx,
        )?;
        child.offset = (longest_iteration * iteration as i64).into();
        update_absolute_start(child, absolute_start, ctx);
    }

    // Set loop length
    node.schedule
        .resolve_length((longest_iteration * loop_obj.iterations as i64).into());
    Ok(absolute_start)
}

fn schedule_constant_repetition_mode_loop(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let IrKind::Loop(loop_obj) = &node.kind else {
        unreachable!()
    };
    let repetition_time = match &node.schedule.repetition_mode {
        Some(RepetitionMode::Constant { time }) => *time,
        _ => unreachable!(),
    };
    let adjusted_time = ceil_to_grid(repetition_time.value(), node.schedule.grid.value());
    let mut child_offset = 0;
    for (iteration, child) in node.children.iter_mut().enumerate() {
        child.offset = child_offset.into();
        calculate_node_timing(
            child.node.make_mut(),
            absolute_start + child_offset.into(),
            ctx,
        )?;
        if adjusted_time < child.node.length().value() {
            return Err(Error::new(
                create_insufficient_repetition_time_error_message(
                    repetition_time,
                    child.node.length(),
                    loop_obj.uid,
                    iteration,
                ),
            ));
        }
        adjust_section_length(
            child.node.make_mut(),
            adjusted_time.into(),
            absolute_start,
            ctx,
        )?;
        child_offset += adjusted_time;
    }
    Ok(absolute_start)
}

fn schedule_loop(
    node: &mut ScheduledNode,
    is_compressed: bool,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    if is_compressed {
        schedule_compressed_loop(node, absolute_start, ctx)?;
    } else {
        match &node.schedule.repetition_mode {
            Some(RepetitionMode::Constant { .. }) => {
                schedule_constant_repetition_mode_loop(node, absolute_start, ctx)?;
            }
            Some(RepetitionMode::Auto) => {
                schedule_auto_repetition_mode_loop(node, absolute_start, ctx)?;
            }
            _ => {
                schedule_generic_loop(node, absolute_start, ctx)?;
            }
        }
        calculate_section_length(node, absolute_start, ctx)?;
    }
    Ok(absolute_start)
}

fn schedule_match(
    node: &mut ScheduledNode,
    absolute_start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let IrKind::Match(match_info) = &node.kind else {
        unreachable!()
    };
    let absolute_start = if let MatchTarget::Handle(_) = &match_info.target {
        schedule_feedback_match(node, absolute_start, ctx)?
    } else {
        absolute_start
    };
    for child in node.children.iter_mut() {
        calculate_node_timing(child.node.make_mut(), absolute_start, ctx)?;
    }
    calculate_section_length(node, absolute_start, ctx)?;
    Ok(absolute_start)
}

fn schedule_feedback_match(
    node: &mut ScheduledNode,
    start: TinySamples,
    ctx: &mut Context<impl FeedbackCalculator>,
) -> Result<TinySamples> {
    let IrKind::Match(match_info) = &node.kind else {
        unreachable!()
    };
    let MatchTarget::Handle(handle) = &match_info.target else {
        unreachable!()
    };
    // Get the latest acquire for the handle
    let latest_acquire = ctx
        .acquires_by_handle
        .get(handle)
        .expect("Acquires for handle not found");

    // Compute earliest execute time based on feedback latency
    let earliest_execute_table_entry = ctx
        .feedback_calculator
        .expect("Feedback calculator not set")
        .compute_feedback_latency(
            tinysamples_to_seconds(latest_acquire.absolute_start),
            tinysamples_to_seconds(latest_acquire.length),
            match_info.local,
            latest_acquire.signal,
            node.schedule.signals.iter().cloned(),
        )
        .map_err(Error::new)?;
    let earliest_execute_table_entry = seconds_to_tinysamples(earliest_execute_table_entry);
    let earliest_execute_table_entry_ts = ceil_to_grid(
        earliest_execute_table_entry.value(),
        node.schedule.grid.value(),
    );

    // Warn if start time is shifted
    if earliest_execute_table_entry_ts > start.value() {
        ctx.timing_result
            .add_warning(TimingWarning::MatchStartShifted {
                section_uid: match_info.uid,
                delay: tinysamples_to_seconds(
                    (earliest_execute_table_entry_ts - start.value()).into(),
                ),
            });
    }

    let start = start.max(earliest_execute_table_entry_ts.into());
    Ok(start)
}
