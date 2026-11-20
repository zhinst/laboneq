// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_dsl::types::{HandleUid, SectionAlignment, SectionTimingMode, SignalUid};
use laboneq_ir::{IrKind, Match, MatchTarget};
use laboneq_units::tinysample::tiny_samples;

use crate::error::{Error, Result};
use crate::{RepetitionMode, ScheduleInfo, ScheduledNode};

/// Validate the IR for correctness.
///
/// * Checks that no acquisitions are present within match statements
///   when matching against targets other than sweep parameters.
///
/// * Ensures that match statements with handle targets are not nested
///   within Auto repetition modes or right-aligned sections.
///
/// * Checks in program order that each handle is acquired on a single signal,
///   before any match on it.
///
/// * Rejects phase resets that take time in strict timing sections, and loop preambles
///   that take time or step a PPC in strict timing loops.
pub(crate) fn validate_ir(node: &ScheduledNode) -> Result<()> {
    let ctx = ValidationContext::new();
    validate_ir_impl(node, &ctx, &mut HashMap::new())?;
    Ok(())
}

/// The signal of each handle acquired so far in program order.
type AcquiredHandles = HashMap<HandleUid, SignalUid>;

struct ValidationContext {
    repetition_mode: Option<RepetitionMode>,
    parent_alignment_mode: SectionAlignment,
    parent_timing_mode: SectionTimingMode,
}

impl ValidationContext {
    fn new() -> Self {
        Self {
            repetition_mode: None,
            parent_alignment_mode: SectionAlignment::Left,
            parent_timing_mode: SectionTimingMode::Relaxed,
        }
    }

    fn child_scope(&self, parent: &ScheduleInfo) -> Self {
        Self {
            repetition_mode: parent.repetition_mode.or(self.repetition_mode),
            parent_alignment_mode: parent.alignment_mode,
            parent_timing_mode: parent.section_timing_mode,
        }
    }
}

fn validate_node(
    node: &ScheduledNode,
    ctx: &ValidationContext,
    handles: &mut AcquiredHandles,
) -> Result<()> {
    match &node.kind {
        IrKind::Acquire(obj) => {
            if let Some(signal) = handles.insert(obj.handle, obj.signal)
                && signal != obj.signal
            {
                return Err(Error::new(format!(
                    "Acquisition handle '{}' is associated with multiple signals, only one allowed.",
                    obj.handle.0
                )));
            }
        }
        IrKind::Match(
            obj @ Match {
                target: MatchTarget::Handle(handle),
                ..
            },
        ) => {
            if !handles.contains_key(handle) {
                return Err(Error::new(format!(
                    "Handle '{}' is used in a match before being acquired.",
                    handle.0
                )));
            }
            if ctx.repetition_mode == Some(RepetitionMode::Auto) {
                let msg = format!(
                    "Match statement '{}' with handle cannot be inside an Auto repetition mode.",
                    obj.uid.0
                );
                return Err(Error::new(&msg));
            }
            if ctx.parent_alignment_mode == SectionAlignment::Right {
                let msg = format!(
                    "Match statement '{}' with handle cannot be a subsection of a right-aligned section.",
                    obj.uid.0
                );
                return Err(Error::new(&msg));
            }
        }
        IrKind::ClearPrecompensation { .. }
            if ctx.parent_alignment_mode == SectionAlignment::Right =>
        {
            return Err(Error::new(
                "Cannot reset the precompensation filter inside a right-aligned section.",
            ));
        }
        IrKind::ResetOscillatorPhase { .. }
            if ctx.parent_timing_mode == SectionTimingMode::Strict && takes_time(node) =>
        {
            return Err(Error::new(
                "HW oscillator phase reset is not allowed in strict timing sections.",
            ));
        }
        IrKind::LoopIterationPreamble if ctx.parent_timing_mode == SectionTimingMode::Strict => {
            validate_strict_preamble(node)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_strict_preamble(preamble: &ScheduledNode) -> Result<()> {
    for child in &preamble.children {
        match &child.node.kind {
            IrKind::SetOscillatorFrequency(_) if takes_time(&child.node) => {
                return Err(Error::new(
                    "HW oscillator frequency sweep is not allowed in strict timing sections.",
                ));
            }
            IrKind::ResetOscillatorPhase { .. } if takes_time(&child.node) => {
                return Err(Error::new(
                    "HW oscillator phase reset is not allowed in strict timing sections.",
                ));
            }
            IrKind::PpcStep(_) => {
                return Err(Error::new(
                    "SHFPPC sweeps are not allowed in strict timing sections.",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn takes_time(node: &ScheduledNode) -> bool {
    node.schedule.try_length() > Some(tiny_samples(0))
}

fn validate_match_children(
    node: &ScheduledNode,
    match_obj: &Match,
    ctx: &ValidationContext,
    handles: &mut AcquiredHandles,
) -> Result<()> {
    let disallow_acquisitions = !matches!(match_obj.target, MatchTarget::SweepParameter(_));

    for child in &node.children {
        validate_ir_impl(&child.node, &ctx.child_scope(&node.schedule), handles)?;
        if disallow_acquisitions && subtree_has_acquisitions(&child.node) {
            return Err(Error::new(format!(
                "Acquisitions are not allowed within match statements when matching against {}",
                match_obj.target.description()
            )));
        }
    }
    Ok(())
}

fn subtree_has_acquisitions(node: &ScheduledNode) -> bool {
    match &node.kind {
        IrKind::Acquire(_) => true,
        _ => node
            .children
            .iter()
            .any(|child| subtree_has_acquisitions(&child.node)),
    }
}

fn validate_ir_impl(
    node: &ScheduledNode,
    ctx: &ValidationContext,
    handles: &mut AcquiredHandles,
) -> Result<()> {
    validate_node(node, ctx, handles)?;
    if node.children.is_empty() {
        return Ok(());
    }
    let mut seen_sections = HashSet::new();

    match &node.kind {
        IrKind::Match(obj) => {
            validate_match_children(node, obj, ctx, handles)?;
        }
        _ => {
            let child_ctx = ctx.child_scope(&node.schedule);
            for child in node.children.iter() {
                // Validate child recursively
                validate_ir_impl(&child.node, &child_ctx, handles)?;

                // Validate play_after constraints
                if let Some(section_info) = child.node.kind.section_info() {
                    for play_after in child.node.schedule.play_after.iter() {
                        if !seen_sections.contains(play_after) {
                            let msg = format!(
                                "Section '{}' should play after section '{}' that is not defined before it on the same level.",
                                section_info.uid.0, play_after.0
                            );
                            return Err(Error::new(&msg));
                        }
                    }
                    seen_sections.insert(section_info.uid);
                }
            }
        }
    }
    Ok(())
}
