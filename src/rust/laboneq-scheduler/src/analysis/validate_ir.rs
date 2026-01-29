// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use laboneq_dsl::types::SectionAlignment;

use crate::error::{Error, Result};
use crate::ir::{IrKind, Match, MatchTarget};
use crate::{RepetitionMode, ScheduledNode};

/// Validate the IR for correctness.
///
/// * Checks that no acquisitions are present within match statements
///   when matching against targets other than sweep parameters.
///
/// * Ensures that match statements with handle targets are not nested
///   within Auto repetition modes or right-aligned sections.
pub(crate) fn validate_ir(node: &ScheduledNode) -> Result<()> {
    let ctx = ValidationContext::new();
    validate_ir_impl(node, &ctx)?;
    Ok(())
}

struct ValidationContext {
    repetition_mode: Option<RepetitionMode>,
    parent_alignment_mode: SectionAlignment,
}

impl ValidationContext {
    fn new() -> Self {
        Self {
            repetition_mode: None,
            parent_alignment_mode: SectionAlignment::Left,
        }
    }

    fn child_scope(
        &self,
        alignment_mode: SectionAlignment,
        repetition_mode: Option<RepetitionMode>,
    ) -> Self {
        Self {
            repetition_mode: repetition_mode.or(self.repetition_mode),
            parent_alignment_mode: alignment_mode,
        }
    }
}

fn validate_node(node: &ScheduledNode, ctx: &ValidationContext) -> Result<()> {
    match &node.kind {
        IrKind::Match(obj) if matches!(obj.target, MatchTarget::Handle(_)) => {
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
        _ => {}
    }
    Ok(())
}

fn validate_match_children(
    node: &ScheduledNode,
    match_obj: &Match,
    ctx: &ValidationContext,
) -> Result<()> {
    let disallow_acquisitions = !matches!(match_obj.target, MatchTarget::SweepParameter(_));

    for child in &node.children {
        validate_ir_impl(
            &child.node,
            &ctx.child_scope(node.schedule.alignment_mode, node.schedule.repetition_mode),
        )?;
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

fn validate_ir_impl(node: &ScheduledNode, ctx: &ValidationContext) -> Result<()> {
    validate_node(node, ctx)?;
    if node.children.is_empty() {
        return Ok(());
    }
    let mut seen_sections = HashSet::new();

    match &node.kind {
        IrKind::Match(obj) => {
            validate_match_children(node, obj, ctx)?;
        }
        _ => {
            let child_ctx =
                ctx.child_scope(node.schedule.alignment_mode, node.schedule.repetition_mode);
            for child in node.children.iter() {
                // Validate child recursively
                validate_ir_impl(&child.node, &child_ctx)?;

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
