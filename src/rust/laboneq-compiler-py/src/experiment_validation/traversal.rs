// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! One walk of the experiment tree, for the rules of the compiler itself.

use std::collections::{HashMap, HashSet};

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{ParameterUid, PulseDef, PulseUid, SectionUid, SignalUid, SweepParameter};

use crate::error::{Error, Result};
use crate::signal_view::SignalView;

/// The experiment, and where in it a validator is looking.
#[derive(Clone, Copy)]
pub(super) struct VisitContext<'a> {
    pub(super) pulses: &'a HashMap<PulseUid, PulseDef>,
    pub(super) parameters: &'a HashMap<ParameterUid, &'a SweepParameter>,
    pub(super) signals: &'a HashMap<SignalUid, SignalView<'a>>,
    pub(super) nt_only_parameters: &'a HashSet<ParameterUid>,
    /// UID of the section being visited, or of the closest enclosing one.
    pub(super) section_uid: Option<SectionUid>,
    /// Whether the node sits inside the real-time boundary.
    pub(super) inside_rt_bound: bool,
}

impl VisitContext<'_> {
    /// UID of the section under validation.
    ///
    /// Every operation is expected to sit in a section.
    pub(super) fn section_uid(&self) -> Result<SectionUid> {
        self.section_uid
            .ok_or_else(|| Error::new("Internal error: section not found."))
    }
}

/// A set of rules to check against an experiment.
///
/// The compiler offers every node of the experiment tree to the validator, so
/// a rule that spans nodes can collect what it needs on the way and report in
/// [`Validator::finish`].
pub(super) trait Validator<'a> {
    /// Check what holds of the experiment as a whole, before the walk.
    fn start(&mut self, _ctx: &VisitContext<'a>) -> Result<()> {
        Ok(())
    }

    /// Check the node the walk enters.
    fn visit(&mut self, _node: &ExperimentNode, _ctx: &VisitContext<'a>) -> Result<()> {
        Ok(())
    }

    /// Check the node once its children have been visited.
    fn leave(&mut self, _node: &ExperimentNode, _ctx: &VisitContext<'a>) -> Result<()> {
        Ok(())
    }

    /// Check what the walk collected, once the whole tree has been visited.
    fn finish(&mut self, _ctx: &VisitContext<'a>) -> Result<()> {
        Ok(())
    }
}

/// Walk the experiment tree, offering every node to the validator.
pub(super) fn validate_tree<'a>(
    root: &ExperimentNode,
    ctx: VisitContext<'a>,
    validator: &mut impl Validator<'a>,
) -> Result<()> {
    validator.start(&ctx)?;
    visit_node(root, ctx, validator)?;
    validator.finish(&ctx)
}

fn visit_node<'a>(
    node: &ExperimentNode,
    ctx: VisitContext<'a>,
    validator: &mut impl Validator<'a>,
) -> Result<()> {
    let ctx = VisitContext {
        section_uid: node.kind.section_info().map(|s| *s.uid).or(ctx.section_uid),
        ..ctx
    };
    validator.visit(node, &ctx)?;

    let child_ctx = VisitContext {
        inside_rt_bound: ctx.inside_rt_bound || matches!(node.kind, Operation::RealTimeBoundary),
        ..ctx
    };
    for child in node.children.iter() {
        visit_node(child, child_ctx, validator)?;
    }

    validator.leave(node, &ctx)
}
