// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Rules that a QCCS experiment must satisfy, in addition to the rules checked
//! by the compiler for every backend.
//!
//! The compiler has already refused a signal whose device cannot do what the
//! signal asks of it, so these rules only refine what it accepted.

mod operations;
mod ppc_sweeps;
mod signals;

use laboneq_compiler_py::compiler_backend::ProcessedExperiment;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::SectionUid;
use laboneq_error::laboneq_error;

use crate::Result;
use crate::preprocessor::QccsBackendPreprocessedData;

/// Check a processed experiment against the QCCS rules.
pub(crate) fn validate(
    experiment: &ProcessedExperiment<'_>,
    data: &QccsBackendPreprocessedData,
) -> Result<()> {
    signals::validate(data, &experiment.parameters, experiment.acquisition_type)?;

    let mut operations = operations::OperationRules::new(data);
    let mut ppc_sweeps = ppc_sweeps::PpcSweepRules::new(data);
    walk(experiment.root, &mut |node, ctx| {
        ppc_sweeps.visit(node, ctx);
        operations.visit(node, ctx)
    })?;
    ppc_sweeps.report_conflicts()
}

/// Where in the experiment a rule is looking.
#[derive(Clone, Copy, Default)]
pub(super) struct NodeContext {
    /// UID of the section being visited, or of the closest enclosing one.
    section_uid: Option<SectionUid>,
    /// Whether the node sits inside the real-time boundary.
    pub(super) inside_rt_bound: bool,
}

impl NodeContext {
    /// UID of the section the node belongs to.
    ///
    /// Every operation is expected to sit in a section.
    pub(super) fn section_uid(&self) -> Result<SectionUid> {
        self.section_uid
            .ok_or_else(|| laboneq_error!("Internal error: section not found."))
    }
}

/// Walk the experiment tree, offering every node to `visit`.
fn walk(
    root: &ExperimentNode,
    visit: &mut impl FnMut(&ExperimentNode, NodeContext) -> Result<()>,
) -> Result<()> {
    visit_node(root, NodeContext::default(), visit)
}

fn visit_node(
    node: &ExperimentNode,
    ctx: NodeContext,
    visit: &mut impl FnMut(&ExperimentNode, NodeContext) -> Result<()>,
) -> Result<()> {
    let ctx = NodeContext {
        section_uid: node.kind.section_info().map(|s| *s.uid).or(ctx.section_uid),
        ..ctx
    };
    visit(node, ctx)?;

    let child_ctx = NodeContext {
        inside_rt_bound: ctx.inside_rt_bound || matches!(node.kind, Operation::RealTimeBoundary),
        ..ctx
    };
    for child in node.children.iter() {
        visit_node(child, child_ctx, visit)?;
    }
    Ok(())
}
