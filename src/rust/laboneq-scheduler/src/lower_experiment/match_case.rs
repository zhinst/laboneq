// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::panic;

use crate::error::{Error, Result};
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{self as experiment_types, NumericLiteral, Operation};
use crate::experiment::{ExperimentNode, NodeChild};

use crate::experiment_context::ExperimentContext;
use crate::ir::{Case, IrKind, Match};
use crate::lower_experiment::local_context::LocalContext;
use crate::lower_experiment::lower_children;
use crate::schedule_info::ScheduleInfoBuilder;

use crate::{ScheduledNode, SignalInfo};
use laboneq_units::tinysample::tiny_samples;

/// Lower a match section from the experiment definition to the scheduled IR.
///
/// This involves creating a [`ScheduledNode`] of kind [`IrKind::Match`],
/// and lowering each of its child cases appropriately.
pub(super) fn lower_match(
    node: &ExperimentNode,
    ctx: &ExperimentContext<impl SignalInfo>,
    local_ctx: &mut LocalContext,
) -> Result<ScheduledNode> {
    let section = try_cast_match(&node.kind);
    let mut root = ScheduledNode::new(
        IrKind::Match(Match {
            uid: section.uid,
            target: section.target.clone(),
            // TODO: Resolve in experiment processor!
            // At this point the local flag should be resolved to concrete value.
            local: section.local.unwrap_or(false),
            play_after: section.play_after.clone(),
        }),
        ScheduleInfoBuilder::new().build(),
    );
    // Matching a sweep parameter requires special handling as each case
    // maps to a specific iteration of a loop.
    if let experiment_types::MatchTarget::SweepParameter(param_uid) = &section.target {
        let matching_cases = find_matching_cases(
            &node.children,
            local_ctx
                .parameter_resolver()
                .try_resolve_parameter(param_uid)?,
        )?;
        for (target_case, iteration) in matching_cases {
            let case = cast_case(&target_case.kind).unwrap();
            let kind = IrKind::Case(Case {
                uid: case.uid,
                state: iteration,
            });
            let mut case_node = ScheduledNode::new(kind, ScheduleInfoBuilder::new().build());
            local_ctx
                .with_section(case.uid, |local| {
                    lower_children(&target_case.children, ctx, local)
                })?
                .into_iter()
                .for_each(|child| {
                    case_node.add_child(tiny_samples(0), child);
                });
            root.add_child(tiny_samples(0), case_node);
        }
    } else {
        for child in &node.children {
            let case =
                cast_case(&child.kind).ok_or_else(|| Error::new("Expected a case operation."))?;
            let kind = IrKind::Case(Case {
                uid: case.uid,
                state: case.state.try_into().map_err(|e| {
                    Error::new(format!(
                        "Invalid case state value '{}' for case uid '{}': {}",
                        case.state, case.uid.0, e
                    ))
                })?,
            });
            let mut case_node = ScheduledNode::new(kind, ScheduleInfoBuilder::new().build());
            local_ctx
                .with_section(case.uid, |local_ctx| {
                    lower_children(&child.children, ctx, local_ctx)
                })?
                .into_iter()
                .for_each(|child| {
                    case_node.add_child(tiny_samples(0), child);
                });
            root.add_child(tiny_samples(0), case_node);
        }
    }
    Ok(root)
}

/// Find the cases that match the values of a sweep parameter.
///
/// Each value of the sweep parameter must be covered by a case.
fn find_matching_cases<'a>(
    cases: &'a [NodeChild],
    parameter: &SweepParameter,
) -> Result<Vec<(&'a ExperimentNode, usize)>> {
    // Map the iteration number of a loop to a specific case.
    if parameter.len() > cases.len() {
        let msg = format!(
            "Using a match statement for sweep parameter must cover all values.
        Match statement for parameter '{}' has {} cases, but parameter has {} values.",
            parameter.uid.0,
            cases.len(),
            parameter.len(),
        );
        return Err(Error::new(msg));
    }
    let mut matches: Vec<(&'a ExperimentNode, usize)> = Vec::with_capacity(parameter.len());
    for idx in 0..parameter.len() {
        for child_node in cases {
            let target_value: NumericLiteral = parameter
                .value_numeric_at_index(idx)
                .unwrap_or_else(|| panic!("Expected value to exist"));
            let case = cast_case(&child_node.kind)
                .ok_or_else(|| Error::new("Match must have only case operations."))?;
            if case.state == target_value {
                matches.push((child_node, idx));
                break;
            }
        }
    }
    Ok(matches)
}

fn cast_case(kind: &Operation) -> Option<&experiment_types::Case> {
    if let Operation::Case(case) = kind {
        Some(case)
    } else {
        None
    }
}

fn try_cast_match(kind: &Operation) -> &experiment_types::Match {
    if let Operation::Match(match_) = kind {
        match_
    } else {
        panic!("Expected a match operation.");
    }
}
