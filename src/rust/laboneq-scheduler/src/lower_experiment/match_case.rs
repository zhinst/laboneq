// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use core::panic;

use crate::error::{Error, Result};

use crate::experiment_context::ExperimentContext;
use crate::ir::{Case, IrKind, Match};
use crate::lower_experiment::local_context::LocalContext;
use crate::lower_experiment::{adjust_node_grids, lower_children};
use crate::schedule_info::ScheduleInfoBuilder;

use crate::{ScheduledNode, SignalInfo};
use laboneq_dsl::operation::{Case as CaseDsl, Match as MatchDsl, Operation};
use laboneq_dsl::types::{MatchTarget, NumericLiteral, SectionAlignment, SweepParameter};
use laboneq_dsl::{ExperimentNode, NodeChild};
use laboneq_units::tinysample::{seconds_to_tinysamples, tiny_samples};

/// Grid size for local feedback
const LOCAL_FEEDBACK_GRID_SAMPLES: u8 = 8;

/// Grid size for global feedback
const GLOBAL_FEEDBACK_GRID_SAMPLES: u8 = 200;

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
    let match_ = Match {
        uid: section.uid,
        target: section.target.clone(),
        local: section.local.unwrap_or(false),
    };

    let mut schedule_builder = ScheduleInfoBuilder::new().play_after(section.play_after.clone());
    if let MatchTarget::Handle(handle) = &match_.target {
        // TODO (PW) is this correct? should it not be 100 ns regardless of the sampling rate?
        let signal = ctx.get_signal(ctx.handle_to_signal.get(handle).unwrap())?;
        let grid = if match_.local {
            LOCAL_FEEDBACK_GRID_SAMPLES
        } else {
            GLOBAL_FEEDBACK_GRID_SAMPLES
        };
        schedule_builder = schedule_builder.compressed_loop_grid(seconds_to_tinysamples(
            (grid as f64 / signal.sampling_rate()).into(),
        ));
    }

    let mut root = ScheduledNode::new(IrKind::Match(match_), schedule_builder.build());
    // Matching a sweep parameter requires special handling as each case
    // maps to a specific iteration of a loop.
    if let MatchTarget::SweepParameter(param_uid) = &section.target {
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
            let (children, reserved_signals) = local_ctx.with_section(case.uid, |local| {
                lower_children(&target_case.children, ctx, local)
            })?;
            case_node
                .schedule
                .signals
                .extend(reserved_signals.iter().cloned());
            for child in children {
                case_node.add_child(tiny_samples(0), child);
            }
            let (grid, sequencer_grid) =
                local_ctx.calculate_grids(case_node.schedule.signals.iter().cloned(), false, false);
            case_node.schedule.grid = grid;
            case_node.schedule.sequencer_grid = sequencer_grid;
            adjust_node_grids(&mut case_node);
            root.add_child(tiny_samples(0), case_node);
        }
        let (grid, sequencer_grid) =
            local_ctx.calculate_grids(root.schedule.signals.iter().cloned(), false, false);
        root.schedule.grid = grid;
        root.schedule.sequencer_grid = sequencer_grid;
    } else {
        root.schedule.grid = local_ctx.system_grid;
        root.schedule.sequencer_grid = local_ctx.system_grid;
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
            let mut case_node = ScheduledNode::new(
                kind,
                ScheduleInfoBuilder::new()
                    .grid(local_ctx.system_grid)
                    .build(),
            );
            let (children, reserved_signals) = local_ctx.with_section(case.uid, |local| {
                lower_children(&child.children, ctx, local)
            })?;
            case_node
                .schedule
                .signals
                .extend(reserved_signals.iter().cloned());
            for child in children {
                case_node.add_child(tiny_samples(0), child);
            }
            let (grid, sequencer_grid) =
                local_ctx.calculate_grids(case_node.schedule.signals.iter().cloned(), false, false);
            case_node.schedule.grid = grid;
            case_node.schedule.sequencer_grid = sequencer_grid;
            adjust_node_grids(&mut case_node);
            root.add_child(tiny_samples(0), case_node);
        }
        process_empty_branches(&mut root);
    }
    adjust_node_grids(&mut root);
    Ok(root)
}

/// For each case branch that is empty, assign it a schedule that
/// has length equal to the parent's grid.
///
/// The branch covers all signals of the parent as well.
fn process_empty_branches(parent: &mut ScheduledNode) {
    for child in &mut parent.children {
        if child.node.children.is_empty() {
            let schedule = ScheduleInfoBuilder::new()
                .grid(parent.schedule.grid)
                .length(0)
                .sequencer_grid(parent.schedule.grid)
                .alignment_mode(SectionAlignment::Left)
                .signals(parent.schedule.signals.clone())
                .build();
            child.node.make_mut().schedule = schedule;
        }
    }
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

fn cast_case(kind: &Operation) -> Option<&CaseDsl> {
    if let Operation::Case(case) = kind {
        Some(case)
    } else {
        None
    }
}

fn try_cast_match(kind: &Operation) -> &MatchDsl {
    if let Operation::Match(match_) = kind {
        match_
    } else {
        panic!("Expected a match operation.");
    }
}
