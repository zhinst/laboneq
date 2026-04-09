// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};

use crate::experiment_context::ExperimentContext;
use crate::lower_experiment::local_context::LocalContext;
use crate::lower_experiment::{adjust_node_grids, lower_children};
use crate::schedule_info::ScheduleInfoBuilder;
use laboneq_ir::{Case, IrKind, Match};

use crate::{ScheduledNode, SignalInfo};
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Case as CaseDsl, Match as MatchDsl, Operation};
use laboneq_dsl::types::{MatchTarget, NumericLiteral, SweepParameter};
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

    let mut schedule_builder = ScheduleInfoBuilder::new()
        .play_after(section.play_after.clone())
        .section_timing_mode(local_ctx.section_timing_mode);
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

    let mut root = ScheduledNode::new_with_capacity(
        IrKind::Match(match_),
        schedule_builder.build(),
        node.children.len(),
    );

    if let MatchTarget::SweepParameter(param_uid) = &section.target {
        // Matching a sweep parameter requires special handling as each case
        // maps to a specific iteration of a loop.

        // 1. Sort the cases according to the order of the sweep parameter values
        // 2. Handle each case and add it to the match node.
        for case_iteration in sort_match_cases_to_parameter(
            node,
            local_ctx
                .parameter_resolver()
                .try_resolve_parameter(param_uid)?,
        )? {
            let child = &node.children[case_iteration];
            let case = cast_case(&child.kind).unwrap();
            let kind = IrKind::Case(Case {
                uid: case.uid,
                state: case_iteration,
            });
            let mut case_node = ScheduledNode::new(kind, ScheduleInfoBuilder::new().build());
            let (children, reserved_signals) =
                local_ctx.with_section(case.uid, case.section_timing_mode, |local| {
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
            let mut case_node = ScheduledNode::new_with_capacity(
                kind,
                ScheduleInfoBuilder::new()
                    .grid(local_ctx.system_grid)
                    .build(),
                child.children.len(),
            );
            let (children, reserved_signals) =
                local_ctx.with_section(case.uid, case.section_timing_mode, |local| {
                    lower_children(&child.children, ctx, local)
                })?;
            case_node.schedule.signals.extend(reserved_signals);
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
    }
    adjust_node_grids(&mut root);
    Ok(root)
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

fn sort_match_cases_to_parameter(
    node: &ExperimentNode,
    sweep_parameter: &SweepParameter,
) -> Result<Vec<usize>> {
    let mut state_to_index: HashMap<NumericLiteral, usize> =
        HashMap::with_capacity(node.children.len());
    for (idx, child) in node.children.iter().enumerate() {
        let case = cast_case(&child.kind)
            .ok_or_else(|| Error::new("Expected a case operation as child of match."))?;
        state_to_index.insert(case.state.to_float(), idx);
    }

    let mut out = Vec::with_capacity(state_to_index.len());
    for sweep_value in sweep_parameter.values() {
        let target_idx = state_to_index.get(&sweep_value.to_float()).ok_or_else(|| {
            Error::new(format!(
                "Using a match statement for sweep parameter must cover all values.
        Match statement for parameter '{}' is missing a case for value '{}'.",
                sweep_parameter.uid.0, sweep_value,
            ))
        })?;
        out.push(*target_idx);
    }
    Ok(out)
}
