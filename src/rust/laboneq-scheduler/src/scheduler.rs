// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::resolve_ids;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::AcquisitionType;
use laboneq_dsl::types::ParameterUid;
use laboneq_dsl::types::SweepParameter;
use laboneq_log::warn;

use crate::ChunkingInfo;
use crate::ExperimentContext;
use crate::FeedbackCalculator;
use crate::ScheduledNode;
use crate::adjust_acquire_lengths::adjust_acquisition_lengths;
use crate::analysis::validate_ir;
use crate::chunk_experiment::chunk_experiment;
use crate::error::Result;
use crate::ir_unroll::unroll_loops;
use crate::lower_experiment::lower_to_ir;
use crate::parameter_store::ParameterStore;
use crate::resolve_parameters::resolve_parameters;
use crate::resolve_repetition_mode::resolve_repetition_mode;
use crate::signal_info::SignalInfo;
use crate::timing_resolver::calculate_timing;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ScheduledExperiment {
    /// Parameters used in the scheduled experiment
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub root: Option<ScheduledNode>,
}

/// Schedule real time part of an Experiment
///
/// The scheduler will schedule the real time portion of the experiment
pub fn schedule_experiment<T: SignalInfo>(
    root: &ExperimentNode,
    mut context: ExperimentContext<T>,
    near_time_parameters: &ParameterStore,
    chunking_info: Option<ChunkingInfo>,
    feedback_calculator: Option<&impl FeedbackCalculator>,
) -> Result<ScheduledExperiment> {
    let real_time_root = find_real_time_root(root);
    if real_time_root.is_none() {
        return Ok(ScheduledExperiment::default());
    }
    let mut root_section = real_time_root.unwrap().clone();
    // TODO: Preferably move chunking after scheduling after Rust migration.
    // Currently not possible as the scheduling in called per chunk.
    if let Some(chunking_info) = &chunking_info {
        chunk_experiment(&mut root_section, &mut context.parameters, chunking_info)?;
    }
    // TODO: Where in the IR tree `acquisition_type` should be stored?
    let acquisition_type =
        find_acquisition_type(&root_section).expect("Unspecified acquisition type.");
    let mut scheduled_node = lower_to_ir(&root_section, &context, near_time_parameters)?;
    resolve_repetition_mode(&mut scheduled_node)?;
    validate_ir(&scheduled_node)?;
    adjust_acquisition_lengths(&mut scheduled_node, context.signals, acquisition_type);
    unroll_loops(
        &mut scheduled_node,
        &context.parameters,
        near_time_parameters,
    )?;
    resolve_parameters(
        &mut scheduled_node,
        &context.parameters,
        near_time_parameters,
    )?;
    // Calculate the timing of the scheduled experiment
    let mut timing_result = calculate_timing(&mut scheduled_node, feedback_calculator)?;
    if timing_result.has_warnings() {
        timing_result.deduplicate_warnings();
        warn!(
            "{}",
            resolve_ids(&timing_result.to_string(), context.id_store)
        );
    }

    let exp = ScheduledExperiment {
        root: Some(scheduled_node),
        parameters: context.parameters.clone(),
    };
    Ok(exp)
}

fn find_real_time_root(root: &ExperimentNode) -> Option<&ExperimentNode> {
    if root.kind == Operation::RealTimeBoundary {
        return Some(root);
    }
    for child in root.children.iter() {
        if let Some(real_time_root) = find_real_time_root(child) {
            return Some(real_time_root);
        }
    }
    None
}

fn find_acquisition_type(root: &ExperimentNode) -> Option<AcquisitionType> {
    if let Operation::AveragingLoop(obj) = &root.kind {
        return Some(obj.acquisition_type);
    }
    for child in root.children.iter() {
        if let Some(averaging_root) = find_acquisition_type(child) {
            return Some(averaging_root);
        }
    }
    None
}
