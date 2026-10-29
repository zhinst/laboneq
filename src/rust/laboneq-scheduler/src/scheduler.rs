// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Context;
use laboneq_common::named_id::resolve_ids;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::AcquisitionType;
use laboneq_dsl::types::ParameterUid;
use laboneq_dsl::types::SweepParameter;
use laboneq_ir::ParameterLookup;
use laboneq_ir::node::IrNode;
use laboneq_log::warn;
use tracing::instrument;

use crate::ChunkingInfo;
use crate::ExperimentContext;
use crate::FeedbackCalculator;
use crate::adjust_acquire_lengths::adjust_acquisition_lengths;
use crate::analysis::validate_ir;
use crate::chunk_experiment::slice_chunked_parameters;
use crate::error::Result;
use crate::ir_unroll::unroll_loops;
use crate::lower_experiment::lower_to_ir;
use crate::resolve_parameters::resolve_parameters;
use crate::resolve_repetition_mode::resolve_repetition_mode;
use crate::scheduled_to_ir::scheduled_node_to_ir_node;
use crate::signal_info::SignalInfo;
use crate::timing_resolver::calculate_timing;
use crate::utils::check_tinysample_commensurability;

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledExperiment {
    pub root: IrNode,
    /// Parameters used in the scheduled experiment
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub acquisition_type: AcquisitionType,
}

/// Schedules the real-time part of an experiment.
///
/// `real_time_root` is the [`Operation::RealTimeBoundary`] node of the experiment.
#[instrument(level = "debug", name = "laboneq.compiler.schedule", skip_all)]
pub fn schedule_experiment<T: SignalInfo>(
    real_time_root: &ExperimentNode,
    mut context: ExperimentContext<T>,
    near_time_parameters: &dyn ParameterLookup,
    chunking_info: Option<ChunkingInfo>,
    feedback_calculator: Option<&impl FeedbackCalculator>,
) -> Result<ScheduledExperiment> {
    context.signals().try_for_each(|s| {
        check_tinysample_commensurability(s.sampling_rate())
            .with_context(|| format!("Incompatible sampling rate on signal: '{}'", s.uid().0))
    })?;

    // TODO: Preferably move chunking after scheduling after Rust migration.
    // Currently not possible as the scheduling in called per chunk.
    if let Some(chunking_info) = &chunking_info {
        slice_chunked_parameters(real_time_root, &mut context.parameters, chunking_info)?;
    }
    // TODO: Where in the IR tree `acquisition_type` should be stored?
    let acquisition_type =
        find_acquisition_type(real_time_root).expect("Unspecified acquisition type.");
    let mut scheduled_node = lower_to_ir(real_time_root, &context, near_time_parameters)?;

    resolve_repetition_mode(&mut scheduled_node)?;
    validate_ir(&scheduled_node)?;
    adjust_acquisition_lengths(&mut scheduled_node, context.signals, acquisition_type);
    let mut scheduled_node =
        unroll_loops(scheduled_node, &context.parameters, near_time_parameters)?;
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
    let ir_node = scheduled_node_to_ir_node(scheduled_node);
    let exp = ScheduledExperiment {
        root: ir_node,
        parameters: context.parameters,
        acquisition_type,
    };
    Ok(exp)
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
