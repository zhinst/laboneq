// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_units::duration::{Duration, Second};
use laboneq_units::tinysample::TinySamples;

use crate::ChunkingInfo;
use crate::ExperimentContext;
use crate::ScheduledNode;
use crate::analysis::{
    RepetitionInfo, calculate_max_acquisition_time, resolve_repetition_time, validate_ir,
};
use crate::chunk_experiment::chunk_experiment;
use crate::error::{Error, Result};
use crate::experiment::ExperimentNode;
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{Operation, ParameterUid};
use crate::ir_unroll::unroll_loops;
use crate::lower_experiment::lower_to_ir;
use crate::parameter_store::ParameterStore;
use crate::resolve_parameters::resolve_parameters;
use crate::signal_info::SignalInfo;
use crate::utils::compute_grid;
use laboneq_common::types::AwgKey;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ScheduledExperiment {
    pub max_acquisition_time: HashMap<AwgKey, Duration<Second>>,
    pub repetition_info: Option<RepetitionInfo>,
    pub system_grid: TinySamples,
    /// Parameters used in the scheduled experiment
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub root: Option<ScheduledNode>,
}

/// Schedule real time part of an Experiment
///
/// The scheduler will schedule the real time portion of the experiment
pub fn schedule_experiment<T: SignalInfo>(
    sections: &[ExperimentNode],
    mut context: ExperimentContext<T>,
    near_time_parameters: &ParameterStore,
    chunking_info: Option<ChunkingInfo>,
) -> Result<ScheduledExperiment> {
    let real_time_sections: Vec<_> = sections
        .iter()
        .filter_map(|root| find_real_time_root(root))
        .collect();
    if real_time_sections.len() > 1 {
        return Err(Error::new(
            "Multiple real time parts found in the experiment. Only one is allowed.".to_string(),
        ));
    }
    if real_time_sections.is_empty() {
        return Ok(ScheduledExperiment::default());
    }
    let mut root_section = real_time_sections[0].clone();
    let system_grid = compute_grid(&context.signals.values().collect::<Vec<_>>()).1;
    let max_acquisition_time =
        calculate_max_acquisition_time(&root_section, context.pulses, context.signals)?;
    let repetition_info = resolve_repetition_time(&root_section)?;
    // TODO: Preferably move chunking after scheduling after Rust migration.
    // Currently not possible as the scheduling in called per chunk.
    if let Some(chunking_info) = &chunking_info {
        chunk_experiment(&mut root_section, &mut context.parameters, chunking_info)?;
    }
    let mut scheduled_node =
        lower_to_ir(&root_section, &context, near_time_parameters, system_grid)?;
    validate_ir(&scheduled_node)?;
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
    let exp = ScheduledExperiment {
        max_acquisition_time,
        repetition_info,
        system_grid,
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
