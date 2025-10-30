// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_units::duration::{Duration, Seconds};

use crate::ChunkingInfo;
use crate::ScheduledNode;
use crate::TinySample;
use crate::analysis::{RepetitionInfo, calculate_max_acquisition_time, resolve_repetition_time};
use crate::chunk_ir::chunk_ir;
use crate::error::{Error, Result};
use crate::experiment::ExperimentNode;
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::SectionUid;
use crate::experiment::types::{Operation, ParameterUid, PulseRef, PulseUid, SignalUid};
use crate::ir_unroll::unroll_loops;
use crate::lower_experiment::lower_to_ir;
use crate::parameter_store::ParameterStore;
use crate::signal_info::SignalInfo;
use crate::utils::compute_grid;
use laboneq_common::types::AwgKey;

pub struct Experiment<'a> {
    pub sections: Vec<&'a ExperimentNode>,
    pub id_store: &'a NamedIdStore,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub pulses: &'a HashMap<PulseUid, PulseRef>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ScheduledExperiment {
    pub max_acquisition_time: HashMap<AwgKey, Duration<Seconds>>,
    pub repetition_info: Option<RepetitionInfo>,
    pub system_grid: TinySample,
    /// Parameters used in the scheduled experiment
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub root: Option<ScheduledNode>,
}

/// Schedule real time part of an Experiment
///
/// The scheduler will schedule the real time portion of the experiment
pub fn schedule_experiment<T: SignalInfo + Sized>(
    mut experiment: Experiment,
    signals: &HashMap<SignalUid, T>,
    near_time_parameters: &ParameterStore,
    chunking_info: Option<ChunkingInfo>,
) -> Result<ScheduledExperiment> {
    let real_time_sections: Vec<_> = experiment
        .sections
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
    let root_section = real_time_sections[0];
    let system_grid = compute_grid(&signals.values().collect::<Vec<_>>()).1;
    let max_acquisition_time =
        calculate_max_acquisition_time(root_section, experiment.pulses, signals)?;
    let repetition_info = resolve_repetition_time(root_section)?;
    let sweep_to_chunk = chunking_info
        .is_some()
        .then(|| find_sweep_to_chunk(root_section))
        .flatten();

    let mut scheduled_node = lower_to_ir(root_section, signals, near_time_parameters, system_grid)?;
    // Apply chunking if needed
    if let (Some(chunking_info), Some(sweep_to_chunk)) = (chunking_info, sweep_to_chunk) {
        let new_parameters = chunk_ir(
            &mut scheduled_node,
            (sweep_to_chunk, chunking_info.index, chunking_info.count),
            &experiment.parameters,
        )?;
        for param in new_parameters {
            experiment.parameters.insert(param.uid, param);
        }
    }
    unroll_loops(&mut scheduled_node)?;
    let exp = ScheduledExperiment {
        max_acquisition_time,
        repetition_info,
        system_grid,
        root: Some(scheduled_node),
        parameters: experiment.parameters.clone(),
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

fn find_sweep_to_chunk(root: &ExperimentNode) -> Option<SectionUid> {
    for child in root.children.iter() {
        if let Operation::Sweep(obj) = &child.kind
            && obj.chunking.is_some()
        {
            return Some(obj.uid);
        }
        if let Some(uid) = find_sweep_to_chunk(child) {
            return Some(uid);
        }
    }
    None
}
