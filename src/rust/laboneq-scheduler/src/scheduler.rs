// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_units::duration::{Duration, Seconds};

use crate::ChunkingInfo;
use crate::ScheduledNode;
use crate::TinySample;
use crate::analysis::{RepetitionInfo, calculate_max_acquisition_time, resolve_repetition_time};
use crate::error::{Error, Result};
use crate::experiment::ExperimentNode;
use crate::experiment::types::{Operation, Parameter, ParameterUid, PulseRef, PulseUid, SignalUid};
use crate::lower_experiment::lower_to_ir;
use crate::parameter_store::ParameterStore;
use crate::signal_info::SignalInfo;
use crate::utils::compute_grid;
use laboneq_common::types::AwgKey;

pub struct Experiment<'a> {
    pub sections: Vec<&'a ExperimentNode>,
    pub id_store: &'a NamedIdStore,
    pub parameters: &'a HashMap<ParameterUid, Parameter>,
    pub pulses: &'a HashMap<PulseUid, PulseRef>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ScheduledExperiment {
    pub max_acquisition_time: HashMap<AwgKey, Duration<Seconds>>,
    pub repetition_info: Option<RepetitionInfo>,
    pub system_grid: TinySample,
    pub root: Option<ScheduledNode>,
}

/// Schedule real time part of an Experiment
///
/// The scheduler will schedule the real time portion of the experiment
pub fn schedule_experiment<T: SignalInfo + Sized>(
    experiment: Experiment,
    signals: &HashMap<SignalUid, T>,
    near_time_parameters: &ParameterStore,
    _chunking_info: Option<ChunkingInfo>, // TODO: Not yet used
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
    let scheduled_node = lower_to_ir(root_section, signals, near_time_parameters, system_grid)?;
    let exp = ScheduledExperiment {
        max_acquisition_time,
        repetition_info,
        system_grid,
        root: Some(scheduled_node),
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
