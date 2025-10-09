// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::all)]
#![allow(unused)]
use core::panic;
use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_units::duration::{Duration, Seconds, seconds};

use crate::IrNode;
use crate::error::{Error, Result};
use crate::ir::{
    Acquire, ExecutionType, IrVariant, Parameter, ParameterUid, PulseLength, PulseRef, PulseUid,
    SignalUid,
};
use crate::node::Node;
use crate::signal_info::SignalInfo;
use laboneq_common::types::AwgKey;

pub struct Experiment<'a> {
    pub sections: Vec<&'a IrNode>,
    pub id_store: &'a NamedIdStore,
    pub parameters: &'a HashMap<ParameterUid, Parameter>,
    pub pulses: &'a HashMap<PulseUid, PulseRef>,
}

#[derive(Debug)]
pub struct ScheduledExperiment {
    pub max_acquisition_time: HashMap<AwgKey, Duration<Seconds>>,
}

/// Schedule real time part of an Experiment
///
/// The scheduler will schedule the real time portion of the experiment
pub fn schedule_experiment(
    mut experiment: Experiment,
    signals: &HashMap<SignalUid, impl SignalInfo>,
) -> Result<ScheduledExperiment> {
    let mut index = ExperimentIndex {
        acquires: Vec::new(),
    };
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
        return Ok(ScheduledExperiment {
            max_acquisition_time: HashMap::new(),
        });
    }
    analyze_experiment(real_time_sections[0], &mut index);
    let max_acquisition_time = calculate_max_acquisition_time(&index, &experiment.pulses, signals)?;
    let exp = ScheduledExperiment {
        max_acquisition_time,
    };
    Ok(exp)
}

/// An index of an experiment for fast queries.
struct ExperimentIndex<'a> {
    acquires: Vec<&'a Acquire>,
}

fn analyze_experiment<'a>(node: &'a IrNode, index: &mut ExperimentIndex<'a>) {
    match node.kind() {
        IrVariant::Acquire(acquire) => {
            index.acquires.push(acquire);
        }
        _ => {
            for child in node.iter_children() {
                analyze_experiment(child.node(), index);
            }
        }
    }
}

fn find_real_time_root(root: &IrNode) -> Option<&IrNode> {
    match root.kind() {
        IrVariant::AcquireLoopRt(_) => Some(root),
        IrVariant::Sweep(obj) => {
            if obj.execution_type == ExecutionType::RealTime {
                Some(root)
            } else {
                for child in root.iter_children() {
                    if let Some(real_time_root) = find_real_time_root(child.node()) {
                        return Some(real_time_root);
                    }
                }
                None
            }
        }
        _ => {
            for child in root.iter_children() {
                if let Some(real_time_root) = find_real_time_root(child.node()) {
                    return Some(real_time_root);
                }
            }
            None
        }
    }
}

fn pulse_length_seconds(pulse: &PulseRef, sampling_rate: f64) -> Duration<Seconds> {
    match &pulse.length {
        PulseLength::Seconds(dur) => *dur,
        PulseLength::Samples(samples) => seconds(*samples as f64 / sampling_rate),
    }
}

fn calculate_max_acquisition_time(
    experiment_index: &ExperimentIndex,
    pulses: &HashMap<PulseUid, PulseRef>,
    signals: &HashMap<SignalUid, impl SignalInfo>,
) -> Result<HashMap<AwgKey, Duration<Seconds>>> {
    let mut max_acquire_time: HashMap<AwgKey, Duration<Seconds>> = HashMap::new();
    for acquire in experiment_index.acquires.iter() {
        let signal_info = signals.get(&acquire.signal).unwrap();
        let acquire_length = if let Some(length) = &acquire.length {
            *length
        } else {
            let kernels = acquire.kernel.iter().map(|p: &PulseUid| {
                pulse_length_seconds(pulses.get(p).unwrap(), signal_info.sampling_rate())
            });
            kernels.max().ok_or(Error::new(
                "Acquire has no length and no kernel pulses".to_string(),
            ))?
        };
        max_acquire_time
            .entry(signal_info.awg_key())
            .and_modify(|v| *v = (*v).max(acquire_length))
            .or_insert_with(|| acquire_length);
    }
    Ok(max_acquire_time)
}
