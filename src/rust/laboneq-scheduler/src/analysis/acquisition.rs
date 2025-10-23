// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::experiment::ExperimentNode;
use crate::experiment::types::{Acquire, Operation, PulseLength, PulseRef, PulseUid, SignalUid};
use crate::signal_info::SignalInfo;
use laboneq_common::types::AwgKey;
use laboneq_units::duration::{Duration, Seconds, seconds};

pub fn calculate_max_acquisition_time(
    node: &ExperimentNode,
    pulses: &HashMap<PulseUid, PulseRef>,
    signals: &HashMap<SignalUid, impl SignalInfo>,
) -> Result<HashMap<AwgKey, Duration<Seconds>>> {
    let mut acquires: Vec<&Acquire> = Vec::new();
    collect_acquires(node, &mut acquires);
    calculate_max_acquisition_time_impl(acquires, pulses, signals)
}

fn collect_acquires<'a>(node: &'a ExperimentNode, acquires: &mut Vec<&'a Acquire>) {
    if let Operation::Acquire(obj) = &node.kind {
        acquires.push(obj);
    }
    for child in node.children.iter() {
        collect_acquires(child, acquires);
    }
}

fn calculate_max_acquisition_time_impl(
    acquires: Vec<&Acquire>,
    pulses: &HashMap<PulseUid, PulseRef>,
    signals: &HashMap<SignalUid, impl SignalInfo>,
) -> Result<HashMap<AwgKey, Duration<Seconds>>> {
    let mut max_acquire_time: HashMap<AwgKey, Duration<Seconds>> = HashMap::new();
    for acquire in acquires.iter() {
        let signal_info = signals.get(&acquire.signal).unwrap();
        let acquire_length = if let Some(length) = &acquire.length {
            *length
        } else {
            let kernels = acquire.kernel.iter().map(|p: &PulseUid| {
                pulse_length_seconds(pulses.get(p).unwrap(), signal_info.sampling_rate())
            });
            kernels
                .max()
                .ok_or(Error::new("Acquire has no length and no kernel pulses"))?
        };
        max_acquire_time
            .entry(signal_info.awg_key())
            .and_modify(|v| *v = (*v).max(acquire_length))
            .or_insert_with(|| acquire_length);
    }
    Ok(max_acquire_time)
}

fn pulse_length_seconds(pulse: &PulseRef, sampling_rate: f64) -> Duration<Seconds> {
    match &pulse.length {
        PulseLength::Seconds(dur) => *dur,
        PulseLength::Samples(samples) => seconds(*samples as f64 / sampling_rate),
    }
}
