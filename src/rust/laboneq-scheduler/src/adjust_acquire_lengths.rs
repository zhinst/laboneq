// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::types::AwgKey;
use laboneq_units::tinysample::TinySamples;

use crate::experiment::types::{AcquisitionType, SignalUid};
use crate::ir::IrKind;
use crate::{ScheduleInfo, ScheduledNode, SignalInfo};

/// Adjust the length of nodes containing '[IrKind::Acquire]' in the given scheduled node so
/// that acquisitions on the same AWG occupy the same length.
pub(crate) fn adjust_acquisition_lengths(
    node: &mut ScheduledNode,
    signals: &HashMap<SignalUid, impl SignalInfo>,
    averaging_mode: AcquisitionType,
) {
    if averaging_mode == AcquisitionType::Raw {
        return;
    }
    // Fast path: check if there are any signals that require adjustment
    let signals_to_adjust = collect_signals_for_adjustment(signals);
    if signals_to_adjust.is_empty() {
        return;
    }
    let mut acquires: HashMap<SignalUid, Vec<&mut ScheduleInfo>> = HashMap::new();
    collect_acquisition_schedules(node, &signals_to_adjust, &mut acquires);
    let acquire_per_awg = group_acquisitions_by_awg(acquires, signals);
    // For each AWG, find the maximum acquisition length and set all acquisition nodes to that length
    let max_per_awg = calculate_max_lengths_per_awg(&acquire_per_awg);
    apply_max_lengths_to_acquisitions(acquire_per_awg, max_per_awg);
}

fn collect_signals_for_adjustment(signals: &HashMap<SignalUid, impl SignalInfo>) -> Vec<SignalUid> {
    signals
        .values()
        .filter(|signal| !signal.supports_multiple_acquisition_lengths())
        .map(|s| s.uid())
        .collect()
}

fn collect_acquisition_schedules<'a>(
    node: &'a mut ScheduledNode,
    target_signals: &Vec<SignalUid>,
    acquires: &mut HashMap<SignalUid, Vec<&'a mut ScheduleInfo>>,
) {
    if let IrKind::Acquire(obj) = &node.kind {
        if !target_signals.contains(&obj.signal) {
            return;
        }
        acquires
            .entry(obj.signal)
            .or_default()
            .push(&mut node.schedule);
    } else {
        for child in node.children.iter_mut() {
            collect_acquisition_schedules(child.node.make_mut(), target_signals, acquires);
        }
    }
}

fn group_acquisitions_by_awg<'a>(
    acquires: HashMap<SignalUid, Vec<&'a mut ScheduleInfo>>,
    signals: &HashMap<SignalUid, impl SignalInfo>,
) -> HashMap<AwgKey, Vec<&'a mut ScheduleInfo>> {
    let mut acquire_per_awg: HashMap<AwgKey, Vec<&'a mut ScheduleInfo>> = HashMap::new();

    for (signal_uid, acquires) in acquires {
        let signal_info = signals
            .get(&signal_uid)
            .expect("Signal should exist in signals map");
        acquire_per_awg
            .entry(signal_info.awg_key())
            .or_default()
            .extend(acquires);
    }

    acquire_per_awg
}

fn calculate_max_lengths_per_awg(
    acquire_per_awg: &HashMap<AwgKey, Vec<&mut ScheduleInfo>>,
) -> HashMap<AwgKey, TinySamples> {
    acquire_per_awg.iter().fold(HashMap::new(), {
        |mut acc, (awg_key, acquires)| {
            let max_length = acquires
                .iter()
                .map(|acquire| acquire.length)
                .max()
                .expect("There should be at least one acquisition");
            acc.insert(*awg_key, max_length);
            acc
        }
    })
}

fn apply_max_lengths_to_acquisitions(
    mut acquires_by_awg: HashMap<AwgKey, Vec<&mut ScheduleInfo>>,
    max_lengths_by_awg: HashMap<AwgKey, TinySamples>,
) {
    for (awg_key, acquires) in acquires_by_awg.iter_mut() {
        let max_length = max_lengths_by_awg
            .get(awg_key)
            .expect("Max length should exist for every AWG");

        for acquire in acquires {
            acquire.length = *max_length;
        }
    }
}
