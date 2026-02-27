// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use crate::ir::compilation_job::{AwgCore, DeviceKind, SignalKind};

pub(crate) fn process_awgs(awgs: &mut [AwgCore]) {
    // Sort the signals for consistent ordering.
    order_awgs(awgs);
    allocate_shfqa_generator_channels(awgs);
}

/// Sort AWGs and their signals for consistent ordering.
fn order_awgs(awgs: &mut [AwgCore]) {
    for awg in awgs.iter_mut() {
        awg.signals.sort_by_key(|s| s.uid);
        awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
    }
    // Sort the AWGs for consistent ordering.
    awgs.sort_by(|a, b| {
        let a_signals = a
            .signals
            .iter()
            .flat_map(|s| &s.channels)
            .collect::<Vec<_>>();
        let b_signals = b
            .signals
            .iter()
            .flat_map(|s| &s.channels)
            .collect::<Vec<_>>();
        a_signals.cmp(&b_signals)
    });
    awgs.sort_by_key(|a| a.key());
}

/// Allocate generator channels for SHFQA devices.
///
/// Each signal needs to correspond to a unique generator channel (waveform slot) on the device.
fn allocate_shfqa_generator_channels(awgs: &mut [AwgCore]) {
    let mut shfqa_generator_allocation = HashMap::new();

    for awg in awgs.iter_mut() {
        if awg.device_kind() != &DeviceKind::SHFQA {
            continue;
        }
        let awg_key = awg.key();
        for signal in awg.signals.iter_mut() {
            if signal.kind != SignalKind::IQ {
                continue;
            }
            let generator_channel = *shfqa_generator_allocation
                .entry(awg_key.clone())
                .and_modify(|f| *f += 1)
                .or_insert_with(|| 0);

            let signal_mut =
                Arc::get_mut(signal).expect("Expected to get mutable reference to signal");
            signal_mut.channels = vec![generator_channel];
        }
    }
}
