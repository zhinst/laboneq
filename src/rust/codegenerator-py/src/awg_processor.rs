// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use codegenerator::ir::compilation_job::{AwgCore, DeviceKind, SignalKind};

pub(crate) fn process_awgs(awgs: &mut [AwgCore]) {
    allocate_shfqa_generator_channels(awgs);
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
