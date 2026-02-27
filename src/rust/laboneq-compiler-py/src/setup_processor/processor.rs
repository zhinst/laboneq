// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::setup_processor::DelayRegistry;
use crate::setup_processor::delays::compute_delays;
use crate::setup_processor::precompensation::{AssignedPrecompensation, adapt_precompensations};
use crate::signal_properties::SignalProperties;
use laboneq_common::types::DeviceKind;
use laboneq_dsl::types::DeviceUid;
use laboneq_ir::signal::Signal;
use laboneq_ir::system::Device;

pub(crate) struct SignalProcessingResult {
    pub signals: Vec<Signal>,
    pub on_device_delays: DelayRegistry,
}

/// Generates the signals from the signal properties and device information.
pub(crate) fn process_signals(
    mut signal_properties: Vec<SignalProperties>,
    devices: &HashMap<DeviceUid, Device>,
    desktop_setup: bool,
) -> Result<SignalProcessingResult> {
    // Adapt precompensation before computing the delays, as the presence of precompensation can affect the delay calculations.
    adapt_precompensation_on_signals(&mut signal_properties, devices)?;
    let delays = compute_delays(
        &signal_properties.iter().collect::<Vec<_>>(),
        devices,
        desktop_setup,
    )
    .map_err(Error::new)?;

    let signals = signal_properties
        .into_iter()
        .map(|prop| -> Result<Signal> {
            let signal = Signal {
                uid: prop.uid,
                channels: prop.channels,
                automute: prop.automute,
                port_mode: prop.port_mode,
                port_delay: prop.port_delay,
                start_delay: delays.signal_start_delay(prop.uid),
                range: prop.range,
                precompensation: prop.precompensation,
                signal_delay: prop.signal_delay,
                awg_key: prop.awg_key,
                device_uid: prop.device_uid,
                sampling_rate: prop.sampling_rate,
                amplifier_pump: prop.amplifier_pump,
                oscillator: prop.oscillator,
                lo_frequency: prop.lo_frequency,
                voltage_offset: prop.voltage_offset,
                kind: prop.kind,
                added_outputs: prop.added_outputs,
            };
            Ok(signal)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SignalProcessingResult {
        signals,
        on_device_delays: delays,
    })
}

/// Adapt precompensation on HDAWG signals if needed, and check for unsupported precompensation usage on other devices.
///
/// The precompensation settings must be adapted for HDAWG devices when multiple signals are present on the same AWG.
/// This ensures all the signals on that AWG have the delay introduced by the precompensation correctly applied,
/// even if only some of them have precompensation explicitly set.
fn adapt_precompensation_on_signals(
    signals: &mut [SignalProperties],
    devices: &HashMap<DeviceUid, Device>,
) -> Result<()> {
    // Find AWGs that may need precompensation adaptation
    let should_adapt_pc: Vec<_> = signals
        .iter()
        .filter_map(|s| {
            let device = devices.get(&s.device_uid).unwrap();
            if s.precompensation.is_some() && !device.traits().supports_precompensation {
                return Some(Err(Error::new(format!(
                    "Precompensation is not supported on device '{}'.",
                    device.kind(),
                ))));
            }
            if s.precompensation.is_some() && device.kind() == &DeviceKind::Hdawg {
                Some(Ok(s.awg_key))
            } else {
                None
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // Extract precompensations for adaptation
    let mut pcs = Vec::new();
    for signal in signals.iter_mut() {
        if !should_adapt_pc.contains(&signal.awg_key) {
            continue;
        }
        if let Some(precomp) = signal.precompensation.take() {
            pcs.push(AssignedPrecompensation {
                signal_uid: signal.uid,
                awg: signal.awg_key,
                precompensation: Some(precomp),
            });
        } else {
            // Placeholder for signal without precompensation, to ensure it gets adapted if needed
            pcs.push(AssignedPrecompensation {
                signal_uid: signal.uid,
                awg: signal.awg_key,
                precompensation: None,
            });
        }
    }

    adapt_precompensations(&mut pcs)?;

    // Apply adapted precompensations back to signals
    for pc in pcs {
        if let Some(signal) = signals.iter_mut().find(|s| s.uid == pc.signal_uid) {
            signal.precompensation = pc.precompensation;
        }
    }
    Ok(())
}
