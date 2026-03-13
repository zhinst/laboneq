// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_common::device_traits;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::{AuxiliaryDeviceKind, DeviceKind, ReferenceClock};
use laboneq_dsl::types::{DeviceUid, SignalUid};
use laboneq_ir::awg::AwgCore;
use laboneq_ir::signal::{Signal, SignalKind};
use laboneq_ir::system::AwgDevice;
use laboneq_log::debug;
use laboneq_units::duration::{Duration, Frequency, Hertz, Second};

use crate::SetupProperties;
use crate::error::{Error, Result};
use crate::setup_processor::DelayRegistry;
use crate::setup_processor::delays::{SignalDelayProperties, compute_signal_delays};
use crate::setup_processor::precompensation::{AssignedPrecompensation, adapt_precompensations};
use crate::signal_properties::SignalProperties;

pub(crate) struct ProcessedSetup {
    pub signals: Vec<Signal>,
    pub devices: Vec<AwgDevice>,
    pub awgs: Vec<AwgCore>,
    pub on_device_delays: DelayRegistry,
}

pub(crate) fn process_setup(
    mut setup: SetupProperties,
    desktop_setup: bool,
    id_store: &NamedIdStore,
) -> Result<ProcessedSetup> {
    analyze_setup(&setup)?;

    let device_map = setup
        .awg_devices
        .iter()
        .map(|d| (d.uid(), d))
        .collect::<HashMap<_, _>>();

    let sampling_rates = eval_sampling_rates(&setup.signals, &device_map);

    // Adapt precompensation before computing the delays, as the presence of precompensation can affect the delay calculations.
    adapt_precompensation_on_signals(&mut setup.signals, &device_map)?;

    // Compute the on-device delays based on the signal properties and device information.
    let delays = compute_delays(&setup.signals, &device_map, &sampling_rates, desktop_setup)
        .map_err(Error::new)?;

    // Process the signals, generating the final signal configurations with the computed delays and adapted precompensation settings.
    let signals = process_signals(
        setup.signals,
        &sampling_rates,
        &device_map,
        &delays,
        id_store,
    )?;

    Ok(ProcessedSetup {
        signals,
        devices: setup.awg_devices,
        awgs: setup.awgs,
        on_device_delays: delays,
    })
}

/// Evaluates the sampling rates for each signal based on the device information and signal properties.
fn eval_sampling_rates(
    signals: &[SignalProperties],
    device_map: &HashMap<DeviceUid, &AwgDevice>,
) -> HashMap<SignalUid, Frequency<Hertz>> {
    let has_shf = device_map
        .values()
        .any(|d| matches!(d.kind(), DeviceKind::Shfqa | DeviceKind::Shfsg));

    signals
        .iter()
        .map(|signal| {
            let device = device_map.get(&signal.device_uid).unwrap();
            let sampling_rate = match device.kind() {
                DeviceKind::Shfsg => device_traits::SHFSG_SAMPLING_RATE,
                DeviceKind::Shfqa => device_traits::SHFQA_SAMPLING_RATE,
                DeviceKind::Hdawg => {
                    if has_shf {
                        device_traits::HDAWG_SAMPLING_RATE_WITH_SHF
                    } else {
                        device_traits::HDAWG_SAMPLING_RATE_WITHOUT_SHF
                    }
                }
                DeviceKind::Uhfqa => device_traits::UHFQA_SAMPLING_RATE,
                DeviceKind::Zqcs => {
                    if signal.kind == SignalKind::Integration {
                        device_traits::ZQCS_INPUT_SAMPLING_RATE
                    } else {
                        device_traits::ZQCS_OUTPUT_SAMPLING_RATE
                    }
                }
            };
            (signal.uid, sampling_rate)
        })
        .collect()
}

fn analyze_setup(setup: &SetupProperties) -> Result<()> {
    let all_devices = setup
        .awg_devices
        .iter()
        .map(|d| d.kind())
        .collect::<HashSet<_>>();

    let has_sync_devices = setup.auxiliary_devices.iter().any(|i| {
        matches!(
            i.kind(),
            AuxiliaryDeviceKind::Pqsc | AuxiliaryDeviceKind::Qhub
        )
    });

    if !has_sync_devices
        && all_devices.contains(&DeviceKind::Hdawg)
        && all_devices.contains(&DeviceKind::Uhfqa)
    {
        // Check that no internal reference clock is used for UHFQA+HDAWG.
        // TODO: Shall we move this to the Controller? This is the only place where
        // the reference clock is accessed.
        for device in &setup.awg_devices {
            if device.kind() == DeviceKind::Hdawg
                && let Some(ReferenceClock::Internal) = device.reference_clock()
            {
                return Err(Error::new(
                    "HDAWG+UHFQA system can only be used with an external clock connected to HDAWG in order to prevent jitter.",
                ));
            }
        }
    }
    Ok(())
}

/// Generates the signals from the signal properties and device information.
fn process_signals(
    signal_properties: Vec<SignalProperties>,
    sampling_rates: &HashMap<SignalUid, Frequency<Hertz>>,
    devices: &HashMap<DeviceUid, &AwgDevice>,
    delays: &DelayRegistry,
    id_store: &NamedIdStore,
) -> Result<Vec<Signal>> {
    let signals = signal_properties
        .into_iter()
        .map(|prop| -> Result<Signal> {
            let device = devices.get(&prop.device_uid).unwrap();
            let sampling_rate = sampling_rates.get(&prop.uid).unwrap().value();

            let signal_delay = round_signal_delay(
                prop.signal_delay,
                sampling_rate,
                device.traits().sample_multiple,
                prop.uid,
                id_store,
            )?;

            let signal = Signal {
                uid: prop.uid,
                channels: prop.channels,
                automute: prop.automute,
                port_mode: prop.port_mode,
                port_delay: prop.port_delay,
                start_delay: delays.signal_start_delay(prop.uid),
                range: prop.range,
                precompensation: prop.precompensation,
                signal_delay,
                awg_key: prop.awg_key,
                device_uid: prop.device_uid,
                sampling_rate,
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
    Ok(signals)
}

/// Adapt precompensation on HDAWG signals if needed, and check for unsupported precompensation usage on other devices.
///
/// The precompensation settings must be adapted for HDAWG devices when multiple signals are present on the same AWG.
/// This ensures all the signals on that AWG have the delay introduced by the precompensation correctly applied,
/// even if only some of them have precompensation explicitly set.
fn adapt_precompensation_on_signals(
    signals: &mut [SignalProperties],
    devices: &HashMap<DeviceUid, &AwgDevice>,
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
            if s.precompensation.is_some() && device.kind() == DeviceKind::Hdawg {
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

fn round_signal_delay(
    delay: Duration<Second>,
    sampling_rate: f64,
    sample_multiple: u16,
    signal_id: SignalUid,
    id_store: &NamedIdStore,
) -> Result<Duration<Second>> {
    if delay.value() < 0.0 {
        return Err(Error::new(format!(
            "Negative signal delay specified for '{}'.",
            id_store.resolve(signal_id).unwrap()
        )));
    }
    let rounded_delay = round_delay(delay, sampling_rate, sample_multiple);
    let delay_samples = delay.value() * sampling_rate;
    let rounded_delay_samples = rounded_delay.value() * sampling_rate;

    // Log only if rounding changed the delay by more than 1 sample.
    // The epsilon avoids a false-positive log when the computed difference is
    // very close to 1.0 due to floating-point arithmetic.
    if (delay_samples - rounded_delay_samples).abs() > 1.0 + 1e-12 {
        debug!(
            "Signal delay {:.2} ns of '{}' will be rounded to {:.2} ns, a multiple of {} samples.",
            delay.value() * 1e9,
            id_store.resolve(signal_id).unwrap(),
            rounded_delay.value() * 1e9,
            sample_multiple,
        );
    }
    Ok(rounded_delay)
}

/// Rounds the given delay to the nearest multiple of the sample period, with ties rounded towards zero.
fn round_delay(
    delay: Duration<Second>,
    sampling_rate: f64,
    sample_multiple: u16,
) -> Duration<Second> {
    let samples = delay * sampling_rate;
    let sample_multiple = sample_multiple as f64;
    let delay_samples = ((samples.value() / sample_multiple + 0.5).ceil() - 1.0) * sample_multiple;
    (delay_samples / sampling_rate).into()
}

/// Computes the on-device delays for a set of signals based on their properties and device information.
///
/// Once the on-device delays are computed, the start and port delays for each signal are calculated accordingly
/// to compensate for the on-device delays, ensuring that the signals are correctly aligned in time when executed on the device.
///
/// The alignment takes into account following factors:
/// - Lead delay: The inherent delay of the device before the signal starts being output.
/// - Output routing delay: Delay introduced by output routing.
/// - Precompensation delay: Delay introduced by precompensation settings on the signal.
///
/// Returns a struct containing the start and port delays for each signal, which can be used to adjust the signal timing in the setup processing.
fn compute_delays(
    signals: &[SignalProperties],
    devices: &HashMap<DeviceUid, &AwgDevice>,
    sampling_rates: &HashMap<SignalUid, Frequency<Hertz>>,
    desktop_setup: bool,
) -> Result<DelayRegistry> {
    let signal_delay_props = signals
        .iter()
        .map(|s| -> Result<SignalDelayProperties> {
            let device = devices.get(&s.device_uid).unwrap();
            SignalDelayProperties::new(
                s.uid,
                s.channels.clone(),
                device.physical_device_uid(),
                sampling_rates.get(&s.uid).unwrap().value(),
                device.kind(),
                s.added_outputs.iter().collect(),
                s.precompensation.as_ref(),
            )
            .map_err(Error::new)
        })
        .collect::<Result<Vec<_>>>()?;

    let delays = compute_signal_delays(&signal_delay_props, desktop_setup);
    Ok(delays)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_delay() {
        let sampling_rate = 2.0e9;
        let sample_multiple = 4;

        for (delay_samples, delay_rounded_samples) in [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.5, 0.0),
            (2.0, 0.0),
            (2.5, 4.0),
            (3.0, 4.0),
            (3.5, 4.0),
            (4.0, 4.0),
            (4.5, 4.0),
            (5.0, 4.0),
            (7.0, 8.0),
            (8.0, 8.0),
        ] {
            let delay_seconds = delay_samples / sampling_rate;
            let rounded_seconds = round_delay(delay_seconds.into(), sampling_rate, sample_multiple);
            let rounded_samples = rounded_seconds.value() * sampling_rate;
            assert_eq!(
                rounded_samples, delay_rounded_samples,
                "Delay {} should round to {}, but got {}.",
                delay_samples, delay_rounded_samples, rounded_samples
            );
        }
    }
}
