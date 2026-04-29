// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_common::device_traits;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::{DeviceKind, SignalKind};
use laboneq_dsl::device_setup::AuxiliaryDevice;
use laboneq_dsl::device_setup::DeviceSignal;
use laboneq_dsl::types::{AcquisitionType, DeviceUid, OscillatorKind, SignalUid};
use laboneq_ir::signal::Signal;
use laboneq_ir::system::AwgDevice;
use laboneq_log::{debug, info, warn};
use laboneq_units::duration::{Duration, Frequency, Hertz, Second};
use smallvec::SmallVec;

use crate::SetupProperties;
use crate::compiler_backend::PreprocessedBackendData;
use crate::error::{Error, Result};
use crate::experiment_context::ExperimentContext;
use crate::setup_processor::DelayRegistry;
use crate::setup_processor::delays::{SignalDelayProperties, compute_signal_delays};
use crate::setup_processor::precompensation::{AssignedPrecompensation, adapt_precompensations};

pub(crate) struct ProcessedSetup {
    pub signals: Vec<Signal>,
    pub devices: Vec<AwgDevice>,
    pub auxiliary_devices: Vec<AuxiliaryDevice>,
    pub on_device_delays: DelayRegistry,
}

/// Process the setup properties.
///
/// TODO: Most of the functionality could be moved to the Compiler backends.
pub(crate) fn process_setup(
    setup: SetupProperties,
    backend_processed: &impl PreprocessedBackendData,
    desktop_setup: bool,
    id_store: &NamedIdStore,
    context: &ExperimentContext,
) -> Result<ProcessedSetup> {
    let mut signals = setup.signals;

    let device_map = setup
        .awg_devices
        .iter()
        .map(|d| (d.uid(), d))
        .collect::<HashMap<_, _>>();

    let sampling_rates = eval_sampling_rates(&signals, &device_map);

    resolve_oscillator_modulation(
        &mut signals,
        &device_map,
        &sampling_rates,
        context,
        id_store,
        backend_processed,
    )?;

    // Adapt precompensation before computing the delays, as the presence of precompensation can affect the delay calculations.
    adapt_precompensation_on_signals(&mut signals, &device_map, backend_processed)?;

    // Compute the on-device delays based on the signal properties and device information.
    let delays = compute_delays(&signals, &device_map, &sampling_rates, desktop_setup)
        .map_err(Error::new)?;

    // Process the signals, generating the final signal configurations with the computed delays and adapted precompensation settings.
    let signals = process_signals(
        signals,
        &sampling_rates,
        &device_map,
        &delays,
        id_store,
        backend_processed,
    )?;

    Ok(ProcessedSetup {
        signals,
        devices: setup.awg_devices,
        auxiliary_devices: setup.auxiliary_devices,
        on_device_delays: delays,
    })
}

/// Evaluates the sampling rates for each signal based on the device information and signal properties.
fn eval_sampling_rates(
    signals: &[DeviceSignal],
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

/// Resolves the modulation type for oscillators with AUTO modulation based on the device capabilities.
fn resolve_oscillator_modulation(
    signals: &mut [DeviceSignal],
    device_map: &HashMap<DeviceUid, &AwgDevice>,
    sampling_rates: &HashMap<SignalUid, Frequency<Hertz>>,
    context: &ExperimentContext,
    id_store: &NamedIdStore,
    backend_processed: &impl PreprocessedBackendData,
) -> Result<()> {
    fn hw_channel_key(
        signal: &DeviceSignal,
        backend_processed: &impl PreprocessedBackendData,
    ) -> Option<(DeviceUid, SmallVec<[u16; 4]>)> {
        if let Some(channels) = backend_processed.channels(signal.uid) {
            let device_uid = signal.device_uid;
            let mut channels = channels.clone();
            channels.sort();
            return Some((device_uid, channels));
        }
        None
    }

    // Threshold above which AUTO modulation resolves to HARDWARE on QA devices
    // with LRT option. Below this, SOFTWARE is used for integration mode.
    // This corresponds to 4096 samples at the SHFQA's 2 GHz sampling rate.
    const _LRT_HW_MODULATION_THRESHOLD: f64 = 4096.0 / 2e9; // 2.048 us

    // Pre-compute which HW channels have any SHFQA signal with a long acquire.
    // This ensures that all signals sharing a HW channel (e.g. measure + acquire)
    // get consistent modulation resolution.
    let shfqa_channels_with_long_readout = signals
        .iter()
        .filter(|s| {
            let device = device_map.get(&s.device_uid).unwrap();
            matches!(device.kind(), DeviceKind::Shfqa)
        })
        .filter_map(|signal| {
            let sampling_rate = sampling_rates.get(&signal.uid).unwrap();
            let max_acq_len = max_acquisition_length_seconds(&signal.uid, context, *sampling_rate);
            let has_long_readout =
                max_acq_len.is_some_and(|len| len.value() > _LRT_HW_MODULATION_THRESHOLD);
            if has_long_readout {
                return Some(hw_channel_key(signal, backend_processed));
            }
            None
        })
        .collect::<HashSet<_>>();

    for signal in signals {
        let hw_channel_key = hw_channel_key(signal, backend_processed);
        if let Some(osc) = signal.calibration.oscillator.as_mut()
            && osc.kind == OscillatorKind::Auto
        {
            let device = device_map.get(&signal.device_uid).unwrap();
            let oscillator_kind: OscillatorKind = match device.kind() {
                DeviceKind::Shfqa => {
                    let has_lrt = device.options().is_some_and(|o| o.contains("LRT"));
                    let has_long_readout =
                        shfqa_channels_with_long_readout.contains(&hw_channel_key);

                    if !has_long_readout {
                        if context.acquisition_type().is_spectroscopy() {
                            OscillatorKind::Hardware
                        } else {
                            OscillatorKind::Software
                        }
                    } else if !has_lrt && !context.acquisition_type().is_spectroscopy() {
                        let msg = format!(
                            "Acquisition length on signal '{}' exceeds \
                        {} (4096 samples) and \
                        requires hardware modulation, but the device \
                        '{}' does not have the LRT option \
                        installed. Either reduce the acquisition length or \
                        set the oscillator modulation type explicitly.",
                            signal.uid.0,
                            _LRT_HW_MODULATION_THRESHOLD,
                            device.uid().0
                        );
                        return Err(Error::new(msg));
                    } else if has_lrt && !context.acquisition_type().is_spectroscopy() {
                        if context.acquisition_type() == &AcquisitionType::Raw {
                            warn!(
                                "Oscillator '{}' on signal \
                                 '{}' resolved to HARDWARE modulation \
                                 in RAW acquisition mode. Set \
                                reset_oscillator_phase=True on the \
                                acquire_loop_rt, or use \
                                ModulationType.SOFTWARE explicitly, to avoid \
                                the signal averaging out.",
                                id_store.resolve(osc.uid).unwrap(),
                                id_store.resolve(signal.uid).unwrap()
                            );
                        }
                        OscillatorKind::Hardware
                    } else {
                        OscillatorKind::Hardware
                    }
                }
                DeviceKind::Uhfqa => {
                    if context.acquisition_type().is_spectroscopy() {
                        OscillatorKind::Hardware
                    } else {
                        OscillatorKind::Software
                    }
                }
                DeviceKind::Hdawg => {
                    if signal.kind == SignalKind::Rf {
                        // For HDAWG RF signals, SW modulation tends to be more useful
                        OscillatorKind::Software
                    } else {
                        OscillatorKind::Hardware
                    }
                }
                _ => OscillatorKind::Hardware,
            };
            osc.kind = oscillator_kind;
            info!(
                "Resolved modulation type of oscillator on signal: '{}' to {}",
                id_store.resolve(signal.uid).unwrap(),
                osc.kind
            );
        }
    }
    Ok(())
}

fn max_acquisition_length_seconds(
    signal_uid: &SignalUid,
    context: &ExperimentContext,
    sampling_rate: Frequency<Hertz>,
) -> Option<Duration<Second>> {
    if let Some((max_seconds, max_samples)) = context.maximum_acquisition_lengths(signal_uid) {
        let longest = max_seconds
            .value()
            .max(*max_samples as f64 / sampling_rate.value());
        Some(longest.into())
    } else {
        None
    }
}

/// Generates the signals from the signal properties and device information.
fn process_signals(
    signal_properties: Vec<DeviceSignal>,
    sampling_rates: &HashMap<SignalUid, Frequency<Hertz>>,
    devices: &HashMap<DeviceUid, &AwgDevice>,
    delays: &DelayRegistry,
    id_store: &NamedIdStore,
    backend_processed: &impl PreprocessedBackendData,
) -> Result<Vec<Signal>> {
    let signals = signal_properties
        .into_iter()
        .map(|prop| -> Result<Signal> {
            let device = devices.get(&prop.device_uid).unwrap();
            let sampling_rate = sampling_rates.get(&prop.uid).unwrap().value();

            let signal_delay = round_signal_delay(
                prop.calibration.signal_delay,
                sampling_rate,
                device.traits().sample_multiple,
                prop.uid,
                id_store,
            )?;

            let signal = Signal {
                uid: prop.uid,
                ports: prop.ports.into(),
                automute: prop.calibration.automute,
                port_mode: prop.calibration.port_mode,
                port_delay: prop.calibration.port_delay,
                start_delay: delays.signal_start_delay(prop.uid),
                amplitude: prop.calibration.amplitude,
                range: prop.calibration.range,
                precompensation: prop.calibration.precompensation,
                signal_delay,
                awg_key: backend_processed.awg_key(prop.uid)?,
                device_uid: prop.device_uid,
                sampling_rate,
                amplifier_pump: prop.calibration.amplifier_pump,
                oscillator: prop.calibration.oscillator,
                lo_frequency: prop.calibration.lo_frequency,
                voltage_offset: prop.calibration.voltage_offset,
                kind: prop.kind,
                added_outputs: prop.calibration.added_outputs,
                thresholds: prop.calibration.thresholds,
                mixer_calibration: prop.calibration.mixer_calibration,
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
    signals: &mut [DeviceSignal],
    devices: &HashMap<DeviceUid, &AwgDevice>,
    backend_processed: &impl PreprocessedBackendData,
) -> Result<()> {
    // Find AWGs that may need precompensation adaptation
    let should_adapt_pc: Vec<_> = signals
        .iter()
        .filter_map(|s| {
            let device = devices.get(&s.device_uid).unwrap();
            if s.calibration.precompensation.is_some() && !device.traits().supports_precompensation
            {
                return Some(Err(Error::new(format!(
                    "Precompensation is not supported on device '{}'.",
                    device.kind(),
                ))));
            }
            if s.calibration.precompensation.is_some() && device.kind() == DeviceKind::Hdawg {
                Some(Ok(backend_processed
                    .awg_key(s.uid)
                    .expect("Expected AWG key")))
            } else {
                None
            }
        })
        .collect::<Result<Vec<_>>>()?;

    // Extract precompensations for adaptation
    let mut pcs = Vec::new();
    for signal in signals.iter_mut() {
        let awg_key = &backend_processed.awg_key(signal.uid)?;
        if !should_adapt_pc.contains(awg_key) {
            continue;
        }
        if let Some(precomp) = signal.calibration.precompensation.take() {
            pcs.push(AssignedPrecompensation {
                signal_uid: signal.uid,
                awg: *awg_key,
                precompensation: Some(precomp),
            });
        } else {
            // Placeholder for signal without precompensation, to ensure it gets adapted if needed
            pcs.push(AssignedPrecompensation {
                signal_uid: signal.uid,
                awg: *awg_key,
                precompensation: None,
            });
        }
    }

    adapt_precompensations(&mut pcs)?;

    // Apply adapted precompensations back to signals
    for pc in pcs {
        if let Some(signal) = signals.iter_mut().find(|s| s.uid == pc.signal_uid) {
            signal.calibration.precompensation = pc.precompensation;
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
    signals: &[DeviceSignal],
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
                &s.ports,
                sampling_rates.get(&s.uid).unwrap().value(),
                device.uid(),
                device.kind(),
                s.calibration
                    .added_outputs
                    .iter()
                    .map(|r| r.source_channel.as_str())
                    .collect(),
                s.calibration.precompensation.as_ref(),
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
