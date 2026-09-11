// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU16;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{OscillatorKind, SignalUid};
use laboneq_ir::signal::Signal;
use laboneq_log::debug;
use laboneq_units::duration::{Duration, Second};

use crate::compiler_backend::PreprocessedBackendData;
use crate::error::{Error, Result};
use crate::experiment::DeviceSignal;
use crate::setup_processor::DelayRegistry;
use crate::setup_processor::delays::{SignalDelayProperties, compute_signal_delays};

/// Build the signals the rest of the compiler works with.
///
/// TODO: Most of the functionality could be moved to the Compiler backends.
pub(crate) fn process_setup(
    signals: Vec<DeviceSignal>,
    backend_processed: &impl PreprocessedBackendData,
    id_store: &NamedIdStore,
) -> Result<Vec<Signal>> {
    let delays = compute_delays(&signals, backend_processed);
    process_signals(signals, &delays, id_store, backend_processed)
}

/// Assemble each signal from its setup properties and its computed delays.
fn process_signals(
    signal_properties: Vec<DeviceSignal>,
    delays: &DelayRegistry,
    id_store: &NamedIdStore,
    backend_processed: &impl PreprocessedBackendData,
) -> Result<Vec<Signal>> {
    let signals = signal_properties
        .into_iter()
        .map(|prop| -> Result<Signal> {
            let sampling_rate = prop.sampling_rate;

            // Resolving AUTO modulation requires hardware knowledge and is therefore the
            // responsibility of the compiler backend.
            if let Some(osc) = &prop.calibration.oscillator
                && osc.kind == OscillatorKind::Auto
            {
                return Err(Error::new(format!(
                    "Internal error: modulation type of the oscillator on signal '{}' \
                     was not resolved by the compiler backend.",
                    id_store.resolve(prop.uid).unwrap()
                )));
            }

            let signal_delay = round_signal_delay(
                prop.calibration.signal_delay,
                sampling_rate.value(),
                prop.sample_multiple,
                prop.uid,
                id_store,
            )?;

            let signal = Signal {
                uid: prop.uid,
                automute: prop.calibration.automute,
                port_mode: prop.calibration.port_mode,
                port_delay: prop.calibration.port_delay,
                start_delay: delays.signal_start_delay(prop.uid),
                signal_alignment_delay: delays.signal_alignment_delay(prop.uid),
                amplitude: prop.calibration.amplitude,
                range: prop.calibration.range,
                precompensation: prop.calibration.precompensation,
                signal_delay,
                awg_key: backend_processed.awg_key(prop.uid)?,
                device_uid: prop.device_uid,
                physical_device_uid: prop.physical_device_uid,
                traits: prop.traits,
                sampling_rate: prop.sampling_rate.value(),
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

/// Round a calibrated signal delay onto the signal's sequencer grid.
///
/// Errors on a negative delay, and logs when rounding moves it by more than a
/// sample.
fn round_signal_delay(
    delay: Duration<Second>,
    sampling_rate: f64,
    sample_multiple: NonZeroU16,
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

/// Round a delay to the nearest whole number of sequencer grid steps, with ties
/// rounded towards zero.
fn round_delay(
    delay: Duration<Second>,
    sampling_rate: f64,
    sample_multiple: NonZeroU16,
) -> Duration<Second> {
    let samples = delay * sampling_rate;
    let sample_multiple = sample_multiple.get() as f64;
    let delay_samples = ((samples.value() / sample_multiple + 0.5).ceil() - 1.0) * sample_multiple;
    (delay_samples / sampling_rate).into()
}

/// Work out how long each signal waits so that they all start together.
///
/// A signal arrives carrying the delay its backend already accounts for, such
/// as output routing or precompensation. The largest of those sets the common
/// start; every signal waits the difference, on top of the head start its
/// instrument needs.
fn compute_delays(
    signals: &[DeviceSignal],
    backend_processed: &impl PreprocessedBackendData,
) -> DelayRegistry {
    let signal_delay_props = signals
        .iter()
        .map(|s| SignalDelayProperties {
            uid: s.uid,
            sampling_rate: s.sampling_rate.value(),
            sample_multiple: s.sample_multiple,
            delay_signal: s.delay_signal,
            lead_delay: backend_processed.lead_delay(s.uid),
        })
        .collect::<Vec<_>>();

    compute_signal_delays(&signal_delay_props)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_delay_snaps_to_the_sequencer_grid() {
        let sampling_rate = 2.0e9;
        let sample_multiple = NonZeroU16::new(4).unwrap();

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
                "{} samples should round to {}, but got {}.",
                delay_samples, delay_rounded_samples, rounded_samples
            );
        }
    }
}
