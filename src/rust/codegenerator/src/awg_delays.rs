// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::Context;

use crate::ir::Samples;
use crate::ir::compilation_job::{AwgCore, AwgKind, DeviceKind, SignalKind};
use crate::tinysample::length_to_samples;
use crate::{Error, Result};

#[derive(Default)]
pub struct AwgTiming {
    delay: Samples,
    signal_delays: HashMap<String, Samples>,
}

impl AwgTiming {
    /// Total delay for the AWG.
    pub fn delay(&self) -> Samples {
        self.delay
    }

    /// Delay for a specific signal identified by its UID.
    pub fn signal_delay(&self, uid: &str) -> Samples {
        self.signal_delays.get(uid).cloned().unwrap_or(0)
    }
}

fn validate_signal_delays(awg: &AwgCore) -> Result<()> {
    if awg.kind != AwgKind::IQ {
        return Ok(());
    }
    let delays: Vec<_> = awg
        .signals
        .iter()
        .filter_map(|s| {
            if s.kind == SignalKind::INTEGRATION {
                None
            } else {
                Some(s.delay())
            }
        })
        .collect();
    if !delays.iter().all(|&x| x == delays[0]) {
        // This error should be caught earlier in the compiler. We check it anyways.
        return Err(Error::new(
            "Signals on AWG RF channels must have the same delay.",
        ));
    }
    Ok(())
}

fn ensure_delay_granularity(
    delay: f64,
    sampling_rate: f64,
    sample_multiple: i64,
) -> Result<Samples> {
    let samples = length_to_samples(delay, sampling_rate);
    if samples % sample_multiple != 0 {
        return Err(Error::new(&format!(
            "Delay {delay} s = {samples} samples is not compatible with the sample multiple of {sample_multiple}.",
        )));
    }
    Ok(samples)
}

/// Calculate the delays for an AWG based on its signals and code generation delays.
///
/// This function computes the total delay for the AWG and individual delays for each signal.
/// It ensures that all signals on the AWG share the same delay, except for integration signals.
/// If the delays differ for RF signals, an error is returned.
///
/// # Arguments
///
/// * `awg`: The AWG core object containing the signals and their properties.
/// * `delays`: A map of signal UIDs to additional delays in seconds.
pub fn calculate_awg_delays(awg: &AwgCore, delays: &HashMap<String, f64>) -> Result<AwgTiming> {
    validate_signal_delays(awg)?;
    let mut signal_delays = HashMap::new();
    let mut awg_delay: Option<f64> = None;
    for signal in awg.signals.iter() {
        let total_delay = signal.delay() + delays.get(&signal.uid).cloned().unwrap_or(0.0);
        signal_delays.insert(signal.uid.clone(), total_delay);
        if signal.kind != SignalKind::INTEGRATION {
            // Evaluate the common lead time for the AWG.
            // The minimum of all delays is used for the AWG.
            let lead_time = if awg.device_kind == DeviceKind::UHFQA {
                // On the UHFQA, we allow an individual delay_signal on the measure (play) line, even though we can't
                // shift the play time with a node on the device
                // for this to work, we need to ignore the play delay when generating code for loop events
                // and we use the start delay (lead time) to calculate the global delay
                signal.start_delay
            } else {
                total_delay
            };
            // Use minimum delay to ensure e.g. loops happen before body events.
            awg_delay = Some(awg_delay.map_or(lead_time, |delay| delay.min(lead_time)));
        }
    }

    let signal_delays: HashMap<String, Samples> = signal_delays
        .into_iter()
        .map(|(signal, delay)| -> Result<(String, Samples)> {
            let delay_samples = ensure_delay_granularity(
                delay,
                awg.sampling_rate,
                awg.device_kind.traits().sample_multiple.into(),
            )
            .context(format!("Invalid delay for signal '{signal}': {delay}"))?;
            Ok((signal, delay_samples))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;

    let mut awg_delay_samples =
        awg_delay.map_or(0, |delay| length_to_samples(delay, awg.sampling_rate));
    if awg_delay_samples < (awg.device_kind.traits().min_play_wave).into() {
        awg_delay_samples = 0;
    }

    let out = AwgTiming {
        delay: awg_delay_samples,
        signal_delays,
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device_traits::SHFQA_TRAITS,
        ir::compilation_job::{AwgCore, Signal, SignalKind},
    };
    use std::{collections::HashMap, rc::Rc};

    fn create_signal(uid: &str, kind: SignalKind, delay: f64) -> Signal {
        Signal {
            uid: uid.to_string(),
            kind,
            signal_delay: delay,
            start_delay: 0.0,
            channels: vec![],
            oscillator: None,
            mixer_type: None,
        }
    }

    fn create_awg_core(signals: Vec<Signal>) -> AwgCore {
        AwgCore {
            kind: AwgKind::MULTI,
            device_kind: DeviceKind::SHFQA,
            sampling_rate: 2e9,
            osc_allocation: HashMap::new(),
            signals: signals.iter().map(|s| Rc::new(s.clone())).collect(),
        }
    }

    #[test]
    fn test_calculate_delays() {
        let awg = create_awg_core(vec![
            create_signal("acq", SignalKind::INTEGRATION, 0.0),
            create_signal("meas0", SignalKind::IQ, 16e-8),
            create_signal("meas1", SignalKind::IQ, 8e-8),
        ]);

        // Test with no additional delays
        let timing =
            calculate_awg_delays(&awg, &HashMap::new()).expect("Failed to calculate AWG delays");
        assert_eq!(timing.delay(), (8e-8 * awg.sampling_rate) as i64);
        assert_eq!(
            timing.signal_delay("meas0"),
            (16e-8 * awg.sampling_rate) as i64
        );
        assert_eq!(
            timing.signal_delay("meas1"),
            (8e-8 * awg.sampling_rate) as i64
        );

        // Test with additional delays
        let timing = calculate_awg_delays(
            &awg,
            &HashMap::from_iter(vec![
                ("meas0".to_string(), 0.0),
                ("meas1".to_string(), 32e-8),
            ]),
        )
        .expect("Failed to calculate AWG delays");
        assert_eq!(timing.delay(), (16e-8 * awg.sampling_rate) as i64);
        assert_eq!(
            timing.signal_delay("meas0"),
            (16e-8 * awg.sampling_rate) as i64
        );
        assert_eq!(
            timing.signal_delay("meas1"),
            ((8e-8 + 32e-8) * awg.sampling_rate) as i64
        );
    }

    /// Test that values between 0 and the minimum play wave are rounded to 0.
    #[test]
    fn test_calculate_delays_minimum_awg_delay() {
        let delay = (SHFQA_TRAITS.min_play_wave as f64 / 2.0) / SHFQA_TRAITS.sampling_rate;
        let awg = create_awg_core(vec![create_signal("meas1", SignalKind::IQ, delay)]);
        let timing =
            calculate_awg_delays(&awg, &HashMap::new()).expect("Failed to calculate AWG delays");
        assert_eq!(timing.delay(), 0);
    }
}
