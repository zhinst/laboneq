// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::ir::Samples;
use crate::ir::compilation_job::{AwgCore, AwgKind, DeviceKind, SignalKind};
use crate::tinysample::samples_to_grid;
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

/// Calculate the delays for an AWG based on its signals and code generation delays.
///
/// This function computes the total delay for the AWG and individual delays for each signal.
/// It ensures that all signals on the AWG share the same delay, except for integration signals.
/// If the delays differ for RF signals, an error is returned.
///
/// # Arguments
///
/// * `awg`: The AWG core object containing the signals and their properties.
/// * `delays`: A map of signal UIDs to additional delays.
pub fn calculate_awg_delays(awg: &AwgCore, delays: &HashMap<&str, Samples>) -> Result<AwgTiming> {
    validate_signal_delays(awg)?;
    let mut signal_delays: HashMap<String, Samples> = HashMap::new();
    let mut awg_delay: Option<Samples> = None;
    for signal in awg.signals.iter() {
        let (total_delay, remainder) = samples_to_grid(
            signal.delay() + delays.get(signal.uid.as_str()).cloned().unwrap_or(0),
            awg.device_kind().traits().sample_multiple.into(),
        );
        if remainder != 0 {
            return Err(Error::new(&format!(
                "Internal error: Signal {} has a delay of {} samples, which is not a multiple of the device's sample multiple {}.",
                signal.uid,
                signal.delay(),
                awg.device_kind().traits().sample_multiple
            )));
        }
        signal_delays.insert(signal.uid.clone(), total_delay);
        if signal.kind != SignalKind::INTEGRATION {
            // Evaluate the common lead time for the AWG.
            // The minimum of all delays is used for the AWG.
            let lead_time = if awg.device_kind() == &DeviceKind::UHFQA {
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
    let awg_delay_samples = awg_delay.map_or(0, |delay| {
        if delay < awg.device_kind().traits().min_play_wave.into() {
            0
        } else {
            delay
        }
    });
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
        ir::compilation_job::{AwgCore, Device, Signal, SignalKind},
    };
    use std::{collections::HashMap, sync::Arc};

    fn create_signal(uid: &str, kind: SignalKind, delay: Samples) -> Signal {
        Signal {
            uid: uid.to_string(),
            kind,
            signal_delay: delay,
            start_delay: 0,
            channels: vec![],
            oscillator: None,
            mixer_type: None,
            automute: false,
        }
    }

    fn create_awg_core(signals: Vec<Signal>) -> AwgCore {
        AwgCore::new(
            0,
            AwgKind::MULTI,
            signals.iter().map(|s| Arc::new(s.clone())).collect(),
            2e9,
            Arc::new(Device::new("".to_string().into(), DeviceKind::SHFQA)),
            HashMap::new(),
            None,
            false,
        )
    }

    #[test]
    fn test_calculate_delays() {
        let awg = create_awg_core(vec![
            create_signal("acq", SignalKind::INTEGRATION, 0),
            create_signal("meas0", SignalKind::IQ, 320),
            create_signal("meas1", SignalKind::IQ, 160),
        ]);

        // Test with no additional delays
        let timing =
            calculate_awg_delays(&awg, &HashMap::new()).expect("Failed to calculate AWG delays");
        assert_eq!(timing.delay(), (8e-8 * awg.sampling_rate) as Samples);
        assert_eq!(timing.signal_delay("meas0"), 320);
        assert_eq!(timing.signal_delay("meas1"), 160);

        // Test with additional delays
        let timing = calculate_awg_delays(
            &awg,
            &HashMap::from_iter(vec![("meas0", 0), ("meas1", 640)]),
        )
        .expect("Failed to calculate AWG delays");
        assert_eq!(timing.delay(), 320);
        assert_eq!(timing.signal_delay("meas0"), 320);
        assert_eq!(timing.signal_delay("meas1"), 160 + 640);
    }

    /// Test that values between 0 and the minimum play wave are rounded to 0.
    #[test]
    fn test_calculate_delays_minimum_awg_delay() {
        let delay = (SHFQA_TRAITS.min_play_wave / 2) / SHFQA_TRAITS.sampling_rate as u32;
        let awg = create_awg_core(vec![create_signal("meas1", SignalKind::IQ, delay.into())]);
        let timing =
            calculate_awg_delays(&awg, &HashMap::new()).expect("Failed to calculate AWG delays");
        assert_eq!(timing.delay(), 0);
    }
}
