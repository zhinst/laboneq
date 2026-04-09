// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::AwgKey;
use laboneq_dsl::{
    signal_calibration::{MixerCalibration, OutputRoute, PortMode, Precompensation},
    types::{AmplifierPump, DeviceUid, Oscillator, Quantity, SignalUid, ValueOrParameter},
};
use laboneq_units::duration::{Duration, Second};
use smallvec::SmallVec;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    // Identification parameters
    pub uid: SignalUid,
    pub awg_key: AwgKey,
    pub device_uid: DeviceUid,

    // Configuration parameters
    pub sampling_rate: f64,
    pub port_mode: Option<PortMode>,
    pub channels: SmallVec<[u16; 4]>,
    pub kind: SignalKind,

    // Calibration parameters
    pub amplitude: Option<ValueOrParameter<f64>>,
    pub oscillator: Option<Oscillator>,
    pub lo_frequency: Option<ValueOrParameter<f64>>,
    pub voltage_offset: Option<ValueOrParameter<f64>>,
    pub amplifier_pump: Option<AmplifierPump>,
    pub automute: bool,
    pub range: Option<Quantity>,
    pub precompensation: Option<Precompensation>,
    pub added_outputs: Vec<OutputRoute>,
    pub thresholds: Vec<f64>,
    pub mixer_calibration: Option<MixerCalibration>,

    // Timing parameters
    pub port_delay: ValueOrParameter<Duration<Second>>,
    pub start_delay: Duration<Second>,
    pub signal_delay: Duration<Second>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalKind {
    Rf,
    Integration,
    Iq,
}

impl FromStr for SignalKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rf" => Ok(SignalKind::Rf),
            "iq" => Ok(SignalKind::Iq),
            "integration" => Ok(SignalKind::Integration),
            _ => Err(format!("Unknown signal type: {}", s)),
        }
    }
}

pub mod builder {
    use super::*;
    use laboneq_units::duration::seconds;
    use smallvec::smallvec;

    pub struct SignalBuilder {
        inner: Signal,
    }

    impl SignalBuilder {
        pub fn new(
            uid: SignalUid,
            sampling_rate: f64,
            awg_key: AwgKey,
            device_uid: DeviceUid,
            kind: SignalKind,
        ) -> Self {
            Self {
                inner: Signal {
                    uid,
                    sampling_rate,
                    awg_key,
                    device_uid,
                    oscillator: None,
                    lo_frequency: None,
                    voltage_offset: None,
                    kind,
                    amplifier_pump: None,
                    channels: smallvec![],
                    port_mode: None,
                    automute: false,
                    amplitude: None,
                    signal_delay: 0.0.into(),
                    port_delay: ValueOrParameter::Value(seconds(0.0)),
                    start_delay: 0.0.into(),
                    range: None,
                    precompensation: None,
                    added_outputs: vec![],
                    thresholds: vec![],
                    mixer_calibration: None,
                },
            }
        }

        pub fn oscillator(mut self, oscillator: Oscillator) -> Self {
            self.inner.oscillator = Some(oscillator);
            self
        }

        pub fn lo_frequency(mut self, lo_frequency: ValueOrParameter<f64>) -> Self {
            self.inner.lo_frequency = Some(lo_frequency);
            self
        }

        pub fn voltage_offset(mut self, voltage_offset: ValueOrParameter<f64>) -> Self {
            self.inner.voltage_offset = Some(voltage_offset);
            self
        }

        pub fn amplifier_pump(mut self, amplifier_pump: AmplifierPump) -> Self {
            self.inner.amplifier_pump = Some(amplifier_pump);
            self
        }

        pub fn signal_delay(mut self, signal_delay: f64) -> Self {
            self.inner.signal_delay = signal_delay.into();
            self
        }

        pub fn port_delay(mut self, port_delay: ValueOrParameter<Duration<Second>>) -> Self {
            self.inner.port_delay = port_delay;
            self
        }

        pub fn start_delay(mut self, start_delay: f64) -> Self {
            self.inner.start_delay = start_delay.into();
            self
        }

        pub fn range(mut self, range: Quantity) -> Self {
            self.inner.range = Some(range);
            self
        }

        pub fn precompensation(mut self, precompensation: Precompensation) -> Self {
            self.inner.precompensation = Some(precompensation);
            self
        }

        pub fn add_output(mut self, output_route: OutputRoute) -> Self {
            self.inner.added_outputs.push(output_route);
            self
        }

        pub fn amplitude(mut self, amplitude: ValueOrParameter<f64>) -> Self {
            self.inner.amplitude = Some(amplitude);
            self
        }

        pub fn automute(mut self, automute: bool) -> Self {
            self.inner.automute = automute;
            self
        }

        pub fn port_mode(mut self, port_mode: PortMode) -> Self {
            self.inner.port_mode = Some(port_mode);
            self
        }

        pub fn channels(mut self, channels: Vec<u16>) -> Self {
            self.inner.channels = SmallVec::from_vec(channels);
            self
        }

        pub fn build(self) -> Signal {
            self.inner
        }
    }
}
