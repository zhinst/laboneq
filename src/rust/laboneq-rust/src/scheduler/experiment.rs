// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::str::FromStr;

use crate::error::Error;
use crate::scheduler::NamedIdStore;
use crate::scheduler::pulse::PulseDef;
use crate::scheduler::py_object_interner::PyObjectInterner;
use laboneq_common::types::{AwgKey, DeviceKind, PhysicalDeviceUid};
use laboneq_dsl::{
    ExperimentNode,
    types::{
        AmplifierPump, DeviceUid, ExternalParameterUid, Oscillator, ParameterUid, PulseUid,
        SignalUid, SweepParameter, ValueOrParameter,
    },
};
use laboneq_units::duration::{Duration, Second};
use smallvec::SmallVec;

pub(crate) struct Experiment {
    /// Root node of the experiment tree
    pub root: ExperimentNode,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub pulses: HashMap<PulseUid, PulseDef>,
    pub py_object_store: PyObjectInterner<ExternalParameterUid>,
}

/// Device and signal setup used in the experiment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct DeviceSetup {
    pub signals: HashMap<SignalUid, Signal>,
    pub devices: HashMap<DeviceUid, Device>,
}

impl DeviceSetup {
    pub(crate) fn new(
        signals: HashMap<SignalUid, Signal>,
        devices: HashMap<DeviceUid, Device>,
    ) -> Result<Self, Error> {
        // Validate all signals reference existing devices
        for signal in signals.values() {
            if !devices.contains_key(&signal.device_uid) {
                return Err(Error::new(format!(
                    "Signal '{}' 'references unknown device",
                    signal.uid.0
                )));
            }
        }
        Ok(Self { signals, devices })
    }
}

/// Device used in the experiment.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Device {
    pub uid: DeviceUid,
    /// Physical device this device maps to
    /// This UID is used to group virtual devices that share the same
    /// physical hardware, enabling proper device detection.
    pub physical_device_uid: PhysicalDeviceUid,
    /// Whether the device is part of a SHFQC
    /// This is needed as SHFQC device is split internally into virtual
    /// SHFQA + SHFSG devices.
    pub is_shfqc: bool,
    pub kind: DeviceKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Signal {
    pub uid: SignalUid,
    pub sampling_rate: f64,
    pub awg_key: AwgKey,
    pub device_uid: DeviceUid,
    pub oscillator: Option<Oscillator>,
    pub lo_frequency: Option<ValueOrParameter<f64>>,
    pub voltage_offset: Option<ValueOrParameter<f64>>,
    pub amplifier_pump: Option<AmplifierPump>,
    pub kind: SignalKind,
    pub channels: SmallVec<[u16; 1]>,
    pub port_mode: Option<PortMode>,
    pub automute: bool,
    pub signal_delay: Duration<Second>,
    pub port_delay: ValueOrParameter<Duration<Second>>,
    pub start_delay: Duration<Second>,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PortMode {
    LF,
    RF,
}

impl FromStr for PortMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lf" => Ok(PortMode::LF),
            "rf" => Ok(PortMode::RF),
            _ => Err(format!("Unknown port mode: {}", s)),
        }
    }
}

#[cfg(test)]
pub(crate) mod builders {
    use super::{Signal, SignalKind};
    use laboneq_common::types::AwgKey;
    use laboneq_dsl::types::{AmplifierPump, DeviceUid, Oscillator, SignalUid, ValueOrParameter};
    use laboneq_units::duration::{Duration, Second, seconds};
    use smallvec::smallvec;

    pub(crate) struct SignalBuilder {
        inner: Signal,
    }

    impl SignalBuilder {
        pub(crate) fn new(
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
                    signal_delay: 0.0.into(),
                    port_delay: ValueOrParameter::Value(seconds(0.0)),
                    start_delay: 0.0.into(),
                },
            }
        }

        pub(crate) fn oscillator(mut self, oscillator: Oscillator) -> Self {
            self.inner.oscillator = Some(oscillator);
            self
        }

        #[expect(dead_code)]
        pub(crate) fn lo_frequency(mut self, lo_frequency: ValueOrParameter<f64>) -> Self {
            self.inner.lo_frequency = Some(lo_frequency);
            self
        }

        #[expect(dead_code)]
        pub(crate) fn voltage_offset(mut self, voltage_offset: ValueOrParameter<f64>) -> Self {
            self.inner.voltage_offset = Some(voltage_offset);
            self
        }

        #[expect(dead_code)]
        pub(crate) fn amplifier_pump(mut self, amplifier_pump: AmplifierPump) -> Self {
            self.inner.amplifier_pump = Some(amplifier_pump);
            self
        }

        pub(crate) fn signal_delay(mut self, signal_delay: f64) -> Self {
            self.inner.signal_delay = signal_delay.into();
            self
        }

        pub(crate) fn port_delay(mut self, port_delay: ValueOrParameter<Duration<Second>>) -> Self {
            self.inner.port_delay = port_delay;
            self
        }

        pub(crate) fn start_delay(mut self, start_delay: f64) -> Self {
            self.inner.start_delay = start_delay.into();
            self
        }

        pub(crate) fn build(self) -> Signal {
            self.inner
        }
    }
}
