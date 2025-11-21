// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use pyo3::prelude::*;

use crate::scheduler::NamedIdStore;
use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::{AwgKey, DeviceKind};
use laboneq_scheduler::SignalInfo;
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::sweep_parameter::SweepParameter;
use laboneq_scheduler::experiment::types::{
    AmplifierPump, ExternalParameterUid, Oscillator, ParameterUid, PulseRef, PulseUid, SignalUid,
    ValueOrParameter,
};

pub(crate) struct Experiment {
    pub sections: Vec<ExperimentNode>,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub pulses: HashMap<PulseUid, PulseRef>,
    #[allow(dead_code)]
    // Signal defined in the experiment
    pub experiment_signals: HashSet<SignalUid>, // Not yet used except in tests
    // Resolved signals with full info.
    pub signals: HashMap<SignalUid, Signal>,
    #[allow(dead_code)] // Not yet used except in tests
    pub external_parameters: HashMap<ExternalParameterUid, Py<PyAny>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    pub uid: SignalUid,
    pub sampling_rate: f64,
    pub awg_key: AwgKey,
    pub device_type: DeviceKind,
    pub oscillator: Option<Oscillator>,
    pub lo_frequency: Option<ValueOrParameter<f64>>,
    pub voltage_offset: Option<ValueOrParameter<f64>>,
    pub amplifier_pump: Option<AmplifierPump>,
    pub kind: SignalKind,
}

impl SignalInfo for Signal {
    fn uid(&self) -> SignalUid {
        self.uid
    }

    fn awg_key(&self) -> AwgKey {
        self.awg_key
    }

    fn sampling_rate(&self) -> f64 {
        self.sampling_rate
    }

    fn device_traits(&self) -> &'static DeviceTraits {
        DeviceTraits::from_device_kind(&self.device_type)
    }

    fn oscillator(&self) -> Option<&Oscillator> {
        self.oscillator.as_ref()
    }

    fn lo_frequency(&self) -> Option<&ValueOrParameter<f64>> {
        self.lo_frequency.as_ref()
    }

    fn supports_initial_local_oscillator_frequency(&self) -> bool {
        DeviceTraits::from_device_kind(&self.device_type).device_class != 0
    }

    fn voltage_offset(&self) -> Option<&ValueOrParameter<f64>> {
        self.voltage_offset.as_ref()
    }

    fn supports_initial_voltage_offset(&self) -> bool {
        DeviceTraits::from_device_kind(&self.device_type).device_class != 0
    }

    fn amplifier_pump(&self) -> Option<&AmplifierPump> {
        self.amplifier_pump.as_ref()
    }
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

#[cfg(test)]
pub mod builders {
    use super::{Signal, SignalKind};
    use laboneq_common::types::{AwgKey, DeviceKind};
    use laboneq_scheduler::experiment::types::{
        AmplifierPump, Oscillator, SignalUid, ValueOrParameter,
    };

    pub struct SignalBuilder {
        inner: Signal,
    }

    impl SignalBuilder {
        pub fn new(
            uid: SignalUid,
            sampling_rate: f64,
            awg_key: AwgKey,
            device_type: DeviceKind,
            kind: SignalKind,
        ) -> Self {
            Self {
                inner: Signal {
                    uid,
                    sampling_rate,
                    awg_key,
                    device_type,
                    oscillator: None,
                    lo_frequency: None,
                    voltage_offset: None,
                    kind,
                    amplifier_pump: None,
                },
            }
        }

        pub fn oscillator(mut self, oscillator: Oscillator) -> Self {
            self.inner.oscillator = Some(oscillator);
            self
        }

        #[expect(dead_code)]
        pub fn lo_frequency(mut self, lo_frequency: ValueOrParameter<f64>) -> Self {
            self.inner.lo_frequency = Some(lo_frequency);
            self
        }

        #[expect(dead_code)]
        pub fn voltage_offset(mut self, voltage_offset: ValueOrParameter<f64>) -> Self {
            self.inner.voltage_offset = Some(voltage_offset);
            self
        }

        #[expect(dead_code)]
        pub fn amplifier_pump(mut self, amplifier_pump: AmplifierPump) -> Self {
            self.inner.amplifier_pump = Some(amplifier_pump);
            self
        }

        pub fn build(self) -> Signal {
            self.inner
        }
    }
}
