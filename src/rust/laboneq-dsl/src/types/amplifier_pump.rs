// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::{DeviceUid, ValueOrParameter};

#[derive(Debug, Clone, PartialEq)]
pub struct AmplifierPump {
    pub device: DeviceUid,
    pub channel: u16,
    pub pump_power: Option<ValueOrParameter<f64>>,
    pub pump_frequency: Option<ValueOrParameter<f64>>,
    pub probe_power: Option<ValueOrParameter<f64>>,
    pub probe_frequency: Option<ValueOrParameter<f64>>,
    pub cancellation_phase: Option<ValueOrParameter<f64>>,
    pub cancellation_attenuation: Option<ValueOrParameter<f64>>,
}

impl AmplifierPump {
    pub fn values_or_parameters(&self) -> impl Iterator<Item = &Option<ValueOrParameter<f64>>> {
        [
            &self.pump_power,
            &self.pump_frequency,
            &self.probe_power,
            &self.probe_frequency,
            &self.cancellation_phase,
            &self.cancellation_attenuation,
        ]
        .into_iter()
    }
}

pub struct AmplifierPumpBuilder {
    inner: AmplifierPump,
}

impl AmplifierPumpBuilder {
    pub fn new(device: DeviceUid, channel: u16) -> Self {
        Self {
            inner: AmplifierPump {
                device,
                channel,
                pump_frequency: None,
                pump_power: None,
                probe_power: None,
                probe_frequency: None,
                cancellation_phase: None,
                cancellation_attenuation: None,
            },
        }
    }

    pub fn pump_frequency(mut self, value: ValueOrParameter<f64>) -> Self {
        self.inner.pump_frequency = Some(value);
        self
    }

    pub fn pump_power(mut self, value: ValueOrParameter<f64>) -> Self {
        self.inner.pump_power = Some(value);
        self
    }

    pub fn probe_frequency(mut self, value: ValueOrParameter<f64>) -> Self {
        self.inner.probe_frequency = Some(value);
        self
    }

    pub fn probe_power(mut self, value: ValueOrParameter<f64>) -> Self {
        self.inner.probe_power = Some(value);
        self
    }

    pub fn cancellation_phase(mut self, value: ValueOrParameter<f64>) -> Self {
        self.inner.cancellation_phase = Some(value);
        self
    }

    pub fn cancellation_attenuation(mut self, value: ValueOrParameter<f64>) -> Self {
        self.inner.cancellation_attenuation = Some(value);
        self
    }

    pub fn build(self) -> AmplifierPump {
        self.inner
    }
}
