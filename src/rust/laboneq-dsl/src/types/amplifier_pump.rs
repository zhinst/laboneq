// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::str::FromStr;

use crate::types::{DeviceUid, ValueOrParameter};

#[derive(Debug, Clone, PartialEq)]
pub struct AmplifierPump {
    pub device: DeviceUid,
    pub channel: u16,
    pub alc_on: bool,
    pub pump_on: bool,
    pub pump_filter_on: bool,
    pub pump_power: Option<ValueOrParameter<f64>>,
    pub pump_frequency: Option<ValueOrParameter<f64>>,
    pub probe_on: bool,
    pub probe_power: Option<ValueOrParameter<f64>>,
    pub probe_frequency: Option<ValueOrParameter<f64>>,
    pub cancellation_on: bool,
    pub cancellation_phase: Option<ValueOrParameter<f64>>,
    pub cancellation_attenuation: Option<ValueOrParameter<f64>>,
    pub cancellation_source: PumpCancellationSource,
    pub cancellation_source_frequency: Option<f64>,
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

#[derive(Debug, Clone, PartialEq, Copy, Eq, Default)]
pub enum PumpCancellationSource {
    #[default]
    Internal,
    External,
}

impl FromStr for PumpCancellationSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "internal" => Ok(PumpCancellationSource::Internal),
            "external" => Ok(PumpCancellationSource::External),
            _ => Err(format!("Unknown pump cancellation source: {}", s)),
        }
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
                cancellation_source: PumpCancellationSource::default(),
                cancellation_source_frequency: None,
                alc_on: false,
                pump_on: false,
                pump_filter_on: false,
                probe_on: false,
                cancellation_on: false,
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
