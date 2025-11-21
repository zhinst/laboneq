// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub use crate::experiment::types::AmplifierPump;
use crate::experiment::types::{
    Chunking, DeviceUid, ParameterUid, Reserve, ResetOscillatorPhase, SectionAlignment, SectionUid,
    SignalUid, Sweep, ValueOrParameter,
};

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

pub struct SweepBuilder {
    inner: Sweep,
}

impl SweepBuilder {
    pub fn new(uid: SectionUid, parameters: Vec<ParameterUid>, count: u32) -> Self {
        Self {
            inner: Sweep {
                uid,
                parameters,
                count,
                alignment: SectionAlignment::Left,
                reset_oscillator_phase: false,
                chunking: None,
            },
        }
    }

    pub fn alignment(mut self, alignment: SectionAlignment) -> Self {
        self.inner.alignment = alignment;
        self
    }

    pub fn reset_oscillator_phase(mut self) -> Self {
        self.inner.reset_oscillator_phase = true;
        self
    }

    pub fn chunking(mut self, chunking: Chunking) -> Self {
        self.inner.chunking = Some(chunking);
        self
    }

    pub fn build(self) -> Sweep {
        self.inner
    }
}

impl Reserve {
    pub fn new(signal: SignalUid) -> Self {
        Self { signal }
    }
}

impl ResetOscillatorPhase {
    pub fn new(signals: Vec<SignalUid>) -> Self {
        Self { signals }
    }
}
