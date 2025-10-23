// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::scheduler::NamedIdStore;
use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::{AwgKey, DeviceKind};
use laboneq_scheduler::SignalInfo;
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{
    Oscillator, Parameter, ParameterUid, PulseRef, PulseUid, RealValue, SignalUid,
};

pub struct Experiment {
    pub sections: Vec<ExperimentNode>,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, Parameter>,
    pub pulses: HashMap<PulseUid, PulseRef>,
    pub signals: HashMap<SignalUid, Signal>,
}

pub struct Signal {
    pub uid: SignalUid,
    pub sampling_rate: f64,
    pub awg_key: AwgKey,
    pub device_type: DeviceKind,
    pub oscillator: Option<Oscillator>,
    pub lo_frequency: Option<RealValue>,
    pub voltage_offset: Option<RealValue>,
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

    fn lo_frequency(&self) -> Option<&RealValue> {
        self.lo_frequency.as_ref()
    }

    fn supports_initial_local_oscillator_frequency(&self) -> bool {
        DeviceTraits::from_device_kind(&self.device_type).device_class != 0
    }

    fn voltage_offset(&self) -> Option<&RealValue> {
        self.voltage_offset.as_ref()
    }

    fn supports_initial_voltage_offset(&self) -> bool {
        DeviceTraits::from_device_kind(&self.device_type).device_class != 0
    }
}
