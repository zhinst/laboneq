// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::types::AwgKey;
use laboneq_dsl::signal_calibration::Precompensation;
use laboneq_dsl::types::{
    AmplifierPump, DeviceUid, Oscillator, Quantity, SignalUid, ValueOrParameter,
};
use laboneq_ir::signal::{OutputRoute, PortMode, SignalKind};
use laboneq_units::duration::{Duration, Second};
use smallvec::SmallVec;

/// Properties of a signal.
///
/// This is intermediate data structure, which resembles the Python `SignalInfo` class.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SignalProperties {
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
    pub oscillator: Option<Oscillator>,
    pub lo_frequency: Option<ValueOrParameter<f64>>,
    pub voltage_offset: Option<ValueOrParameter<f64>>,
    pub amplifier_pump: Option<AmplifierPump>,
    pub automute: bool,
    pub range: Option<Quantity>,
    pub precompensation: Option<Precompensation>,
    pub added_outputs: Vec<OutputRoute>,

    // Timing parameters
    pub port_delay: ValueOrParameter<Duration<Second>>,
    pub signal_delay: Duration<Second>,
}
