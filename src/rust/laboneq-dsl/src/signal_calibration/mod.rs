// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::types::{AmplifierPump, Oscillator, Quantity, ValueOrParameter};
use laboneq_units::duration::{Duration, Second};

mod mixer_calibration;
mod output_route;
mod port_mode;
mod precompensation;

pub use mixer_calibration::*;
pub use output_route::*;
pub use port_mode::*;
pub use precompensation::*;

pub struct SignalCalibration {
    pub amplitude: Option<ValueOrParameter<f64>>,
    pub added_outputs: Vec<OutputRoute>,
    pub amplifier_pump: Option<AmplifierPump>,
    pub automute: bool,
    pub lo_frequency: Option<ValueOrParameter<f64>>,
    pub oscillator: Option<Oscillator>,
    pub port_delay: ValueOrParameter<Duration<Second>>,
    pub port_mode: Option<PortMode>,
    pub precompensation: Option<Precompensation>,
    pub range: Option<Quantity>,
    pub signal_delay: Duration<Second>,
    pub voltage_offset: Option<ValueOrParameter<f64>>,
    pub thresholds: Vec<f64>,
    pub mixer_calibration: Option<MixerCalibration>,
}
