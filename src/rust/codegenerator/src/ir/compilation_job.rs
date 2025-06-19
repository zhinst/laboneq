// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::device_traits;
use crate::utils::normalize_f64;
use anyhow::anyhow;
use numeric_array::NumericArray;
use std::hash::Hash;
use std::{collections::HashMap, rc::Rc};

/// Represents different kinds of pulse definitions.
#[derive(Debug, Clone, PartialEq)]
pub enum PulseDefKind {
    /// Analog waveform
    Pulse,
    /// Digital marker
    Marker,
}

#[derive(Debug, Clone)]
pub struct PulseDef {
    pub uid: String,
    pub kind: PulseDefKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OscillatorKind {
    SOFTWARE,
    HARDWARE,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Oscillator {
    pub uid: String,
    pub kind: OscillatorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalKind {
    IQ,
    SINGLE,
    INTEGRATION,
}

#[derive(Debug, Clone)]
pub struct Signal {
    pub uid: String,
    pub kind: SignalKind,
    pub channels: Vec<u8>,
    pub delay: i64,
    pub oscillator: Option<Oscillator>,
    pub mixer_type: Option<MixerType>,
}

impl Signal {
    pub(crate) fn is_hw_modulated(&self) -> bool {
        self.oscillator
            .as_ref()
            .is_some_and(|x| matches!(x.kind, OscillatorKind::HARDWARE))
    }

    pub(crate) fn is_sw_modulated(&self) -> bool {
        self.oscillator
            .as_ref()
            .is_some_and(|x| matches!(x.kind, OscillatorKind::SOFTWARE))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AwgKind {
    /// Only one channel is played
    SINGLE,
    /// Two independent channels
    DOUBLE,
    /// Two channels form an I/Q signal
    IQ,
    /// Multiple logical I/Q channels mixed
    MULTI,
}

#[derive(Debug, Clone)]
pub struct AwgCore {
    pub kind: AwgKind,
    // AWG signals
    // In the case of multiplexed, signals with different UID points to the same channel(s)
    pub signals: Vec<Rc<Signal>>,
    pub sampling_rate: f64,
    pub device_kind: DeviceKind,
    // Mapping from HW oscillator to an assigned index
    pub osc_allocation: HashMap<String, u16>,
}

impl AwgCore {
    /// Use command table for pulses
    pub(crate) fn use_command_table(&self) -> bool {
        matches!(self.device_kind, DeviceKind::SHFSG | DeviceKind::HDAWG)
    }

    /// Use command table to set phase / amplitude
    pub(crate) fn use_command_table_phase_amp(&self) -> bool {
        match self.kind {
            AwgKind::DOUBLE => false,
            _ => self.use_command_table(),
        }
    }

    /// Use command table for amplitude increments
    pub(crate) fn use_amplitude_increment(&self) -> bool {
        !matches!(self.kind, AwgKind::DOUBLE)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DeviceKind {
    HDAWG,
    SHFQA,
    SHFSG,
    UHFQA,
}

impl DeviceKind {
    pub const fn traits(&self) -> &device_traits::DeviceTraits {
        match self {
            DeviceKind::HDAWG => &device_traits::HDAWG_TRAITS,
            DeviceKind::SHFQA => &device_traits::SHFQA_TRAITS,
            DeviceKind::SHFSG => &device_traits::SHFSG_TRAITS,
            DeviceKind::UHFQA => &device_traits::UHFQA_TRAITS,
        }
    }

    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            DeviceKind::HDAWG => "HDAWG",
            DeviceKind::SHFQA => "SHFQA",
            DeviceKind::SHFSG => "SHFSG",
            DeviceKind::UHFQA => "UHFQA",
        }
    }
}
impl std::str::FromStr for DeviceKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<DeviceKind, anyhow::Error> {
        match s.to_uppercase().as_str() {
            "HDAWG" => Ok(DeviceKind::HDAWG),
            "SHFQA" => Ok(DeviceKind::SHFQA),
            "SHFSG" => Ok(DeviceKind::SHFSG),
            "UHFQA" => Ok(DeviceKind::UHFQA),
            _ => Err(anyhow!(
                "Unsupported device type: {}. Supported types are: SHFQA, SHFSG, HDAWG, UHFQA",
                s
            )),
        }
    }
}

/// Pulse marker
#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    /// ID of the marker
    pub marker_selector: String,
    pub enable: bool,
    /// Start of the marker relative to the beginning of the pulse
    pub start: Option<f64>,
    // TODO: Marker can actually only have either `length` or `pulse_id`!
    /// Length of the pulse
    pub length: Option<f64>,
    /// Marker pulse ID
    pub pulse_id: Option<String>,
}

impl Marker {
    pub fn new(
        marker_selector: String,
        enable: bool,
        start: Option<f64>,
        length: Option<f64>,
        pulse_id: Option<String>,
    ) -> Self {
        Marker {
            marker_selector,
            enable,
            start,
            length,
            pulse_id,
        }
    }
}

impl Hash for Marker {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.marker_selector.hash(state);
        self.enable.hash(state);
        if let Some(start) = self.start {
            normalize_f64(start).hash(state);
        }
        if let Some(length) = self.length {
            normalize_f64(length).hash(state);
        }
        self.pulse_id.hash(state);
    }
}

/// Parameter that is swept over.
#[derive(Debug, Clone)]
pub struct SweepParameter {
    /// UID of the parameter
    pub uid: String,
    /// Values of the parameter
    pub values: NumericArray,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MixerType {
    /// Mixer performs full complex modulation
    IQ,
    /// Mixer only performs envelope modulation (UHFQA-style)
    UhfqaEnvelope,
}
