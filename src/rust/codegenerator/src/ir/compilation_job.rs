// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use anyhow::anyhow;
use numeric_array::NumericArray;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use crate::device_traits;
use crate::ir::SignalUid;
use crate::utils::normalize_f64;

pub type Samples = i64;
pub type ChannelIndex = u8;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceUid(Arc<String>);

impl Deref for DeviceUid {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<&str> for DeviceUid {
    fn from(s: &str) -> Self {
        DeviceUid(Arc::new(s.to_string()))
    }
}

impl From<String> for DeviceUid {
    fn from(s: String) -> Self {
        DeviceUid(Arc::new(s))
    }
}

/// Represents different kinds of pulse definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PulseDefKind {
    /// Analog waveform
    Pulse,
    /// Digital marker
    Marker,
}

/// Represents the type of an pulse.
///
/// If [`PulseType::Function`] is used, the pulse is defined by a function
/// that generates the waveform.
/// If [`PulseType::Samples`] is used, the pulse is defined by a set of samples
/// that represent the waveform.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PulseType {
    Function,
    Samples,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulseDef {
    pub uid: String,
    pub kind: PulseDefKind,
    pub pulse_type: Option<PulseType>,
}

#[cfg(test)]
impl PulseDef {
    pub fn test(uid: String, kind: PulseDefKind) -> Self {
        PulseDef {
            uid,
            kind,
            pulse_type: Some(PulseType::Function),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum OscillatorKind {
    SOFTWARE,
    HARDWARE,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Oscillator {
    pub uid: String,
    pub kind: OscillatorKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalKind {
    IQ,
    SINGLE,
    INTEGRATION,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    pub uid: SignalUid,
    pub kind: SignalKind,
    pub channels: Vec<ChannelIndex>,
    pub oscillator: Option<Oscillator>,
    /// The delay from the trigger to the start of the sequence (lead time).
    /// Includes lead time and precompensation
    pub start_delay: Samples,
    // Additional delay on the signal
    pub signal_delay: Samples,
    // The signal output can be automatically muted when no waveforms are played
    pub automute: bool,
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

    pub fn delay(&self) -> Samples {
        self.start_delay + self.signal_delay
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AwgKind {
    /// Only one channel is played
    SINGLE,
    /// Two independent channels
    DOUBLE,
    /// Two channels form an I/Q signal
    IQ,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AwgKey {
    device_name: DeviceUid,
    index: u16,
}

impl AwgKey {
    pub fn new(device_name: DeviceUid, index: u16) -> Self {
        AwgKey { device_name, index }
    }

    pub fn device_name(&self) -> &DeviceUid {
        &self.device_name
    }

    pub fn index(&self) -> u16 {
        self.index
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    uid: DeviceUid,
    kind: DeviceKind,
}

impl Device {
    pub fn new(uid: DeviceUid, kind: DeviceKind) -> Self {
        Device { uid, kind }
    }

    pub fn uid(&self) -> &DeviceUid {
        &self.uid
    }

    pub fn kind(&self) -> &DeviceKind {
        &self.kind
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TriggerMode {
    ZSync,
    DioTrigger,
    InternalReadyCheck,
    DioWait,
    InternalTriggerWait,
}

#[derive(Debug, Clone)]
pub struct AwgCore {
    pub uid: u16,
    pub kind: AwgKind,
    // AWG signals
    // In the case of multiplexed, signals with different UID points to the same channel(s)
    pub signals: Vec<Arc<Signal>>,
    pub sampling_rate: f64,
    pub device: Arc<Device>,
    // Mapping from HW oscillator to an assigned index
    pub osc_allocation: HashMap<String, u16>,
    pub trigger_mode: TriggerMode,
    pub is_reference_clock_internal: bool,
}

impl AwgCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        uid: u16,
        kind: AwgKind,
        signals: Vec<Arc<Signal>>,
        sampling_rate: f64,
        device: Arc<Device>,
        osc_allocation: HashMap<String, u16>,
        trigger_mode: Option<TriggerMode>,
        is_reference_clock_internal: bool,
    ) -> Self {
        AwgCore {
            uid,
            kind,
            signals,
            sampling_rate,
            device,
            osc_allocation,
            trigger_mode: trigger_mode.unwrap_or(TriggerMode::ZSync),
            is_reference_clock_internal,
        }
    }

    pub fn key(&self) -> AwgKey {
        AwgKey::new(self.device.uid.clone(), self.uid)
    }

    /// A mapping from signal UID to the index of the oscillator in the `osc_allocation` map.
    pub fn oscillator_index_by_signal_uid(&self) -> HashMap<SignalUid, u16> {
        let mut index_map = HashMap::new();
        for signal in self.signals.iter() {
            if let Some(osc) = &signal.oscillator
                && let Some(osc_index) = self.osc_allocation.get(&osc.uid)
            {
                index_map.insert(signal.uid, *osc_index);
            }
        }
        index_map
    }

    pub fn device_kind(&self) -> &DeviceKind {
        &self.device.kind
    }

    /// Use command table for pulses
    pub(crate) fn use_command_table(&self) -> bool {
        matches!(self.device_kind(), DeviceKind::SHFSG | DeviceKind::HDAWG)
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
    pub const fn traits(&self) -> &'static device_traits::DeviceTraits {
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

    pub fn is_qa_device(&self) -> bool {
        matches!(self, DeviceKind::SHFQA | DeviceKind::UHFQA)
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
                "Unsupported device type: {s}. Supported types are: SHFQA, SHFSG, HDAWG, UHFQA"
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
        self.start.map(normalize_f64).hash(state);
        self.length.map(normalize_f64).hash(state);
        self.pulse_id.hash(state);
    }
}

/// Parameter that is swept over.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepParameter {
    /// UID of the parameter
    pub uid: String,
    /// Values of the parameter
    pub values: Arc<NumericArray>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MixerType {
    /// Mixer performs full complex modulation
    IQ,
    /// Mixer only performs envelope modulation (UHFQA-style)
    UhfqaEnvelope,
}
