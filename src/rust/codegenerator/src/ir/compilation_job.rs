// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::device_options::DeviceOptions;
use laboneq_dsl::signal_calibration::{MixerCalibration, PortMode};
use laboneq_dsl::types::Quantity;
use laboneq_error::{LabOneQError, laboneq_error};
use laboneq_units::duration::{Duration, Second};
use numeric_array::NumericArray;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use crate::device_traits;
use crate::ir::SignalUid;
use crate::result::{FixedValueOrParameter, PpcSettings, RoutedOutput};
use crate::utils::normalize_f64;

pub type Samples = i64;
pub type ChannelIndex = u8;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceUid(Arc<String>);

impl Deref for DeviceUid {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Display for DeviceUid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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

#[derive(Debug, Clone, PartialEq)]
pub struct Oscillator {
    pub uid: String,
    pub kind: OscillatorKind,
    /// Optional fixed frequency for hardware oscillators.
    pub frequency: Option<f64>,
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

    pub(crate) fn delay(&self) -> Samples {
        self.start_delay + self.signal_delay
    }

    /// Returns true if the signal is an output signal (i.e., not an integration signal).
    pub(crate) fn is_output(&self) -> bool {
        self.kind != SignalKind::INTEGRATION
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwgKind {
    /// Only one RF channel is played on HDAWG
    SINGLE,
    /// Two independent RF channels on HDAWG
    DOUBLE,
    /// HDAWG / UHFQA: Two channels form an I/Q signal
    /// SHFQA: Can contain both generator and acquisition signals
    /// SHFSG: Only generator signals
    IQ,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
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

/// Enum used to tell the code generator how the triggering scheme should be
/// configured for a single AWG.
///
/// ZSync: The AWG core is triggered via ZSync.
/// DioTrigger: Used to synchronize HDAWG cores in a standalone HDAWG or
///     HDAWG+UHFQA setups. The generated SeqC code will be such that:
///     1. The first HDAWG core emits a DIO signal and blocks until it is
///        received.
///     2. The rest of the AWG cores block until DIO trigger is received.
/// DioWait: Used exclusively to make UHFQA AWG block until DIO trigger is received.
/// InternalTriggerWait: Used for SHFQC internal triggering scheme.
/// InternalReadyCheck: Used for standalone HDAWGs.
#[derive(Clone, Copy, Debug)]
pub enum TriggerMode {
    ZSync,
    DioTrigger,
    InternalReadyCheck,
    DioWait,
    InternalTriggerWait,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AwgCore {
    pub uid: u16,
    pub kind: AwgKind,
    // AWG signals
    // In the case of multiplexed, signals with different UID points to the same channel(s)
    pub signals: Vec<Arc<Signal>>,
    pub sampling_rate: f64,
    pub device: Arc<Device>,
    pub trigger_mode: TriggerMode,
    pub options: DeviceOptions,
    oscillator_allocation: HashMap<SignalUid, u16>,
    signal_to_channel_mapping: HashMap<SignalUid, Vec<ChannelIndex>>,
    pub is_shfqc: bool,
}

impl AwgCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        uid: u16,
        signals: Vec<Arc<Signal>>,
        sampling_rate: f64,
        device: Arc<Device>,
        trigger_mode: Option<TriggerMode>,
        options: DeviceOptions,
        signal_to_channel_mapping: HashMap<SignalUid, Vec<ChannelIndex>>,
        is_shfqc: bool,
    ) -> Self {
        AwgCore {
            uid,
            kind: Self::eval_awg_kind(&device.kind, &signals),
            signals,
            sampling_rate,
            device,
            trigger_mode: trigger_mode.unwrap_or(TriggerMode::ZSync),
            options,
            signal_to_channel_mapping,
            oscillator_allocation: HashMap::new(),
            is_shfqc,
        }
    }

    fn eval_awg_kind(device_kind: &DeviceKind, signals: &[Arc<Signal>]) -> AwgKind {
        if device_kind == &DeviceKind::HDAWG {
            match &signals.first().unwrap().kind {
                SignalKind::IQ => AwgKind::IQ,
                SignalKind::SINGLE => {
                    let unique_channels = signals
                        .iter()
                        .flat_map(|s| &s.channels)
                        .collect::<HashSet<_>>();
                    if unique_channels.len() == 2 {
                        AwgKind::DOUBLE
                    } else {
                        AwgKind::SINGLE
                    }
                }
                _ => panic!("HDAWG supports only IQ or SINGLE signals."),
            }
        } else {
            AwgKind::IQ
        }
    }

    pub fn key(&self) -> AwgKey {
        AwgKey::new(self.device.uid.clone(), self.uid)
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

    pub(crate) fn add_oscillator_index(&mut self, signal_uid: SignalUid, index: u16) {
        assert!(
            !self.oscillator_allocation.contains_key(&signal_uid),
            "Signal {} already has an assigned oscillator index.",
            signal_uid.0
        );
        self.oscillator_allocation.insert(signal_uid, index);
    }

    pub(crate) fn oscillator_index(&self, signal_uid: &SignalUid) -> Option<u16> {
        self.oscillator_allocation.get(signal_uid).cloned()
    }

    /// Get the AWG channels for a given signal UID.
    ///
    /// These are not generator channels. They are the original signal channels, which can be used for output mapping.
    pub(crate) fn awg_channels_for_signal(
        &self,
        signal_uid: &SignalUid,
    ) -> Option<&Vec<ChannelIndex>> {
        self.signal_to_channel_mapping.get(signal_uid)
    }
}

pub(crate) struct AwgCoreBuilder {
    uid: u16,
    signals: Vec<Arc<Signal>>,
    sampling_rate: f64,
    device: Arc<Device>,
    trigger_mode: Option<TriggerMode>,
    options: DeviceOptions,
    signal_to_channel_mapping: HashMap<SignalUid, Vec<ChannelIndex>>,
    is_shfqc: bool,
}

impl AwgCoreBuilder {
    pub(crate) fn new(uid: u16, device: Arc<Device>, sampling_rate: f64) -> Self {
        AwgCoreBuilder {
            uid,
            signals: Vec::new(),
            sampling_rate,
            device,
            trigger_mode: None,
            options: DeviceOptions::default(),
            signal_to_channel_mapping: HashMap::new(),
            is_shfqc: false,
        }
    }

    pub(crate) fn add_signal(&mut self, signal: Arc<Signal>) -> &mut Self {
        // Store the original channel map for this AWG.
        // This is needed due to the fact that signal channels are modified during the code generation,
        // but we need the original channels for output.
        // TODO: Reverse this and do not modify the signal channels and store QA generator channels in a separate map.
        // The signal channels for QA are modified in `allocate_shfqa_generator_channels()`, which should be refactor
        // to not modify the original signals.
        self.signal_to_channel_mapping
            .insert(signal.uid, signal.channels.clone());
        self.signals.push(signal);
        self
    }

    pub(crate) fn trigger_mode(&mut self, trigger_mode: TriggerMode) -> &mut Self {
        self.trigger_mode = Some(trigger_mode);
        self
    }

    pub(crate) fn options(&mut self, options: DeviceOptions) -> &mut Self {
        self.options = options;
        self
    }

    pub(crate) fn is_shfqc(&mut self) -> &mut Self {
        self.is_shfqc = true;
        self
    }

    pub(crate) fn build(self) -> AwgCore {
        AwgCore::new(
            self.uid,
            self.signals,
            self.sampling_rate,
            self.device,
            self.trigger_mode,
            self.options,
            self.signal_to_channel_mapping,
            self.is_shfqc,
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
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

    pub fn as_str(&self) -> &'static str {
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
    type Err = LabOneQError;

    fn from_str(s: &str) -> Result<DeviceKind, Self::Err> {
        match s.to_uppercase().as_str() {
            "HDAWG" => Ok(DeviceKind::HDAWG),
            "SHFQA" => Ok(DeviceKind::SHFQA),
            "SHFSG" => Ok(DeviceKind::SHFSG),
            "UHFQA" => Ok(DeviceKind::UHFQA),
            _ => Err(laboneq_error!(
                "Unsupported device type: {s}. Supported types are: SHFQA, SHFSG, HDAWG, UHFQA"
            )),
        }
    }
}

impl TryFrom<laboneq_common::types::DeviceKind> for DeviceKind {
    type Error = LabOneQError;

    fn try_from(value: laboneq_common::types::DeviceKind) -> Result<Self, Self::Error> {
        match value {
            laboneq_common::types::DeviceKind::Hdawg => Ok(DeviceKind::HDAWG),
            laboneq_common::types::DeviceKind::Shfqa => Ok(DeviceKind::SHFQA),
            laboneq_common::types::DeviceKind::Shfsg => Ok(DeviceKind::SHFSG),
            laboneq_common::types::DeviceKind::Uhfqa => Ok(DeviceKind::UHFQA),
            _ => Err(laboneq_error!(
                "Unsupported device type: {:?}. Supported types are: SHFQA, SHFSG, HDAWG, UHFQA",
                value
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InitialSignalProperties {
    pub uid: SignalUid,
    pub amplitude: Option<FixedValueOrParameter<f64>>,
    pub thresholds: Vec<f64>,
    pub mixer_calibration: Option<MixerCalibration>,
    pub port_mode: Option<PortMode>,
    pub port_delay: Option<FixedValueOrParameter<Duration<Second>>>,
    pub ppc_settings: Option<PpcSettings>,
    pub voltage_offset: Option<FixedValueOrParameter<f64>>,
    pub range: Option<Quantity>,
    pub lo_frequency: Option<FixedValueOrParameter<f64>>,
    pub routed_outputs: Vec<RoutedOutput>,
}
