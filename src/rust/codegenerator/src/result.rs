// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZero;
use std::{collections::HashMap, sync::Arc};

use indexmap::IndexMap;
use laboneq_common::device_options::DeviceOptions;
use laboneq_dsl::types::ParameterUid;

pub use crate::handle_feedback_registers::Acquisition;
use crate::ir::SignalUid;
use crate::ir::compilation_job::AwgKey;
use crate::ir::compilation_job::AwgKind;
use crate::ir::compilation_job::ChannelIndex;
use crate::ir::compilation_job::DeviceUid;
pub use crate::sample_waveforms::SampledWaveform;
pub use crate::sampled_event_handler::ParameterPhaseIncrement;
pub use crate::sampled_event_handler::SHFPPCSweeperConfig;
pub use crate::sampled_event_handler::seqc_tracker::wave_index_tracker::SignalType;
pub use crate::sampled_event_handler::seqc_tracker::wave_index_tracker::WaveIndex;

use crate::{
    ir::{PpcDevice, Samples},
    sample_waveforms::SampleWaveforms,
};

pub struct SeqCGenOutput<T: SampleWaveforms> {
    pub awg_results: Vec<AwgCodeGenerationResult<T>>,
    pub total_execution_time: f64,
    pub result_handle_maps: HashMap<ResultSource, Vec<Vec<String>>>,
    pub measurements: Vec<Measurement>,
}

pub struct AwgCodeGenerationResult<T: SampleWaveforms> {
    pub awg: AwgProperties,
    pub seqc: SeqCProgram,
    pub wave_indices: IndexMap<String, (WaveIndex, SignalType)>,
    pub command_table: Option<CommandTable>,
    pub shf_sweeper_config: Option<ShfPpcSweepJson>,
    pub sampled_waveforms: Vec<SampledWaveform<T::Signature>>,
    pub integration_kernels: Vec<T::SampledIntegrationKernel>,
    pub signal_delays: HashMap<SignalUid, f64>,
    pub integration_lengths: HashMap<SignalUid, SignalIntegrationInfo>,
    pub feedback_register_config: FeedbackRegisterConfig,
    pub output_channel_properties: Vec<ChannelProperties>,
    pub input_channel_properties: Vec<InputChannelProperties>,
    pub integration_weights: Vec<IntegrationWeight>,
    pub integrator_allocations: Vec<IntegratorAllocation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SequencerType {
    Qa,
    Sg,
    Auto,
}

impl std::fmt::Display for SequencerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequencerType::Qa => write!(f, "qa"),
            SequencerType::Sg => write!(f, "sg"),
            SequencerType::Auto => write!(f, "auto"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeqCProgram {
    pub src: String,
    pub dev_type: String,
    pub dev_opts: Vec<String>,
    pub awg_index: u16,
    pub sequencer: SequencerType,
    pub sampling_rate: Option<f64>,
}

pub struct CommandTable {
    pub src: String,
    pub n_entries: usize,
    pub max_entries: usize,
    pub parameter_phase_increment_map: HashMap<String, Vec<ParameterPhaseIncrement>>,
}

impl CommandTable {
    pub fn resource_usage_percentage(&self) -> f64 {
        self.n_entries as f64 / self.max_entries as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AwgProperties {
    pub key: AwgKey,
    pub kind: AwgKind,
    pub sampling_rate: f64,
    pub options: DeviceOptions,
}

#[derive(Debug, Clone)]
pub struct ShfPpcSweepJson {
    pub ppc_device: Arc<PpcDevice>,
    pub json: String,
}

#[derive(Debug, Clone)]
pub struct FeedbackRegisterConfig {
    pub local: bool,
    // Receiver (SG instruments)
    pub source_feedback_register: Option<i64>,
    pub register_index_select: Option<u8>,
    pub codeword_bitshift: Option<u8>,
    pub codeword_bitmask: Option<u16>,
    pub command_table_offset: Option<u32>,
    // Transmitter (QA instruments)
    pub target_feedback_register: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct SignalIntegrationInfo {
    pub is_play: bool,
    pub length: Samples,
}

#[derive(Debug, Clone)]
pub struct Measurement {
    pub device: DeviceUid,
    pub channel: u16,
    pub length: Samples,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResultSource {
    pub device_id: String,
    pub awg_id: u16,
    pub integrator_idx: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegratorAllocation {
    pub signal: SignalUid,
    pub integration_units: Vec<ChannelIndex>,
    pub kernel_count: NonZero<ChannelIndex>,
    pub thresholds: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MarkerMode {
    Trigger,
    Marker,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelProperties {
    pub signal: SignalUid,
    pub channel: ChannelIndex,
    pub marker_mode: Option<MarkerMode>,
    pub hw_oscillator_index: Option<u16>,
    // Near-time sweep values.
    // Controller accepts either a fixed value or a parameter.
    pub amplitude: Option<FixedValueOrParameter<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InputChannelProperties {
    pub signal: SignalUid,
    pub channel: ChannelIndex,
    pub hw_oscillator_index: Option<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FixedValueOrParameter<T> {
    Value(T),
    Parameter(ParameterUid),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrationWeight {
    pub integration_units: Vec<ChannelIndex>,
    pub basename: String,
    pub downsampling_factor: u8,
}
