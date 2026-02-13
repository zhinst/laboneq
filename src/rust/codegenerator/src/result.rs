// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use indexmap::IndexMap;

pub use crate::handle_feedback_registers::Acquisition;
use crate::ir::SignalUid;
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
    pub integration_unit_allocations: Vec<IntegrationUnitAllocation>,
}

pub struct AwgCodeGenerationResult<T: SampleWaveforms> {
    pub seqc: String,
    pub wave_indices: IndexMap<String, (WaveIndex, SignalType)>,
    pub command_table: Option<String>,
    pub shf_sweeper_config: Option<ShfPpcSweepJson>,
    pub sampled_waveforms: Vec<SampledWaveform<T::Signature>>,
    pub integration_weights: Vec<T::IntegrationWeight>,
    pub signal_delays: HashMap<SignalUid, f64>,
    pub integration_lengths: HashMap<SignalUid, SignalIntegrationInfo>,
    pub parameter_phase_increment_map: Option<HashMap<String, Vec<ParameterPhaseIncrement>>>,
    pub feedback_register_config: FeedbackRegisterConfig,
    pub channel_properties: Vec<ChannelProperties>,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntegrationUnitAllocation {
    pub signal: SignalUid,
    pub channels: Vec<u8>,
    pub kernel_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MarkerMode {
    Trigger,
    Marker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelProperties {
    pub channel: ChannelIndex,
    pub marker_mode: Option<MarkerMode>,
}
