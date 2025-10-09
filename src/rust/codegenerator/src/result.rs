// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use indexmap::IndexMap;
use serde_json::Value;

pub use crate::handle_feedback_registers::Acquisition;
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
    pub simultaneous_acquires: Vec<Vec<Acquisition>>,
}

pub struct AwgCodeGenerationResult<T: SampleWaveforms> {
    pub seqc: String,
    pub wave_indices: IndexMap<String, (WaveIndex, SignalType)>,
    pub command_table: Option<Value>,
    pub shf_sweeper_config: Option<SHFPPCSweeperConfig>,
    pub sampled_waveforms: Vec<SampledWaveform<T::Signature>>,
    pub integration_weights: Vec<T::IntegrationWeight>,
    pub signal_delays: HashMap<String, f64>,
    pub ppc_device: Option<Arc<PpcDevice>>,
    pub integration_lengths: HashMap<String, SignalIntegrationInfo>,
    pub parameter_phase_increment_map: Option<HashMap<String, Vec<ParameterPhaseIncrement>>>,
    pub feedback_register_config: FeedbackRegisterConfig,
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
