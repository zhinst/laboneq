// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod awg_events;
mod command_table_tracker;
mod feedback_register_config;
mod feedback_register_layout;
mod handler;
pub mod seqc_tracker;
mod shfppc_sweeper_config;
mod shfppc_sweeper_config_tracker;

use indexmap::IndexMap;
use seqc_tracker::wave_index_tracker::WaveIndex;
use seqc_tracker::{FeedbackRegisterIndex, wave_index_tracker::SignalType};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub use awg_events::AwgEvent;
pub use command_table_tracker::ParameterPhaseIncrement;
pub use feedback_register_config::FeedbackRegisterConfig;
pub use feedback_register_layout::{
    FeedbackRegister, FeedbackRegisterLayout, SingleFeedbackRegisterLayoutItem,
};
pub use handler::handle_sampled_events;
pub use shfppc_sweeper_config::SHFPPCSweeperConfig;

pub type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

pub type Samples = i64;

pub struct SeqcResults {
    pub seqc: String,
    pub wave_indices: IndexMap<String, (WaveIndex, SignalType)>,
    pub command_table: Option<Value>,
    pub parameter_phase_increment_map: Option<HashMap<String, Vec<ParameterPhaseIncrement>>>,
    pub shf_sweeper_config: Option<SHFPPCSweeperConfig>,
    pub feedback_register_config: FeedbackRegisterConfig,
}

pub type AwgEventList = BTreeMap<Samples, Vec<AwgEvent>>;
