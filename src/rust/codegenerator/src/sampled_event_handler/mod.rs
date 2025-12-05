// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod awg_events;
mod command_table_tracker;
mod feedback_register_config;
mod feedback_register_layout;
mod handler;
pub(crate) mod seqc_tracker;
pub(super) mod shfppc_sweeper_config;
mod shfppc_sweeper_config_tracker;

use indexmap::IndexMap;
use seqc_tracker::wave_index_tracker::WaveIndex;
use seqc_tracker::{FeedbackRegisterIndex, wave_index_tracker::SignalType};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub(crate) use awg_events::AwgEvent;
pub use command_table_tracker::ParameterPhaseIncrement;
pub use feedback_register_config::FeedbackRegisterConfig;
pub use feedback_register_layout::{
    FeedbackRegister, FeedbackRegisterLayout, SingleFeedbackRegisterLayoutItem,
};
pub(crate) use handler::handle_sampled_events;
pub use shfppc_sweeper_config::SHFPPCSweeperConfig;

pub(crate) type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

pub(crate) type Samples = i64;

pub(crate) struct SeqcResults {
    pub seqc: String,
    pub wave_indices: IndexMap<String, (WaveIndex, SignalType)>,
    pub command_table: Option<Value>,
    pub parameter_phase_increment_map: Option<HashMap<String, Vec<ParameterPhaseIncrement>>>,
    pub shf_sweeper_config: Option<String>,
    pub feedback_register_config: FeedbackRegisterConfig,
}

pub(crate) type AwgEventList = BTreeMap<Samples, Vec<AwgEvent>>;
