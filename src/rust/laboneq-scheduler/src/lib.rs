// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod error;
pub mod experiment;
pub mod ir;
pub mod parameter_store;

mod analysis;
mod chunk_ir;
mod ir_unroll;
mod lower_experiment;
mod schedule_info;
mod scheduled_node;
mod scheduler;
mod signal_info;
mod utils;

pub use crate::parameter_store::{ParameterStore, ParameterStoreBuilder};
pub use crate::scheduler::{Experiment, ScheduledExperiment, schedule_experiment};
pub use crate::signal_info::SignalInfo;

/// The smallest time unit used in the compiler.
///
/// Example conversion:
///
/// 2.0 GHz 1 sample = 1800 x TINYSAMPLE
pub type TinySample = u64;
pub use scheduled_node::Node as ScheduledNode;

/// Information about chunking of experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkingInfo {
    /// Current chunking index (0-based).
    pub index: usize,
    /// Total number of chunks.
    pub count: usize,
}
