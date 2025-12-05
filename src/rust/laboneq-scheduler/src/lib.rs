// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod error;
pub mod experiment;
pub mod ir;
pub mod parameter_store;

mod adjust_acquire_lengths;
mod analysis;
mod chunk_experiment;
mod experiment_context;
mod ir_unroll;
mod lower_experiment;
mod parameter_resolver;
mod resolve_parameters;
mod schedule_info;
mod scheduled_node;
mod scheduler;
mod signal_info;
mod utils;

pub use crate::chunk_experiment::ChunkingInfo;
pub use crate::experiment_context::ExperimentContext;
pub use crate::parameter_store::{ParameterStore, ParameterStoreBuilder};
pub use crate::scheduler::{ScheduledExperiment, schedule_experiment};
pub use crate::signal_info::SignalInfo;
pub use schedule_info::ScheduleInfo;
pub use scheduled_node::Node as ScheduledNode;
