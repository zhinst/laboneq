// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod error;
pub mod ir;
mod node;
mod scheduler;
mod signal_info;

pub use crate::scheduler::{Experiment, ScheduledExperiment, schedule_experiment};
pub use crate::signal_info::SignalInfo;

use ir::IrVariant;
use node::Node;
pub type IrNode = Node<IrVariant>;
