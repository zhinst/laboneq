// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::operation::Operation;

pub mod device_setup;
pub mod experiment_signal;
pub mod node;
pub mod operation;
pub mod setup_description_qccs;
pub mod setup_description_zqcs;
pub mod signal_calibration;
pub mod types;

pub type ExperimentNode = node::Node<Operation>;
pub type NodeChild = node::NodeChild<Operation>;
