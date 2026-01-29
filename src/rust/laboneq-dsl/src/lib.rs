// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::operation::Operation;

pub mod node;
pub mod operation;
pub mod types;

pub type ExperimentNode = node::Node<Operation>;
pub type NodeChild = node::NodeChild<Operation>;
