// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod builders;
pub mod node;
pub mod sweep_parameter;
pub mod types;

pub type ExperimentNode = node::Node<types::Operation>;
pub type NodeChild = node::NodeChild<types::Operation>;
pub type NodePtr = *const ExperimentNode;
