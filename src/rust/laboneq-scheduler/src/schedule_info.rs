// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::experiment::types::{ParameterUid, SignalUid};
use laboneq_units::tinysample::{TinySamples, tiny_samples};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleInfo {
    /// Grid of the node and its children.
    /// Defaults to 1.
    pub grid: TinySamples,
    /// Length of the node, defaults to 0.
    pub length: TinySamples,
    pub length_param: Option<ParameterUid>,
    /// Signals involved in this node.
    pub signals: HashSet<SignalUid>,
}

pub(crate) struct ScheduleInfoBuilder {
    grid: TinySamples,
    length: TinySamples,
    signals: HashSet<SignalUid>,
    length_param: Option<ParameterUid>,
}

impl ScheduleInfoBuilder {
    pub(crate) fn new() -> Self {
        Self {
            grid: tiny_samples(1),
            length: tiny_samples(0),
            signals: HashSet::new(),
            length_param: None,
        }
    }

    pub(crate) fn grid(mut self, grid: impl Into<TinySamples>) -> Self {
        self.grid = grid.into();
        self
    }

    pub(crate) fn length(mut self, length: impl Into<TinySamples>) -> Self {
        self.length = length.into();
        self
    }

    pub(crate) fn length_param(mut self, length_param: ParameterUid) -> Self {
        self.length_param = Some(length_param);
        self
    }

    pub(crate) fn build(self) -> ScheduleInfo {
        ScheduleInfo {
            grid: self.grid,
            length: self.length,
            length_param: self.length_param,
            signals: self.signals,
        }
    }
}

impl Default for ScheduleInfoBuilder {
    fn default() -> Self {
        Self::new()
    }
}
