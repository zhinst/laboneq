// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::experiment::types::SignalUid;
use laboneq_units::tinysample::{TinySamples, tiny_samples};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleInfo {
    /// Grid of the node and its children.
    /// Defaults to 1.
    pub grid: TinySamples,
    /// Length of the node, defaults to 0.
    pub length: TinySamples,
    /// Signals involved in this node.
    pub signals: HashSet<SignalUid>,
}

pub struct ScheduleInfoBuilder {
    grid: TinySamples,
    length: TinySamples,
    signals: HashSet<SignalUid>,
}

impl ScheduleInfoBuilder {
    pub fn new() -> Self {
        Self {
            grid: tiny_samples(1),
            length: tiny_samples(0),
            signals: HashSet::new(),
        }
    }

    pub fn grid(mut self, grid: impl Into<TinySamples>) -> Self {
        self.grid = grid.into();
        self
    }

    pub fn length(mut self, length: impl Into<TinySamples>) -> Self {
        self.length = length.into();
        self
    }

    pub fn build(self) -> ScheduleInfo {
        ScheduleInfo {
            grid: self.grid,
            length: self.length,
            signals: self.signals,
        }
    }
}

impl Default for ScheduleInfoBuilder {
    fn default() -> Self {
        Self::new()
    }
}
