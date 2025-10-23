// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::TinySample;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScheduleInfo {
    /// Grid of the node and its children.
    /// Defaults to 1.
    pub grid: TinySample,
    /// Length of the node, defaults to 0.
    pub length: TinySample,
}

pub struct ScheduleInfoBuilder {
    grid: TinySample,
    length: TinySample,
}

impl ScheduleInfoBuilder {
    pub fn new() -> Self {
        Self { grid: 1, length: 0 }
    }

    pub fn grid(mut self, grid: TinySample) -> Self {
        self.grid = grid;
        self
    }

    pub fn length(mut self, length: TinySample) -> Self {
        self.length = length;
        self
    }

    pub fn build(self) -> ScheduleInfo {
        ScheduleInfo {
            grid: self.grid,
            length: self.length,
        }
    }
}

impl Default for ScheduleInfoBuilder {
    fn default() -> Self {
        Self::new()
    }
}
