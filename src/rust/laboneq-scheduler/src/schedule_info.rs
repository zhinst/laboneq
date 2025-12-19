// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::experiment::types::{ParameterUid, SectionAlignment, SignalUid};
use laboneq_units::tinysample::{TinySamples, tiny_samples};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleInfo {
    /// Grid of the node and its children.
    /// Defaults to 1.
    pub grid: TinySamples,
    /// Grid that should be used when this node is part of a compressed loop.
    /// Defaults to 1.
    pub compressed_loop_grid: Option<TinySamples>,
    /// Sequencer grid of the node.
    /// Defaults to 1.
    pub sequencer_grid: TinySamples,
    /// Length of the node, defaults to 0.
    pub length: TinySamples,
    pub length_param: Option<ParameterUid>,
    /// Signals involved in this node.
    pub signals: HashSet<SignalUid>,
    pub alignment_mode: SectionAlignment,
    /// Repetition mode of this node. Should be only
    /// set for loop nodes.
    pub repetition_mode: Option<RepetitionMode>,
}

pub(crate) struct ScheduleInfoBuilder {
    grid: TinySamples,
    compressed_loop_grid: Option<TinySamples>,
    sequencer_grid: TinySamples,
    length: TinySamples,
    signals: HashSet<SignalUid>,
    length_param: Option<ParameterUid>,
    alignment_mode: SectionAlignment,
    repetition_mode: Option<RepetitionMode>,
}

impl ScheduleInfoBuilder {
    pub(crate) fn new() -> Self {
        Self {
            grid: tiny_samples(1),
            length: tiny_samples(0),
            signals: HashSet::new(),
            length_param: None,
            alignment_mode: SectionAlignment::Left,
            repetition_mode: None,
            compressed_loop_grid: None,
            sequencer_grid: tiny_samples(1),
        }
    }

    pub(crate) fn grid(mut self, grid: impl Into<TinySamples>) -> Self {
        self.grid = grid.into();
        self
    }

    pub(crate) fn signals(mut self, signals: HashSet<SignalUid>) -> Self {
        self.signals = signals;
        self
    }

    pub(crate) fn compressed_loop_grid(
        mut self,
        compressed_loop_grid: impl Into<TinySamples>,
    ) -> Self {
        self.compressed_loop_grid = Some(compressed_loop_grid.into());
        self
    }

    pub(crate) fn sequencer_grid(mut self, sequencer_grid: impl Into<TinySamples>) -> Self {
        self.sequencer_grid = sequencer_grid.into();
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

    pub(crate) fn alignment_mode(mut self, alignment_mode: SectionAlignment) -> Self {
        self.alignment_mode = alignment_mode;
        self
    }

    pub(crate) fn repetition_mode(mut self, mode: RepetitionMode) -> Self {
        self.repetition_mode = Some(mode);
        self
    }

    pub(crate) fn build(self) -> ScheduleInfo {
        ScheduleInfo {
            grid: self.grid,
            length: self.length,
            length_param: self.length_param,
            signals: self.signals,
            alignment_mode: self.alignment_mode,
            repetition_mode: self.repetition_mode,
            compressed_loop_grid: self.compressed_loop_grid,
            sequencer_grid: self.sequencer_grid,
        }
    }
}

impl Default for ScheduleInfoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub enum RepetitionMode {
    Fastest,
    Constant { time: TinySamples },
    Auto,
}
