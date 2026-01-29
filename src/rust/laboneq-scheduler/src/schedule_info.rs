// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{ParameterUid, SectionAlignment, SectionUid, SignalUid};
use laboneq_units::tinysample::{TinySamples, tiny_samples};
use std::collections::HashSet;

/// Length that may be deferred (not yet determined).
type DeferredLength = Option<TinySamples>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScheduleInfo {
    /// Grid of the node and its children.
    /// Defaults to 1.
    pub grid: TinySamples,
    /// Grid that should be used when this node is part of a compressed loop.
    /// Defaults to 1.
    pub compressed_loop_grid: TinySamples,
    /// Sequencer grid of the node.
    ///
    /// The time grid along which the interval may be scheduled/shifted, commensurate
    /// with the sequencer rate.
    ///
    /// Defaults to 1.
    pub(crate) sequencer_grid: TinySamples,
    /// Length of the node.
    length: DeferredLength,
    pub length_param: Option<ParameterUid>,
    /// Signals involved in this node.
    pub signals: HashSet<SignalUid>,
    pub alignment_mode: SectionAlignment,
    /// Repetition mode of this node. Should be only
    /// set for loop nodes.
    pub repetition_mode: Option<RepetitionMode>,
    /// Whether the parent node should escalate to sequencer grid.
    pub escalate_to_sequencer_grid: bool,
    /// Sections that should be played after this one.
    pub play_after: Vec<SectionUid>,
    pub(crate) absolute_start: TinySamples,
}

impl ScheduleInfo {
    /// Get the length of the scheduled node.
    ///
    /// This will panic if the length is unresolved.
    pub(crate) fn length(&self) -> TinySamples {
        self.length
            .expect("Attempted to access unresolved length in ScheduleInfo.")
    }

    /// Try to get the length of the scheduled node.
    ///
    /// Returns None if the length is unresolved.
    pub(crate) fn try_length(&self) -> Option<TinySamples> {
        self.length
    }

    /// Resolve the length of the scheduled node.
    pub(crate) fn resolve_length(&mut self, length: TinySamples) {
        self.length = Some(length);
    }
}

pub(crate) struct ScheduleInfoBuilder {
    grid: TinySamples,
    compressed_loop_grid: TinySamples,
    sequencer_grid: TinySamples,
    length: Option<TinySamples>,
    signals: HashSet<SignalUid>,
    length_param: Option<ParameterUid>,
    alignment_mode: SectionAlignment,
    repetition_mode: Option<RepetitionMode>,
    escalate_to_sequencer_grid: bool,
    play_after: Vec<SectionUid>,
}

impl ScheduleInfoBuilder {
    pub(crate) fn new() -> Self {
        Self {
            grid: tiny_samples(1),
            length: None,
            signals: HashSet::new(),
            length_param: None,
            alignment_mode: SectionAlignment::Left,
            repetition_mode: None,
            compressed_loop_grid: tiny_samples(1),
            sequencer_grid: tiny_samples(1),
            escalate_to_sequencer_grid: false,
            play_after: Vec::new(),
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
        self.compressed_loop_grid = compressed_loop_grid.into();
        self
    }

    pub(crate) fn sequencer_grid(mut self, sequencer_grid: impl Into<TinySamples>) -> Self {
        self.sequencer_grid = sequencer_grid.into();
        self
    }

    pub(crate) fn length(mut self, length: impl Into<TinySamples>) -> Self {
        self.length = Some(length.into());
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

    pub(crate) fn escalate_to_sequencer_grid(mut self, escalate: bool) -> Self {
        self.escalate_to_sequencer_grid = escalate;
        self
    }

    pub(crate) fn play_after(mut self, play_after: Vec<SectionUid>) -> Self {
        self.play_after = play_after;
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
            escalate_to_sequencer_grid: self.escalate_to_sequencer_grid,
            play_after: self.play_after,
            absolute_start: tiny_samples(0),
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
