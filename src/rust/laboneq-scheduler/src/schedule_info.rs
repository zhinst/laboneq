// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_dsl::hqcs::IsLiveTarget;
use laboneq_dsl::operation::MarkStaleTarget;
use laboneq_dsl::types::{
    ParameterUid, PulseUid, SectionAlignment, SectionTimingMode, SectionUid, SignalUid, VariableUid,
};
use laboneq_units::tinysample::{TinySamples, tiny_samples};

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
    /// Length of the node.
    length: DeferredLength,
    pub length_param: Option<ParameterUid>,
    /// Signals involved in this node.
    pub signals: HashSet<SignalUid>,
    /// Variable accesses by this node and its children.
    pub accesses: HashMap<CoprocessorValue, Access>,
    pub alignment_mode: SectionAlignment,
    /// Repetition mode of this node. Should be only
    /// set for loop nodes.
    pub repetition_mode: Option<RepetitionMode>,
    /// Whether a section is aligned to the system grid.
    pub on_system_grid: bool,
    /// Whether each iteration of a loop resets the oscillator phase.
    pub reset_oscillator_phase: bool,
    /// Sections that should be played after this one.
    pub play_after: Vec<SectionUid>,
    pub(crate) absolute_start: TinySamples,
    /// The timing mode of the section, controlling whether rounding is allowed or rejected.
    pub section_timing_mode: SectionTimingMode,
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

    /// Merges accesses, with writes taking precedence over reads.
    pub(crate) fn merge_accesses(&mut self, accesses: &HashMap<CoprocessorValue, Access>) {
        for (&value, &access) in accesses {
            let entry = self.accesses.entry(value).or_insert(access);
            *entry = (*entry).max(access);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CoprocessorValue {
    Variable(VariableUid),
    Pulse(PulseUid),
}

impl From<MarkStaleTarget> for CoprocessorValue {
    fn from(target: MarkStaleTarget) -> Self {
        match target {
            MarkStaleTarget::Variable(v) => Self::Variable(v),
            MarkStaleTarget::Pulse(p) => Self::Pulse(p),
        }
    }
}

impl From<IsLiveTarget> for CoprocessorValue {
    fn from(target: IsLiveTarget) -> Self {
        match target {
            IsLiveTarget::Variable(v) => Self::Variable(v),
            IsLiveTarget::Pulse(p) => Self::Pulse(p),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Access {
    Read = 0,
    Write = 1,
}

impl Default for ScheduleInfo {
    fn default() -> Self {
        Self {
            grid: tiny_samples(1),
            compressed_loop_grid: tiny_samples(1),
            length: None,
            signals: HashSet::new(),
            accesses: HashMap::new(),
            length_param: None,
            alignment_mode: SectionAlignment::Left,
            repetition_mode: None,
            on_system_grid: false,
            reset_oscillator_phase: false,
            play_after: Vec::new(),
            section_timing_mode: SectionTimingMode::Relaxed,
            absolute_start: tiny_samples(0),
        }
    }
}

pub(crate) struct ScheduleInfoBuilder {
    inner: ScheduleInfo,
}

impl ScheduleInfoBuilder {
    pub(crate) fn new() -> Self {
        Self {
            inner: ScheduleInfo::default(),
        }
    }

    pub(crate) fn grid(mut self, grid: impl Into<TinySamples>) -> Self {
        self.inner.grid = grid.into();
        self
    }

    pub(crate) fn signals(mut self, signals: HashSet<SignalUid>) -> Self {
        self.inner.signals = signals;
        self
    }

    pub(crate) fn access(mut self, value: impl Into<CoprocessorValue>, access: Access) -> Self {
        self.inner.accesses.insert(value.into(), access);
        self
    }

    pub(crate) fn length(mut self, length: impl Into<TinySamples>) -> Self {
        self.inner.length = Some(length.into());
        self
    }

    pub(crate) fn length_param(mut self, length_param: ParameterUid) -> Self {
        self.inner.length_param = Some(length_param);
        self
    }

    pub(crate) fn alignment_mode(mut self, alignment_mode: SectionAlignment) -> Self {
        self.inner.alignment_mode = alignment_mode;
        self
    }

    pub(crate) fn section_timing_mode(mut self, section_timing_mode: SectionTimingMode) -> Self {
        self.inner.section_timing_mode = section_timing_mode;
        self
    }

    pub(crate) fn repetition_mode(mut self, mode: RepetitionMode) -> Self {
        self.inner.repetition_mode = Some(mode);
        self
    }

    pub(crate) fn on_system_grid(mut self, on_system_grid: bool) -> Self {
        self.inner.on_system_grid = on_system_grid;
        self
    }

    pub(crate) fn reset_oscillator_phase(mut self, reset_oscillator_phase: bool) -> Self {
        self.inner.reset_oscillator_phase = reset_oscillator_phase;
        self
    }

    pub(crate) fn play_after(mut self, play_after: Vec<SectionUid>) -> Self {
        self.inner.play_after = play_after;
        self
    }

    pub(crate) fn build(self) -> ScheduleInfo {
        self.inner
    }
}

impl Default for ScheduleInfoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub(crate) enum RepetitionMode {
    Fastest,
    Constant { time: TinySamples },
    Auto,
}
