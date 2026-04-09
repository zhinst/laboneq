// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use crate::types::{ParameterUid, SectionAlignment, SectionTimingMode, SectionUid};

use crate::operation::{Chunking, Sweep};

pub struct SweepBuilder {
    sweep: Sweep,
}

impl SweepBuilder {
    pub fn new(uid: SectionUid, parameters: Vec<ParameterUid>, count: NonZeroU32) -> Self {
        Self {
            sweep: Sweep {
                uid,
                parameters,
                count,
                alignment: SectionAlignment::Left,
                reset_oscillator_phase: false,
                chunking: None,
                section_timing_mode: Default::default(),
            },
        }
    }

    pub fn alignment(mut self, alignment: SectionAlignment) -> Self {
        self.sweep.alignment = alignment;
        self
    }

    pub fn reset_oscillator_phase(mut self) -> Self {
        self.sweep.reset_oscillator_phase = true;
        self
    }

    pub fn chunking(mut self, chunking: Chunking) -> Self {
        self.sweep.chunking = Some(chunking);
        self
    }

    pub fn section_timing_mode(mut self, section_timing_mode: SectionTimingMode) -> Self {
        self.sweep.section_timing_mode = section_timing_mode;
        self
    }

    pub fn build(self) -> Sweep {
        self.sweep
    }
}
