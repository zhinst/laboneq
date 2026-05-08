// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use crate::types::{ParameterUid, SectionAlignment, SectionTimingMode, SectionUid};

use crate::operation::Sweep;

pub struct SweepBuilder {
    sweep: Sweep,
}

impl SweepBuilder {
    pub fn new(uid: SectionUid, parameters: Vec<ParameterUid>, count: NonZeroU32) -> Self {
        Self {
            sweep: Sweep {
                uid,
                parameters,
                direct_parameters: Vec::new(),
                count,
                alignment: SectionAlignment::Left,
                reset_oscillator_phase: false,
                chunk_count: 1.try_into().unwrap(),
                auto_chunking: false,
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

    pub fn chunk_count(mut self, count: NonZeroU32) -> Self {
        self.sweep.chunk_count = count;
        self
    }

    pub fn auto_chunking(mut self) -> Self {
        self.sweep.auto_chunking = true;
        self
    }

    pub fn section_timing_mode(mut self, section_timing_mode: SectionTimingMode) -> Self {
        self.sweep.section_timing_mode = section_timing_mode;
        self
    }

    pub fn direct_parameters(mut self, direct_parameters: Vec<ParameterUid>) -> Self {
        self.sweep.direct_parameters = direct_parameters;
        self
    }

    pub fn build(self) -> Sweep {
        self.sweep
    }
}
