// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use laboneq_dsl::types::{ParameterUid, PulseUid, SectionTimingMode, SectionUid, SweepParameter};
use laboneq_ir::ParameterLookup;

use crate::error::Result;
use crate::parameter_resolver::ParameterResolver;
use crate::signal_grids::SignalGrids;

pub(super) struct LocalContext<'a> {
    pub section_uid: Option<SectionUid>,
    resolver_stack: Vec<ParameterResolver<'a>>,
    pub grids: &'a SignalGrids,
    pub section_timing_mode: SectionTimingMode,
    /// Pulses targeted by `mark_stale` anywhere in this real-time program.
    pub updated_pulses: HashSet<PulseUid>,
}

impl<'a> LocalContext<'a> {
    pub(super) fn new(
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        nt_parameters: &'a dyn ParameterLookup,
        grids: &'a SignalGrids,
    ) -> Self {
        let resolver: ParameterResolver = ParameterResolver::new(parameters, nt_parameters);
        LocalContext {
            section_uid: None,
            section_timing_mode: SectionTimingMode::Relaxed,
            resolver_stack: vec![resolver],
            grids,
            updated_pulses: HashSet::new(),
        }
    }

    pub(super) fn with_loop<R, T: FnMut(&mut Self) -> R>(
        &mut self,
        section_uid: SectionUid,
        parameters: &[ParameterUid],
        section_timing_mode: SectionTimingMode,
        mut f: T,
    ) -> Result<R> {
        let previous_section_uid = self.section_uid;
        let previous_section_timing_mode = self.section_timing_mode;
        self.section_uid = Some(section_uid);
        self.section_timing_mode = section_timing_mode;
        let resolver = self
            .resolver_stack
            .last()
            .unwrap()
            .child_scope(parameters)?;
        self.resolver_stack.push(resolver);
        let result = f(self);
        self.resolver_stack.pop();
        self.section_uid = previous_section_uid;
        self.section_timing_mode = previous_section_timing_mode;
        Ok(result)
    }

    pub(super) fn with_section<R, T: FnMut(&mut Self) -> R>(
        &mut self,
        section_uid: SectionUid,
        section_timing_mode: SectionTimingMode,
        mut f: T,
    ) -> R {
        let previous_section_uid = self.section_uid;
        let previous_section_timing_mode = self.section_timing_mode;
        self.section_uid = Some(section_uid);
        self.section_timing_mode = section_timing_mode;
        let result = f(self);
        self.section_uid = previous_section_uid;
        self.section_timing_mode = previous_section_timing_mode;
        result
    }

    pub(super) fn parameter_resolver(&self) -> &ParameterResolver<'a> {
        self.resolver_stack.last().unwrap()
    }

    pub(super) fn section_name(&self) -> String {
        self.section_uid
            .map_or_else(|| "unknown".to_string(), |s| s.0.to_string())
    }
}
