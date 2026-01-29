// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{ParameterUid, SectionUid, SignalUid, SweepParameter};
use laboneq_units::tinysample::TinySamples;
use num_integer::lcm;
use std::collections::HashMap;

use crate::error::Result;
use crate::parameter_resolver::ParameterResolver;
use crate::utils::{SignalGridInfo, compute_signal_grids};

pub(super) struct LocalContext<'a> {
    pub section_uid: Option<SectionUid>,
    resolver_stack: Vec<ParameterResolver<'a>>,
    pub system_grid: TinySamples,
    signal_grids: HashMap<SignalUid, (TinySamples, TinySamples)>,
}

impl<'a> LocalContext<'a> {
    pub(super) fn new<'b>(
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        nt_parameters: &'a crate::ParameterStore,
        system_grid: TinySamples,
        signals: impl Iterator<Item = &'b (impl SignalGridInfo + 'b)>,
    ) -> Self {
        let resolver: ParameterResolver = ParameterResolver::new(parameters, nt_parameters);
        LocalContext {
            section_uid: None,
            resolver_stack: vec![resolver],
            system_grid,
            signal_grids: signals
                .map(|s| {
                    let (signal_grid, sequencer_grid) = compute_signal_grids(s);
                    (s.uid(), (signal_grid, sequencer_grid))
                })
                .collect(),
        }
    }

    pub(super) fn signal_grids(&self, signal: &SignalUid) -> (TinySamples, TinySamples) {
        self.signal_grids[signal]
    }

    pub(crate) fn calculate_grids(
        &self,
        signals: impl Iterator<Item = SignalUid>,
        escalate_to_sequencer_grid: bool,
        on_system_grid: bool,
    ) -> (TinySamples, TinySamples) {
        let mut signals_grid = 1;
        let mut sequencer_grid = 1;
        let mut multiple_grids = false;

        for signal in signals {
            let (grid, sequencer) = self.signal_grids(&signal);
            if !multiple_grids && signals_grid != 1 && signals_grid != grid.value() {
                multiple_grids = true;
            }
            signals_grid = lcm(signals_grid, grid.value());
            sequencer_grid = lcm(sequencer_grid, sequencer.value());
        }
        let mut grid = 1;
        if on_system_grid {
            grid = lcm(grid, self.system_grid.value());
        }
        if multiple_grids || escalate_to_sequencer_grid {
            // two different sample rates -> escalate to sequencer grid
            grid = lcm(grid, sequencer_grid);
        } else {
            grid = lcm(grid, signals_grid);
        }
        (grid.into(), sequencer_grid.into())
    }

    pub(super) fn with_loop<R, T: FnMut(&mut Self) -> R>(
        &mut self,
        section_uid: SectionUid,
        parameters: &[ParameterUid],
        mut f: T,
    ) -> Result<R> {
        let previous_section_uid = self.section_uid;
        self.section_uid = Some(section_uid);
        let resolver = self
            .resolver_stack
            .last()
            .unwrap()
            .child_scope(parameters)?;
        self.resolver_stack.push(resolver);
        let result = f(self);
        self.resolver_stack.pop();
        self.section_uid = previous_section_uid;
        Ok(result)
    }

    pub(super) fn with_section<R, T: FnMut(&mut Self) -> R>(
        &mut self,
        section_uid: SectionUid,
        mut f: T,
    ) -> R {
        let previous_section_uid = self.section_uid;
        self.section_uid = Some(section_uid);
        let result = f(self);
        self.section_uid = previous_section_uid;
        result
    }

    pub(super) fn parameter_resolver(&self) -> &ParameterResolver<'a> {
        self.resolver_stack.last().unwrap()
    }
}
