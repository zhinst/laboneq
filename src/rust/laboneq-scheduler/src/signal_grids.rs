// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::SignalUid;
use laboneq_units::tinysample::TinySamples;

use crate::error::Result;
use crate::utils::{SignalGridInfo, compute_grid, compute_signal_grids, lcm};

/// Time grids of the signals of an experiment.
pub(crate) struct SignalGrids {
    /// Sample grid and sequencer grid per signal.
    grids: HashMap<SignalUid, (TinySamples, TinySamples)>,
    system_grid: TinySamples,
}

impl SignalGrids {
    /// Computes the grids of `signals`, and the system grid as the LCM of their sequencer grids.
    pub(crate) fn new<'a, T: SignalGridInfo + 'a>(
        signals: impl Iterator<Item = &'a T> + Clone,
    ) -> Result<Self> {
        let (_, system_grid) = compute_grid(signals.clone())?;
        let grids = signals
            .map(|s| Ok((s.uid(), compute_signal_grids(s)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { grids, system_grid })
    }

    /// The grid commensurate with the sequencers of all signals.
    pub(crate) fn system_grid(&self) -> TinySamples {
        self.system_grid
    }

    /// Returns the sample grid and the sequencer grid of `signal`.
    pub(crate) fn of(&self, signal: &SignalUid) -> (TinySamples, TinySamples) {
        *self.grids.get(signal).expect("unknown signal")
    }

    /// Returns the LCM of the sample grids and the LCM of the sequencer grids of `signals`.
    pub(crate) fn lcm_of<'a>(
        &self,
        signals: impl IntoIterator<Item = &'a SignalUid>,
    ) -> (TinySamples, TinySamples) {
        let (grid, sequencer) = signals
            .into_iter()
            .fold((1, 1), |(grid, sequencer), signal| {
                let (g, s) = self.of(signal);
                (lcm(grid, g.value()), lcm(sequencer, s.value()))
            });
        (grid.into(), sequencer.into())
    }
}
