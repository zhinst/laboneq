// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{
    HandleUid, ParameterUid, SectionTimingMode, SectionUid, SignalUid, SweepParameter,
};
use laboneq_units::tinysample::TinySamples;
use num_integer::lcm;
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::parameter_resolver::ParameterResolver;
use crate::utils::{SignalGridInfo, compute_signal_grids};

pub(super) struct LocalContext<'a> {
    pub section_uid: Option<SectionUid>,
    resolver_stack: Vec<ParameterResolver<'a>>,
    pub system_grid: TinySamples,
    signal_grids: HashMap<SignalUid, (TinySamples, TinySamples)>,
    pub section_timing_mode: SectionTimingMode,
    /// Map from acquisition handle UIDs to signal UIDs, populated as
    /// `Acquire` operations are visited in program order.
    handle_to_signal: HashMap<HandleUid, SignalUid>,
}

impl<'a> LocalContext<'a> {
    pub(super) fn new<'b>(
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        nt_parameters: &'a crate::ParameterStore,
        system_grid: TinySamples,
        signals: impl Iterator<Item = &'b (impl SignalGridInfo + 'b)>,
    ) -> Result<Self> {
        let resolver: ParameterResolver = ParameterResolver::new(parameters, nt_parameters);
        let signal_grids = signals
            .map(|s| Ok((s.uid(), compute_signal_grids(s)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(LocalContext {
            section_uid: None,
            section_timing_mode: SectionTimingMode::Relaxed,
            resolver_stack: vec![resolver],
            system_grid,
            signal_grids,
            handle_to_signal: HashMap::new(),
        })
    }

    /// Record that `handle` is associated with `signal`, as observed at an
    /// `Acquire` operation.
    pub(super) fn set_handle_signal(&mut self, handle: HandleUid, signal: SignalUid) -> Result<()> {
        if let Some(existing_signal) = self.handle_to_signal.get(&handle)
            && existing_signal != &signal
        {
            return Err(Error::new(format!(
                "Acquisition handle '{}' is associated with multiple signals, only one allowed.",
                handle.0
            )));
        }
        self.handle_to_signal.insert(handle, signal);
        Ok(())
    }

    /// Look up the signal most recently associated with `handle` by an
    /// `Acquire` operation earlier in program order.
    pub(super) fn handle_signal(&self, handle: &HandleUid) -> Result<&SignalUid> {
        self.handle_to_signal.get(handle).ok_or_else(|| {
            Error::new(format!(
                "Handle '{}' is used in a match before being acquired.",
                handle.0
            ))
        })
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

        for signal in signals {
            let (grid, sequencer) = self.signal_grids(&signal);
            signals_grid = lcm(signals_grid, grid.value());
            sequencer_grid = lcm(sequencer_grid, sequencer.value());
        }
        // The grid is commensurate with all the contained signals' grids, that is, their
        // LCM. Signals with differing sample rates do not by themselves warrant a coarser
        // grid; only system-level features (acquisitions, triggers, sweeps) do.
        let mut grid = signals_grid;
        if on_system_grid {
            grid = lcm(grid, self.system_grid.value());
        }
        if escalate_to_sequencer_grid {
            grid = lcm(grid, sequencer_grid);
        }
        (grid.into(), sequencer_grid.into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_store::ParameterStoreBuilder;
    use laboneq_common::named_id::NamedId;
    use laboneq_units::tinysample::tiny_samples;

    /// HDAWG at 2.4 GS/s: 1500 tinysamples per sample, 24000 per sequencer cycle.
    const HD: (f64, u16) = (2.4e9, 16);
    /// UHFQA at 1.8 GS/s: 2000 tinysamples per sample, 16000 per sequencer cycle.
    const QA: (f64, u16) = (1.8e9, 8);

    struct TestSignal {
        uid: SignalUid,
        sampling_rate: f64,
        sequencer_rate: f64,
    }

    impl TestSignal {
        fn new(uid: u32, (sampling_rate, sample_multiple): (f64, u16)) -> Self {
            Self {
                uid: SignalUid(NamedId::debug_id(uid)),
                sampling_rate,
                sequencer_rate: sampling_rate / sample_multiple as f64,
            }
        }
    }

    impl SignalGridInfo for TestSignal {
        fn uid(&self) -> SignalUid {
            self.uid
        }

        fn sampling_rate(&self) -> f64 {
            self.sampling_rate
        }

        fn sequencer_rate(&self) -> f64 {
            self.sequencer_rate
        }
    }

    /// Build a context over the given signals, with the system grid set to the LCM of
    /// their sequencer grids, as `lower_experiment` does.
    fn with_context<R>(signals: &[TestSignal], f: impl FnOnce(&LocalContext) -> R) -> R {
        let system_grid = crate::utils::compute_grid(signals.iter()).unwrap().1;
        with_context_and_system_grid(signals, system_grid, f)
    }

    /// As `with_context`, but with an explicit system grid. In a real experiment the
    /// system grid is the LCM over *all* signals, so it can be a multiple of the
    /// LCM over the signals of any single section.
    fn with_context_and_system_grid<R>(
        signals: &[TestSignal],
        system_grid: TinySamples,
        f: impl FnOnce(&LocalContext) -> R,
    ) -> R {
        let parameters = HashMap::new();
        let nt_parameters = ParameterStoreBuilder::new().build();
        let ctx =
            LocalContext::new(&parameters, &nt_parameters, system_grid, signals.iter()).unwrap();
        f(&ctx)
    }

    #[test]
    fn grid_of_single_signal_is_its_sample_grid() {
        let signals = [TestSignal::new(0, HD)];
        with_context(&signals, |ctx| {
            let (grid, sequencer_grid) =
                ctx.calculate_grids(signals.iter().map(|s| s.uid), false, false);
            assert_eq!(grid, tiny_samples(1500));
            assert_eq!(sequencer_grid, tiny_samples(24000));
        });
    }

    #[test]
    fn differing_sample_grids_escalate_to_their_lcm_only() {
        // Timing rule 4: the section grid is the LCM of the contained signals' grids.
        // It must not be escalated all the way to the sequencer grid (QRL-871).
        let signals = [TestSignal::new(0, HD), TestSignal::new(1, QA)];
        with_context(&signals, |ctx| {
            let (grid, sequencer_grid) =
                ctx.calculate_grids(signals.iter().map(|s| s.uid), false, false);
            assert_eq!(grid, tiny_samples(6000)); // lcm(1500, 2000)
            assert_eq!(sequencer_grid, tiny_samples(48000)); // lcm(24000, 16000)
        });
    }

    #[test]
    fn system_level_features_escalate_to_sequencer_and_system_grid() {
        let signals = [TestSignal::new(0, HD), TestSignal::new(1, QA)];
        // We test the general case, where the system grid is coarser than the
        // signals' sequencer grids, as happens when the experiment holds
        // further signals.
        let system_grid = tiny_samples(3 * 48000);
        with_context_and_system_grid(&signals, system_grid, |ctx| {
            let uids = || signals.iter().map(|s| s.uid);

            let (grid, _) = ctx.calculate_grids(uids(), true, false);
            assert_eq!(grid, tiny_samples(48000)); // lcm(24000, 16000)

            let (grid, _) = ctx.calculate_grids(uids(), false, true);
            assert_eq!(grid, system_grid);

            let (grid, _) = ctx.calculate_grids(uids(), true, true);
            assert_eq!(grid, system_grid); // 48000 divides the system grid
        });
    }

    #[test]
    fn grid_of_no_signals_is_trivial() {
        with_context(&[], |ctx| {
            let (grid, sequencer_grid) = ctx.calculate_grids(std::iter::empty(), false, false);
            assert_eq!(grid, tiny_samples(1));
            assert_eq!(sequencer_grid, tiny_samples(1));
        });
    }
}
