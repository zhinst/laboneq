// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{HandleUid, SignalUid};
use laboneq_ir::{IrKind, MatchTarget};
use laboneq_units::tinysample::{TinySamples, tiny_samples};

use crate::ScheduledNode;
use crate::signal_grids::SignalGrids;
use crate::utils::lcm;

/// Grid size for local feedback
const LOCAL_FEEDBACK_GRID_SAMPLES: i64 = 8;

/// Grid size for global feedback
const GLOBAL_FEEDBACK_GRID_SAMPLES: i64 = 200;

/// Writes the grid and the compressed loop grid of every node that can hold children.
///
/// A node's grid is the LCM of its own grid and its children's grids. A node with an
/// acquisition among its children also takes the sequencer grid of its signals.
/// Leaves keep the grids they were built with.
///
/// Expects the signals of every node to be collected, and the acquisitions of each
/// matched handle to precede the match.
pub(crate) fn compute_grids(root: &mut ScheduledNode, grids: &SignalGrids) {
    let mut acquired = HashMap::new();
    visit(root, grids, &mut acquired);
}

/// Computes the grids of `node` and its descendants, recording the signal of each
/// acquisition in program order.
fn visit(
    node: &mut ScheduledNode,
    grids: &SignalGrids,
    acquired: &mut HashMap<HandleUid, SignalUid>,
) {
    if let IrKind::Acquire(obj) = &node.kind {
        acquired.insert(obj.handle, obj.signal);
    }
    if !node.holds_children() {
        return;
    }
    let mut grid = own_grid(node, grids).value();
    let mut compressed_loop_grid = own_compressed_loop_grid(&node.kind, grids, acquired).value();
    let mut has_acquire = false;
    for child in node.children.iter_mut() {
        visit(&mut child.node, grids, acquired);
        grid = lcm(grid, child.node.schedule.grid.value());
        compressed_loop_grid = lcm(
            compressed_loop_grid,
            child.node.schedule.compressed_loop_grid.value(),
        );
        has_acquire |= matches!(child.node.kind, IrKind::Acquire(_));
    }
    if has_acquire {
        grid = lcm(grid, grids.lcm_of(&node.schedule.signals).1.value());
    }
    node.schedule.grid = grid.into();
    node.schedule.compressed_loop_grid = compressed_loop_grid.into();
}

/// Returns the grid that `node` requires by itself, before its children.
fn own_grid(node: &ScheduledNode, grids: &SignalGrids) -> TinySamples {
    let system_grid = grids.system_grid().value();
    let (signals_grid, sequencer_grid) = grids.lcm_of(&node.schedule.signals);
    let (signals_grid, sequencer_grid) = (signals_grid.value(), sequencer_grid.value());
    let grid = match &node.kind {
        IrKind::Root | IrKind::Loop(_) | IrKind::LoopIterationPreamble => 1,
        IrKind::Section(section) => {
            let mut grid = signals_grid;
            if !section.triggers.is_empty() {
                grid = lcm(grid, sequencer_grid);
            }
            if node.schedule.on_system_grid {
                grid = lcm(grid, system_grid);
            }
            grid
        }
        // TODO: We might want to relax this in the future.
        IrKind::LoopIteration => lcm(signals_grid, system_grid),
        IrKind::Match(obj) => match obj.target {
            MatchTarget::SweepParameter(_) => signals_grid,
            _ => system_grid,
        },
        IrKind::Case(_) => signals_grid,
        _ => unreachable!("Internal error: Node holds no children"),
    };
    tiny_samples(grid)
}

/// Returns the compressed loop grid that a node of `kind` requires by itself.
///
/// A match on a handle needs the feedback grid of the signal that acquired the handle.
fn own_compressed_loop_grid(
    kind: &IrKind,
    grids: &SignalGrids,
    acquired: &HashMap<HandleUid, SignalUid>,
) -> TinySamples {
    if let IrKind::Match(obj) = kind
        && let MatchTarget::Handle(handle) = &obj.target
        && let Some(signal) = acquired.get(handle)
    {
        // TODO (PW) is this correct? should it not be 100 ns regardless of the sampling rate?
        let samples = if obj.local {
            LOCAL_FEEDBACK_GRID_SAMPLES
        } else {
            GLOBAL_FEEDBACK_GRID_SAMPLES
        };
        tiny_samples(samples * grids.of(signal).0.value())
    } else {
        tiny_samples(1)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use laboneq_dsl::types::AcquisitionType;
    use laboneq_ir::builders::SectionBuilder;
    use laboneq_ir::{Acquire, Match};

    use super::*;
    use crate::schedule_info::ScheduleInfoBuilder;
    use crate::utils::SignalGridInfo;

    /// HDAWG at 2.4 GS/s: 1500 tinysamples per sample, 24000 per sequencer cycle.
    const HD: (f64, u16) = (2.4e9, 16);
    /// UHFQA at 1.8 GS/s: 2000 tinysamples per sample, 16000 per sequencer cycle.
    const QA: (f64, u16) = (1.8e9, 8);
    /// SHFSG at 2.0 GS/s: 1800 tinysamples per sample, 28800 per sequencer cycle.
    const SG: (f64, u16) = (2.0e9, 16);

    struct TestSignal {
        uid: SignalUid,
        sampling_rate: f64,
        sequencer_rate: f64,
    }

    impl TestSignal {
        fn new(uid: u32, (sampling_rate, sample_multiple): (f64, u16)) -> Self {
            Self {
                uid: uid.into(),
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

    /// Builds a section with a delay on each of `delays` and a trigger on each of `triggers`,
    /// with its signals collected.
    fn section(
        delays: &[&TestSignal],
        triggers: &[&TestSignal],
        on_system_grid: bool,
        grids: &SignalGrids,
    ) -> ScheduledNode {
        let mut kind = SectionBuilder::new(100.into());
        for signal in triggers {
            kind = kind.add_trigger(signal.uid, 1);
        }
        let mut section = ScheduledNode::new(
            IrKind::Section(kind.build()),
            ScheduleInfoBuilder::new()
                .on_system_grid(on_system_grid)
                .signals(delays.iter().map(|signal| signal.uid).collect())
                .build(),
        );
        for signal in delays {
            section.add_child(
                tiny_samples(0),
                ScheduledNode::new(
                    IrKind::Delay { signal: signal.uid },
                    ScheduleInfoBuilder::new()
                        .grid(grids.of(&signal.uid).0)
                        .length(0)
                        .build(),
                ),
            );
        }
        section
    }

    fn grid_of(mut node: ScheduledNode, grids: &SignalGrids) -> TinySamples {
        compute_grids(&mut node, grids);
        node.schedule.grid
    }

    #[test]
    fn grid_of_single_signal_is_its_sample_grid() {
        let hd = TestSignal::new(0, HD);
        let grids = SignalGrids::new([&hd].into_iter()).unwrap();
        let section = section(&[&hd], &[], false, &grids);
        assert_eq!(grid_of(section, &grids), tiny_samples(1500));
    }

    #[test]
    fn differing_sample_grids_escalate_to_their_lcm_only() {
        // Timing rule 4: the section grid is the LCM of the contained signals' grids.
        // It must not be escalated all the way to the sequencer grid (QRL-871).
        let (hd, qa) = (TestSignal::new(0, HD), TestSignal::new(1, QA));
        let grids = SignalGrids::new([&hd, &qa].into_iter()).unwrap();
        let section = section(&[&hd, &qa], &[], false, &grids);
        assert_eq!(grid_of(section, &grids), tiny_samples(6000)); // lcm(1500, 2000)
    }

    #[test]
    fn system_level_features_escalate_to_sequencer_and_system_grid() {
        // We test the general case, where the system grid is coarser than the
        // signals' sequencer grids, as happens when the experiment holds
        // further signals.
        let (hd, qa, sg) = (
            TestSignal::new(0, HD),
            TestSignal::new(1, QA),
            TestSignal::new(2, SG),
        );
        let grids = SignalGrids::new([&hd, &qa, &sg].into_iter()).unwrap();
        let system_grid = tiny_samples(3 * 48000); // lcm(24000, 16000, 28800)
        assert_eq!(grids.system_grid(), system_grid);

        let triggered = section(&[&hd], &[&qa], false, &grids);
        assert_eq!(grid_of(triggered, &grids), tiny_samples(48000)); // lcm(24000, 16000)

        let on_system_grid = section(&[&hd, &qa], &[], true, &grids);
        assert_eq!(grid_of(on_system_grid, &grids), system_grid);

        let both = section(&[&hd], &[&qa], true, &grids);
        assert_eq!(grid_of(both, &grids), system_grid); // 48000 divides the system grid
    }

    #[test]
    fn grid_of_no_signals_is_trivial() {
        let no_signals: Vec<TestSignal> = vec![];
        let grids = SignalGrids::new(no_signals.iter()).unwrap();
        let section = section(&[], &[], false, &grids);
        assert_eq!(grid_of(section, &grids), tiny_samples(1));
    }

    #[test]
    fn acquisition_escalates_its_parent_to_the_sequencer_grid() {
        let qa = TestSignal::new(0, QA);
        let grids = SignalGrids::new([&qa].into_iter()).unwrap();
        let acquire = ScheduledNode::new(
            IrKind::Acquire(Acquire {
                signal: qa.uid,
                handle: 1.into(),
                integration_length: tiny_samples(0),
                kernels: vec![],
                parameters: vec![],
                pulse_parameters: vec![],
                acquisition_type: AcquisitionType::Integration,
            }),
            ScheduleInfoBuilder::new()
                .grid(grids.of(&qa.uid).0)
                .length(0)
                .build(),
        );
        let mut section = section(&[], &[], false, &grids);
        section.add_child(tiny_samples(0), acquire);
        section.schedule.signals.insert(qa.uid);
        assert_eq!(grid_of(section, &grids), tiny_samples(16000));
    }

    #[test]
    fn match_on_handle_takes_the_feedback_grid_of_the_acquiring_signal() {
        let qa = TestSignal::new(0, QA);
        let grids = SignalGrids::new([&qa].into_iter()).unwrap();
        let mut root = ScheduledNode::new(IrKind::Root, ScheduleInfoBuilder::new().build());
        let mut acquiring = section(&[], &[], false, &grids);
        acquiring.add_child(
            tiny_samples(0),
            ScheduledNode::new(
                IrKind::Acquire(Acquire {
                    signal: qa.uid,
                    handle: 1.into(),
                    integration_length: tiny_samples(0),
                    kernels: vec![],
                    parameters: vec![],
                    pulse_parameters: vec![],
                    acquisition_type: AcquisitionType::Integration,
                }),
                ScheduleInfoBuilder::new().length(0).build(),
            ),
        );
        root.add_child(tiny_samples(0), acquiring);
        for local in [true, false] {
            let match_ = ScheduledNode::new(
                IrKind::Match(Match {
                    uid: 2.into(),
                    target: MatchTarget::Handle(1.into()),
                    local,
                }),
                ScheduleInfoBuilder::new()
                    .signals(HashSet::from([qa.uid]))
                    .build(),
            );
            root.add_child(tiny_samples(0), match_);
        }

        compute_grids(&mut root, &grids);
        let compressed_loop_grids: Vec<_> = root.children[1..]
            .iter()
            .map(|child| child.node.schedule.compressed_loop_grid)
            .collect();
        assert_eq!(
            compressed_loop_grids,
            [tiny_samples(8 * 2000), tiny_samples(200 * 2000)]
        );
        assert_eq!(root.schedule.compressed_loop_grid, tiny_samples(200 * 2000));
    }
}
