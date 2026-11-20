// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{AveragingLoop, Operation};
use laboneq_dsl::types::AveragingMode;

use crate::error::{Error, Result};

/// Transformation pass to resolve sequential averaging loop in the experiment.
///
/// With sequential averaging, the real-time acquisition loop is moved to the bottom
/// of the innermost sweep.
///
/// The following must be fulfilled for sequential averaging:
///
/// * The section graph from the acquisition loop to the innermost sweep must be a linear
///   chain, with only a single subsection at each level. The innermost sweep structure is not
///   restricted.
///
/// # Returns
///
/// * `Ok(true)` if the IR was modified.
/// * `Ok(false)` if no modifications were made.
/// * `Err` if the experiment structure is invalid.
pub(super) fn resolve_averaging(node: &mut ExperimentNode) -> Result<bool> {
    if let Some(avg_index) = find_sequential_averaging_position(node) {
        let Operation::AveragingLoop(avg) = node.children[avg_index].kind.clone() else {
            unreachable!("Internal error: Expected AveragingLoop operation");
        };
        if !insert_to_innermost_sweep(&mut node.children[avg_index], &avg)? {
            return Ok(false);
        }
        // Remove the original averaging loop node and move its children up
        let mut avg = node.children.remove(avg_index);
        node.children = avg.take_children();
        return Ok(true);
    } else {
        for child in node.children.iter_mut() {
            if resolve_averaging(child)? {
                // Exit early if a change was made
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn find_sequential_averaging_position(node: &ExperimentNode) -> Option<usize> {
    for (idx, child) in node.children.iter().enumerate() {
        if matches!(&child.kind, Operation::AveragingLoop(obj) if obj.averaging_mode == AveragingMode::Sequential)
        {
            return Some(idx);
        }
    }
    None
}

fn is_sweep(node: &Operation) -> bool {
    matches!(node, Operation::Sweep(_))
}

/// Finds the path of child indices from `node` down to the inner-most sweep in its subtree,
/// preferring the most deeply nested sweep when sweeps are chained.
///
/// Returns `None` if `node`'s subtree (`node` included) contains no sweep at all, The path is returned in
/// leaf-to-root order; reverse it to walk from `node` down to the sweep.
fn find_innermost_sweep_path(node: &ExperimentNode) -> Option<Vec<usize>> {
    for (idx, child) in node.children.iter().enumerate() {
        if let Some(mut path) = find_innermost_sweep_path(child) {
            path.push(idx);
            return Some(path);
        }
    }
    is_sweep(&node.kind).then(Vec::new)
}

/// Descends into `node` and inserts `averaging_op` as the sole child of the inner-most sweep,
/// wrapping the sweep's previous children.
///
/// A sweep is considered inner-most if none of its descendants are themselves a sweep;
/// its own children structure is unrestricted (see module docs). Returns `Ok(true)` once
/// inserted, or `Ok(false)` if `node`'s subtree contains no sweep at all.
fn insert_to_innermost_sweep(
    node: &mut ExperimentNode,
    averaging_op: &AveragingLoop,
) -> Result<bool> {
    let Some(mut path) = find_innermost_sweep_path(node) else {
        // No sweep anywhere below `node`: nothing to do, and the linear-chain constraint
        // below does not apply to this subtree.
        return Ok(false);
    };
    path.reverse();

    // Walk down to the inner-most sweep, validating along the way that the section graph
    // from the averaging loop to the inner-most sweep is a linear chain: each section must
    // have exactly one child, except for the inner-most sweep itself.
    let mut current = node;
    for idx in path {
        if current.children.len() > 1 {
            let msg = format!(
                "Section {} has multiple children. \
                With sequential averaging, the section graph from acquire loop to inner-most sweep must be a linear chain, with only a single subsection at each level.",
                current
                    .kind
                    .section_info()
                    .map(|info| info.uid.0)
                    .expect("Internal error: Section must have a UID")
            );
            return Err(Error::new(msg));
        }
        current = &mut current.children[idx];
    }

    let mut averaging = averaging_op.clone();
    averaging.alignment = if let Operation::Sweep(sweep) = &current.kind {
        sweep.alignment
    } else {
        unreachable!("Internal error: Expected Sweep operation")
    };
    let mut acq_node = ExperimentNode::new(Operation::AveragingLoop(averaging));
    acq_node.children = current.take_children();
    current.children.push(acq_node);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use laboneq_common::named_id::NamedIdStore;
    use laboneq_dsl::{
        node_structure,
        operation::{AveragingLoop, ResetOscillatorPhase, Sweep, builders::SweepBuilder},
        types::{AcquisitionType, RepetitionMode, SectionAlignment, SectionTimingMode, SectionUid},
    };

    fn make_sweep(store: &mut NamedIdStore, name: &str) -> Sweep {
        let builder = SweepBuilder::new(
            SectionUid(store.get_or_insert(name)),
            vec![],
            1.try_into().unwrap(),
        )
        .alignment(SectionAlignment::Right);
        builder.build()
    }

    fn make_averaging_rt(
        store: &mut NamedIdStore,
        averaging_mode: AveragingMode,
        alignment: SectionAlignment,
    ) -> AveragingLoop {
        AveragingLoop {
            uid: SectionUid(store.get_or_insert("shots")),
            acquisition_type: AcquisitionType::Spectroscopy,
            count: NonZeroU32::new(1).unwrap(),
            averaging_mode,
            repetition_mode: RepetitionMode::Fastest,
            reset_oscillator_phase: false,
            alignment,
            section_timing_mode: SectionTimingMode::Relaxed,
            reserved_signals: Vec::new(),
        }
    }

    /// Test that averaging loop is handled appropriately for sequential averaging.
    #[test]
    fn test_resolve_averaging_sequential_averaging() {
        let mut store = NamedIdStore::new();
        let leaf = Operation::ResetOscillatorPhase(ResetOscillatorPhase::default());

        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::Sweep(make_sweep(&mut store, "near-time-sweep")),
                [(
                    Operation::AveragingLoop(make_averaging_rt(
                        &mut store,
                        AveragingMode::Sequential,
                        SectionAlignment::Left
                    )),
                    [(
                        Operation::Sweep(make_sweep(&mut store, "sweep0")),
                        [(
                            Operation::Sweep(make_sweep(&mut store, "sweep1")),
                            [(leaf.clone(), []), (leaf.clone(), [])]
                        )]
                    )]
                )]
            )]
        );
        resolve_averaging(&mut tree).unwrap();
        let tree_expected = node_structure!(
            Operation::Root,
            [(
                Operation::Sweep(make_sweep(&mut store, "near-time-sweep")),
                [(
                    Operation::Sweep(make_sweep(&mut store, "sweep0")),
                    [(
                        Operation::Sweep(make_sweep(&mut store, "sweep1")),
                        [(
                            Operation::AveragingLoop(make_averaging_rt(
                                &mut store,
                                AveragingMode::Sequential,
                                SectionAlignment::Right // Averaging inherits alignment from innermost sweep
                            )),
                            [(leaf.clone(), []), (leaf.clone(), [])]
                        )]
                    )]
                )]
            )]
        );
        assert_eq!(tree, tree_expected);
    }

    /// Test that averaging loop is handled appropriately for non-sequential averaging.
    #[test]
    fn test_resolve_averaging_non_sequential_averaging() {
        let mut store = NamedIdStore::new();
        let leaf = Operation::ResetOscillatorPhase(ResetOscillatorPhase::default());

        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_averaging_rt(
                    &mut store,
                    AveragingMode::Cyclic,
                    SectionAlignment::Left
                )),
                [(
                    Operation::Sweep(make_sweep(&mut store, "sweep0")),
                    [(
                        Operation::Sweep(make_sweep(&mut store, "sweep1")),
                        [(leaf.clone(), [])]
                    )]
                )]
            )]
        );
        // No changes expected
        let tree_expected = tree.clone();
        resolve_averaging(&mut tree).unwrap();
        assert_eq!(tree, tree_expected);
    }

    #[test]
    fn test_invalid_experiment_structure() {
        let mut store = NamedIdStore::new();
        let leaf = Operation::ResetOscillatorPhase(ResetOscillatorPhase::default());

        // AveragingLoop with sequential averaging children cannot have siblings
        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_averaging_rt(
                    &mut store,
                    AveragingMode::Sequential,
                    SectionAlignment::Left
                )),
                [
                    (
                        Operation::Sweep(make_sweep(&mut store, "sweep0")),
                        [(
                            Operation::Sweep(make_sweep(&mut store, "sweep1")),
                            [(leaf.clone(), [])]
                        )]
                    ),
                    (leaf.clone(), [])
                ]
            )]
        );
        let err_msg = format!(
            "Section {} has multiple children.",
            store.get("shots").unwrap()
        );
        assert!(
            resolve_averaging(&mut tree)
                .unwrap_err()
                .to_string()
                .contains(&err_msg)
        );

        // Sections inside AveragingLoop must be linear to innermost sweep (Each subsection must have exactly one child)
        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_averaging_rt(
                    &mut store,
                    AveragingMode::Sequential,
                    SectionAlignment::Left
                )),
                [(
                    Operation::Sweep(make_sweep(&mut store, "sweep0")),
                    [
                        (
                            Operation::Sweep(make_sweep(&mut store, "sweep1")),
                            [(leaf.clone(), [])]
                        ),
                        (leaf.clone(), [])
                    ]
                ),]
            ),]
        );
        let err_msg = format!(
            "Section {} has multiple children.",
            store.get("sweep0").unwrap()
        );
        assert!(
            resolve_averaging(&mut tree)
                .unwrap_err()
                .to_string()
                .contains(&err_msg)
        );
    }
}
