// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::error::{Error, Result};
use laboneq_dsl::{ExperimentNode, NodeChild, operation::Operation, types::AveragingMode};

type NodePtr = *const ExperimentNode;

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
    if let Some(acq_index) = find_sequential_acquire_position(node) {
        let innermost_sweep = find_innermost_sweep(&node.children[acq_index]);
        if innermost_sweep.is_none() {
            return Ok(false);
        }
        let acquire_value = node.children[acq_index].kind.clone();
        insert_to_innermost_sweep(
            node.children.get_mut(acq_index).unwrap(),
            &acquire_value,
            &innermost_sweep.unwrap(),
        )?;
        // Remove the original acquire node and move its children up
        let mut acq = node.children.remove(acq_index);
        node.children = acq.make_mut().take_children();
        return Ok(true);
    } else {
        for child in node.children.iter_mut() {
            if resolve_averaging(child.make_mut())? {
                // Exit early if a change was made
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn find_sequential_acquire_position(node: &ExperimentNode) -> Option<usize> {
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

fn find_innermost_sweep(node: &NodeChild) -> Option<NodePtr> {
    let mut prev_sweep = None;
    for child in node.children.iter() {
        if let Some(curr_sweep) = find_innermost_sweep(child) {
            prev_sweep = Some(curr_sweep);
        }
    }
    prev_sweep.or_else(|| {
        if is_sweep(&node.kind) {
            Some(node.as_ptr())
        } else {
            None
        }
    })
}

fn insert_to_innermost_sweep(
    node: &mut Arc<ExperimentNode>,
    acquire: &Operation,
    innermost_sweep: &NodePtr,
) -> Result<bool> {
    if &node.as_ptr() == innermost_sweep {
        let mut averaging = if let Operation::AveragingLoop(acq) = acquire {
            acq.clone()
        } else {
            unreachable!("Internal error: Expected AveragingLoop operation")
        };
        averaging.alignment = if let Operation::Sweep(sweep) = &node.kind {
            sweep.alignment
        } else {
            unreachable!("Internal error: Expected Sweep operation")
        };
        let mut acq = ExperimentNode::new(Operation::AveragingLoop(averaging));
        let node = node.make_mut();
        acq.children = node.take_children();
        node.children.push(acq.into());
        return Ok(true);
    }
    // This validation will ensure that the section graph from acquire loop to inner-most sweep is a linear chain.
    // Each section must have exactly one child section, except for the inner-most sweep.
    if node.children.len() > 1 && &node.as_ptr() != innermost_sweep {
        let msg = format!(
            "Section {} has multiple children. \
            With sequential averaging, the section graph from acquire loop to inner-most sweep must be a linear chain, with only a single subsection at each level.",
            &node
                .kind
                .section_info()
                .map(|info| info.uid.0)
                .expect("Internal error: Section must have a UID")
        );
        return Err(Error::new(msg));
    }
    for child in node.make_mut().children.iter_mut() {
        if insert_to_innermost_sweep(child, acquire, innermost_sweep)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use laboneq_common::named_id::NamedIdStore;
    use laboneq_dsl::{
        node_structure,
        operation::{AveragingLoop, Reserve, Sweep},
        types::{AcquisitionType, RepetitionMode, SectionAlignment, SectionUid, SignalUid},
    };

    fn make_sweep(store: &mut NamedIdStore, name: &str) -> Sweep {
        Sweep {
            uid: SectionUid(store.get_or_insert(name)),
            parameters: vec![],
            alignment: SectionAlignment::Right,
            reset_oscillator_phase: false,
            count: 0,
            chunking: None,
        }
    }

    fn make_acquire_rt(
        store: &mut NamedIdStore,
        averaging_mode: AveragingMode,
        alignment: SectionAlignment,
    ) -> AveragingLoop {
        AveragingLoop {
            uid: SectionUid(store.get_or_insert("shots")),
            acquisition_type: AcquisitionType::Spectroscopy,
            count: 1,
            averaging_mode,
            repetition_mode: RepetitionMode::Fastest,
            reset_oscillator_phase: false,
            alignment,
        }
    }

    /// Test that acquire loop is handled appropriately for sequential averaging.
    #[test]
    fn test_resolve_averaging_sequential_averaging() {
        let mut store = NamedIdStore::new();
        let reserve = Reserve {
            signal: SignalUid(store.get_or_insert("reserve")),
        };

        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::Sweep(make_sweep(&mut store, "near-time-sweep")),
                [(
                    Operation::AveragingLoop(make_acquire_rt(
                        &mut store,
                        AveragingMode::Sequential,
                        SectionAlignment::Left
                    )),
                    [(
                        Operation::Sweep(make_sweep(&mut store, "sweep0")),
                        [(
                            Operation::Sweep(make_sweep(&mut store, "sweep1")),
                            [
                                (Operation::Reserve(reserve.clone()), []),
                                (Operation::Reserve(reserve.clone()), [])
                            ]
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
                            Operation::AveragingLoop(make_acquire_rt(
                                &mut store,
                                AveragingMode::Sequential,
                                SectionAlignment::Right // Averaging inherits alignment from innermost sweep
                            )),
                            [
                                (Operation::Reserve(reserve.clone()), []),
                                (Operation::Reserve(reserve.clone()), [])
                            ]
                        )]
                    )]
                )]
            )]
        );
        assert_eq!(tree, tree_expected);
    }

    /// Test that acquire loop is handled appropriately for non-sequential averaging.
    #[test]
    fn test_resolve_averaging_non_sequential_averaging() {
        let mut store = NamedIdStore::new();
        let reserve = Reserve {
            signal: SignalUid(store.get_or_insert("reserve")),
        };

        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_acquire_rt(
                    &mut store,
                    AveragingMode::Cyclic,
                    SectionAlignment::Left
                )),
                [(
                    Operation::Sweep(make_sweep(&mut store, "sweep0")),
                    [(
                        Operation::Sweep(make_sweep(&mut store, "sweep1")),
                        [(Operation::Reserve(reserve.clone()), [])]
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
        let reserve = Reserve {
            signal: SignalUid(store.get_or_insert("reserve")),
        };

        // AcquireLoopRT with sequential averaging children cannot have siblings
        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_acquire_rt(
                    &mut store,
                    AveragingMode::Sequential,
                    SectionAlignment::Left
                )),
                [
                    (
                        Operation::Sweep(make_sweep(&mut store, "sweep0")),
                        [(
                            Operation::Sweep(make_sweep(&mut store, "sweep1")),
                            [(Operation::Reserve(reserve.clone()), [])]
                        )]
                    ),
                    (Operation::Reserve(reserve.clone()), [])
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

        // Sections inside AcquireLoopRT must be linear to innermost sweep (Each subsection must have exactly one child)
        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_acquire_rt(
                    &mut store,
                    AveragingMode::Sequential,
                    SectionAlignment::Left
                )),
                [(
                    Operation::Sweep(make_sweep(&mut store, "sweep0")),
                    [
                        (
                            Operation::Sweep(make_sweep(&mut store, "sweep1")),
                            [(Operation::Reserve(reserve.clone()), [])]
                        ),
                        (Operation::Reserve(reserve.clone()), [])
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
