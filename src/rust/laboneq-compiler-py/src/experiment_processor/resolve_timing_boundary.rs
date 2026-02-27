// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::{ExperimentNode, operation::Operation};

use crate::error::{Error, Result};

/// Resolve real-time and near-time boundaries in the IR tree.
///
/// This function ensures that real-time averaging loop ([`IrVariant::AveragingLoop`]) is enclosed within a [`IrVariant::RealTimeSection`].
/// It also checks that there is exactly one real-time averaging loop in the entire experiment.
///
/// # Returns
///
/// * `Ok(())` if the experiment has exactly one real-time averaging loop.
/// * `Err(Error)` if there are zero or multiple real-time averaging loops.
pub(super) fn resolve_timing_boundary(node: &mut ExperimentNode) -> Result<()> {
    let averaging_loop_count = resolve_timing_boundary_impl(node)?;
    if averaging_loop_count != 1 {
        Err(Error::new(format!(
            "Experiment must have exactly one real time acquisition loop. Found {averaging_loop_count}."
        )))
    } else {
        Ok(())
    }
}

fn resolve_timing_boundary_impl(node: &mut ExperimentNode) -> Result<usize> {
    let mut averaging_loop_count = 0;
    let mut averaging_loops_indexes = vec![];
    for (i, child) in node.children.iter_mut().enumerate() {
        if matches!(child.kind, Operation::AveragingLoop(_)) {
            averaging_loops_indexes.push(i);
        }
        averaging_loop_count += resolve_timing_boundary_impl(child.make_mut())?;
    }
    for averaging_loop in &averaging_loops_indexes {
        let mut new_node = ExperimentNode::new(Operation::RealTimeBoundary);
        let children = node.children.remove(*averaging_loop);
        new_node.children.push(children);
        validate_real_time_boundary_nodes(&new_node)?;
        node.children.push(new_node.into());
    }
    Ok(averaging_loop_count + averaging_loops_indexes.len())
}

/// Validate that all nodes under a real-time boundary are compatible with real-time execution.
fn validate_real_time_boundary_nodes(node: &ExperimentNode) -> Result<()> {
    node.kind
        .validate_real_time_compatible()
        .map_err(Error::new)?;
    for child in &node.children {
        validate_real_time_boundary_nodes(child)?
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use laboneq_common::named_id::NamedIdStore;
    use laboneq_dsl::{
        node_structure,
        operation::{AveragingLoop, Operation},
        types::{AcquisitionType, AveragingMode, RepetitionMode, SectionAlignment, SectionUid},
    };

    fn make_acquire_rt(store: &mut NamedIdStore) -> AveragingLoop {
        AveragingLoop {
            uid: SectionUid(store.get_or_insert("shots")),
            acquisition_type: AcquisitionType::Spectroscopy,
            count: NonZeroU32::new(1).unwrap(),
            averaging_mode: AveragingMode::Cyclic,
            repetition_mode: RepetitionMode::Fastest,
            reset_oscillator_phase: false,
            alignment: SectionAlignment::Left,
        }
    }

    #[test]
    fn test_resolve_timing_boundary() {
        let mut store = NamedIdStore::new();

        let mut tree = node_structure!(
            Operation::Root,
            [(Operation::AveragingLoop(make_acquire_rt(&mut store)), []),]
        );
        resolve_timing_boundary(&mut tree).unwrap();
        let tree_expected = node_structure!(
            Operation::Root,
            [(
                Operation::RealTimeBoundary,
                [(Operation::AveragingLoop(make_acquire_rt(&mut store)), [])]
            ),]
        );
        assert_eq!(tree, tree_expected);
    }

    #[test]
    fn test_multiple_averaging_loops() {
        let mut store = NamedIdStore::new();

        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_acquire_rt(&mut store)),
                [(Operation::AveragingLoop(make_acquire_rt(&mut store)), [])]
            ),]
        );
        let err_msg = format!("Found {}.", 2);
        assert!(
            resolve_timing_boundary(&mut tree)
                .unwrap_err()
                .to_string()
                .contains(&err_msg)
        );
    }
}
