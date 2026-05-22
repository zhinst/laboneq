// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;

use crate::error::{Error, Result};

/// Resolve real-time boundary in the experiment tree.
///
/// This function ensures that real-time averaging loop ([`Operation::AveragingLoop`]) is enclosed within a [`Operation::RealTimeBoundary`].
/// It also checks that there is exactly one real-time averaging loop in the entire experiment.
///
/// # Returns
///
/// * `Ok(())` if the experiment has exactly one real-time averaging loop.
/// * `Err(Error)` if there are zero or multiple real-time averaging loops.
pub(super) fn resolve_timing_boundary(node: &mut ExperimentNode) -> Result<()> {
    let averaging_loop_count = resolve_timing_boundary_impl(node)?;
    if averaging_loop_count == 1 {
        // If there is exactly one averaging loop, we are good.
        return Ok(());
    }
    Err(Error::new(format!(
        "Experiment must have exactly one real time acquisition loop. Found {averaging_loop_count}."
    )))
}

fn resolve_timing_boundary_impl(node: &mut ExperimentNode) -> Result<usize> {
    let mut averaging_loop_count = 0;
    let mut i = 0;

    while i < node.children.len() {
        averaging_loop_count += resolve_timing_boundary_impl(node.children[i].make_mut())?;
        if matches!(node.children[i].kind, Operation::AveragingLoop(_)) {
            // Wrap the averaging loop inside a real-time boundary.
            let children_count = node.children.len();
            let averaging_loop = node.children.swap_remove(i);
            let mut rt_node = ExperimentNode::new(Operation::RealTimeBoundary);
            rt_node.children.push(averaging_loop);
            validate_real_time_boundary_nodes(&rt_node)?;
            node.children.push(rt_node.into());
            node.children.swap(i, children_count - 1);
            averaging_loop_count += 1;
        }
        i += 1;
    }
    Ok(averaging_loop_count)
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
    use laboneq_common::named_id::NamedId;
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::{AveragingLoop, NearTimeCallback};
    use laboneq_dsl::types::{
        AcquisitionType, AveragingMode, RepetitionMode, SectionAlignment, SectionTimingMode,
    };

    fn make_acquire_rt() -> AveragingLoop {
        AveragingLoop {
            uid: 1.into(),
            acquisition_type: AcquisitionType::Spectroscopy,
            count: NonZeroU32::new(1).unwrap(),
            averaging_mode: AveragingMode::Cyclic,
            repetition_mode: RepetitionMode::Fastest,
            reset_oscillator_phase: false,
            alignment: SectionAlignment::Left,
            section_timing_mode: SectionTimingMode::Relaxed,
        }
    }

    fn near_time_callback(uid: u32) -> Operation {
        Operation::NearTimeCallback(NearTimeCallback {
            callback_id: NamedId::debug_id(uid),
            args: vec![],
        })
    }

    #[test]
    fn test_resolve_timing_boundary() {
        let mut tree = node_structure!(
            Operation::Root,
            [
                (near_time_callback(1), []),
                (near_time_callback(2), []),
                (Operation::AveragingLoop(make_acquire_rt()), []),
                (near_time_callback(3), []),
                (near_time_callback(4), []),
                (near_time_callback(5), []),
                (near_time_callback(6), []),
            ]
        );

        resolve_timing_boundary(&mut tree).unwrap();

        let tree_expected = node_structure!(
            Operation::Root,
            [
                (near_time_callback(1), []),
                (near_time_callback(2), []),
                (
                    Operation::RealTimeBoundary,
                    [(Operation::AveragingLoop(make_acquire_rt()), [])]
                ),
                (near_time_callback(3), []),
                (near_time_callback(4), []),
                (near_time_callback(5), []),
                (near_time_callback(6), []),
            ]
        );
        assert_eq!(tree, tree_expected);
    }

    #[test]
    fn test_multiple_averaging_loops() {
        let mut tree = node_structure!(
            Operation::Root,
            [(
                Operation::AveragingLoop(make_acquire_rt()),
                [(Operation::AveragingLoop(make_acquire_rt()), [])]
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
