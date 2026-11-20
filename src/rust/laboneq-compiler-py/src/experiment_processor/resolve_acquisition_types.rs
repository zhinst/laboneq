// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{AcquisitionType, HandleUid};

use crate::error::{Error, Result};

/// Assigns the acquisition type of the enclosing averaging loop to every acquire
/// operation that has none.
///
/// Errors if acquisitions sharing a handle resolve to different acquisition types.
pub(super) fn resolve_acquisition_types(node: &mut ExperimentNode) -> Result<()> {
    resolve(node, AcquisitionType::default(), &mut HashMap::new())
}

fn resolve(
    node: &mut ExperimentNode,
    loop_type: AcquisitionType,
    handle_types: &mut HashMap<HandleUid, AcquisitionType>,
) -> Result<()> {
    let loop_type = match &mut node.kind {
        Operation::AveragingLoop(obj) => obj.acquisition_type,
        Operation::Acquire(obj) => {
            let acquisition_type = *obj.acquisition_type.get_or_insert(loop_type);
            if *handle_types.entry(obj.handle).or_insert(acquisition_type) != acquisition_type {
                return Err(Error::new(format!(
                    "Acquisition type must match across all acquisitions for handle '{}'.",
                    obj.handle.0
                )));
            }
            loop_type
        }
        _ => loop_type,
    };
    for child in node.children.iter_mut() {
        resolve(child, loop_type, handle_types)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::{Acquire, AveragingLoop};
    use laboneq_dsl::types::{AveragingMode, RepetitionMode, SectionAlignment, SectionTimingMode};

    fn averaging_loop(acquisition_type: AcquisitionType) -> Operation {
        Operation::AveragingLoop(AveragingLoop {
            uid: 0.into(),
            count: NonZeroU32::new(1).unwrap(),
            acquisition_type,
            averaging_mode: AveragingMode::Cyclic,
            repetition_mode: RepetitionMode::Fastest,
            reset_oscillator_phase: false,
            alignment: SectionAlignment::Left,
            section_timing_mode: SectionTimingMode::Relaxed,
            reserved_signals: Vec::new(),
        })
    }

    fn acquire(handle: u32, acquisition_type: Option<AcquisitionType>) -> Operation {
        Operation::Acquire(Acquire {
            signal: 1.into(),
            handle: handle.into(),
            length: None,
            kernel: vec![],
            parameters: vec![],
            pulse_parameters: vec![],
            acquisition_type,
        })
    }

    #[test]
    fn test_unset_acquisition_type_inherits_loop() {
        let mut experiment = node_structure!(
            averaging_loop(AcquisitionType::Discrimination),
            [
                (acquire(10, None), []),
                (acquire(11, Some(AcquisitionType::Raw)), []),
            ]
        );
        resolve_acquisition_types(&mut experiment).unwrap();

        let expected = node_structure!(
            averaging_loop(AcquisitionType::Discrimination),
            [
                (acquire(10, Some(AcquisitionType::Discrimination)), []),
                (acquire(11, Some(AcquisitionType::Raw)), []),
            ]
        );
        assert_eq!(experiment, expected);
    }

    #[test]
    fn test_handle_with_conflicting_acquisition_types() {
        let mut experiment = node_structure!(
            averaging_loop(AcquisitionType::Integration),
            [
                (acquire(10, None), []),
                (acquire(10, Some(AcquisitionType::Discrimination)), []),
            ]
        );
        let err = resolve_acquisition_types(&mut experiment).unwrap_err();
        assert!(
            err.to_string()
                .contains("Acquisition type must match across all acquisitions for handle")
        );
    }
}
