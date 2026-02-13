// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::num::NonZeroU32;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{ParameterUid, SweepParameter};

use crate::error::Result;

/// Information about chunking of experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkingInfo {
    /// Current chunking index (0-based).
    pub index: usize,
    /// Total number of chunks.
    pub count: NonZeroU32,
}

impl ChunkingInfo {
    fn current_chunk_range(&self, total_iterations: NonZeroU32) -> std::ops::Range<usize> {
        let chunk_size = total_iterations.get() / self.count.get();
        let start = self.index * chunk_size as usize;
        let end = ((self.index + 1) * chunk_size as usize).min(total_iterations.get() as usize);
        start..end
    }

    fn chunk_size(&self, total_iterations: NonZeroU32) -> NonZeroU32 {
        let chunk_size = total_iterations.get() / self.count.get();
        assert!(
            total_iterations.get().is_multiple_of(chunk_size),
            "sweep is not evenly divided into chunks"
        );
        chunk_size
            .try_into()
            .expect("Expected chunk size to be non-zero and fit into u32")
    }
}

/// This function modifies the experiment in place to only include the specified chunk of a sweep loop.
/// When chunking happens, the `parameters` are updated accordingly to match the chunked iterations.
pub(crate) fn chunk_experiment(
    ir: &mut ExperimentNode,
    parameters: &mut HashMap<ParameterUid, SweepParameter>,
    chunking_info: &ChunkingInfo,
) -> Result<()> {
    let _ = chunk_experiment_impl(ir, parameters, chunking_info)?;
    Ok(())
}

fn chunk_experiment_impl(
    ir: &mut ExperimentNode,
    parameters: &mut HashMap<ParameterUid, SweepParameter>,
    chunking_info: &ChunkingInfo,
) -> Result<bool> {
    match &mut ir.kind {
        Operation::Sweep(obj) => {
            if obj.chunking.is_none() {
                for child in ir.children.iter_mut() {
                    chunk_experiment_impl(child.make_mut(), parameters, chunking_info)?;
                }
                return Ok(false);
            }
            for param in obj.parameters.iter() {
                let old_param = parameters.get(param).unwrap();
                let new_param = old_param.slice(chunking_info.current_chunk_range(obj.count));
                parameters.insert(old_param.uid, new_param);
            }
            obj.count = chunking_info.chunk_size(obj.count);
            obj.chunking = None;
            Ok(true)
        }
        _ => {
            for child in ir.children.iter_mut() {
                if chunk_experiment_impl(child.make_mut(), parameters, chunking_info)? {
                    // Chunked sweep found, early return
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChunkingInfo, chunk_experiment};
    use laboneq_common::named_id::NamedId;
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::builders::SweepBuilder;
    use laboneq_dsl::operation::{Chunking, Operation};
    use laboneq_dsl::types::{ParameterUid, SectionUid, SweepParameter};
    use numeric_array::NumericArray;
    use std::collections::HashMap;
    use std::num::NonZeroU32;
    use std::vec;

    #[test]
    fn test_chunk_experiment_first_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12)).unwrap();
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let chunked_sweep = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            (parameter0.len() as u32).try_into().unwrap(),
        )
        .chunking(Chunking::Count {
            count: NonZeroU32::new(3).unwrap(),
        })
        .build();
        let mut root = node_structure!(Operation::Root, [(Operation::Sweep(chunked_sweep), [])]);

        let mut parameters = HashMap::from_iter([(parameter0.uid, parameter0.clone())]);
        // Run chunking
        chunk_experiment(
            &mut root,
            &mut parameters,
            &ChunkingInfo {
                index: 0,
                count: 3.try_into().unwrap(),
            },
        )
        .unwrap();

        // Test that the new parameter has the expected values
        assert_eq!(parameters.len(), 1);
        assert_eq!(
            parameters[&parameter0.uid].values,
            NumericArray::Integer64(vec![0, 1, 2, 3]).into()
        );

        let chunked_sweep_expected =
            SweepBuilder::new(target_loop_uid, vec![parameter0.uid], 4.try_into().unwrap()).build();

        let root_expected = node_structure!(
            Operation::Root,
            [(Operation::Sweep(chunked_sweep_expected), [])]
        );
        assert_eq!(root, root_expected);
    }

    #[test]
    fn test_chunk_experiment_last_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12)).unwrap();
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let chunked_sweep = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            (parameter0.len() as u32).try_into().unwrap(),
        )
        .chunking(Chunking::Count {
            count: NonZeroU32::new(3).unwrap(),
        })
        .build();

        // Source experiment
        let mut root = node_structure!(Operation::Root, [(Operation::Sweep(chunked_sweep), [])]);

        let mut parameters = HashMap::from_iter([(parameter0.uid, parameter0.clone())]);

        // Run chunking
        chunk_experiment(
            &mut root,
            &mut parameters,
            &ChunkingInfo {
                index: 2,
                count: 3.try_into().unwrap(),
            },
        )
        .unwrap();

        // Test that the new parameter has the expected values
        assert_eq!(parameters.len(), 1);
        assert_eq!(
            parameters[&parameter0.uid].values,
            NumericArray::Integer64(vec![8, 9, 10, 11]).into()
        );

        let chunked_sweep_expected =
            SweepBuilder::new(target_loop_uid, vec![parameter0.uid], 4.try_into().unwrap()).build();

        let root_expected = node_structure!(
            Operation::Root,
            [(Operation::Sweep(chunked_sweep_expected), [])]
        );
        assert_eq!(root, root_expected);
    }
}
