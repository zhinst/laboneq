// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::Result;
use crate::experiment::ExperimentNode;
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{Operation, ParameterUid};

/// Information about chunking of experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkingInfo {
    /// Current chunking index (0-based).
    pub index: usize,
    /// Total number of chunks.
    pub count: usize,
}

impl ChunkingInfo {
    fn current_chunk_range(&self, total_iterations: usize) -> std::ops::Range<usize> {
        let chunk_size = total_iterations / self.count;
        let start = self.index * chunk_size;
        let end = ((self.index + 1) * chunk_size).min(total_iterations);
        start..end
    }

    fn chunk_size(&self, total_iterations: usize) -> usize {
        let chunk_size = total_iterations / self.count;
        assert!(
            total_iterations.is_multiple_of(chunk_size),
            "sweep is not evenly divided into chunks"
        );
        chunk_size
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
                let new_param =
                    old_param.slice(chunking_info.current_chunk_range(obj.count as usize));
                parameters.insert(old_param.uid, new_param);
            }
            obj.count = chunking_info.chunk_size(obj.count as usize) as u32;
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
    use crate::experiment::builders::SweepBuilder;
    use crate::experiment::sweep_parameter::SweepParameter;
    use crate::experiment::types::{Chunking, Operation, ParameterUid, SectionUid};
    use crate::node_structure;
    use laboneq_common::named_id::NamedId;
    use numeric_array::NumericArray;
    use std::collections::HashMap;
    use std::vec;

    #[test]
    fn test_chunk_experiment_first_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12));
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let chunked_sweep = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            parameter0.len() as u32,
        )
        .chunking(Chunking::Count { count: 3 })
        .build();
        let mut root = node_structure!(Operation::Root, [(Operation::Sweep(chunked_sweep), [])]);

        let mut parameters = HashMap::from_iter([(parameter0.uid, parameter0.clone())]);
        // Run chunking
        chunk_experiment(
            &mut root,
            &mut parameters,
            &ChunkingInfo { index: 0, count: 3 },
        )
        .unwrap();

        // Test that the new parameter has the expected values
        assert_eq!(parameters.len(), 1);
        assert_eq!(
            parameters[&parameter0.uid].values,
            NumericArray::Integer64(vec![0, 1, 2, 3]).into()
        );

        let chunked_sweep_expected =
            SweepBuilder::new(target_loop_uid, vec![parameter0.uid], 4).build();

        let root_expected = node_structure!(
            Operation::Root,
            [(Operation::Sweep(chunked_sweep_expected), [])]
        );
        assert_eq!(root, root_expected);
    }

    #[test]
    fn test_chunk_experiment_last_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12));
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let chunked_sweep = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            parameter0.len() as u32,
        )
        .chunking(Chunking::Count { count: 3 })
        .build();

        // Source experiment
        let mut root = node_structure!(Operation::Root, [(Operation::Sweep(chunked_sweep), [])]);

        let mut parameters = HashMap::from_iter([(parameter0.uid, parameter0.clone())]);

        // Run chunking
        chunk_experiment(
            &mut root,
            &mut parameters,
            &ChunkingInfo { index: 2, count: 3 },
        )
        .unwrap();

        // Test that the new parameter has the expected values
        assert_eq!(parameters.len(), 1);
        assert_eq!(
            parameters[&parameter0.uid].values,
            NumericArray::Integer64(vec![8, 9, 10, 11]).into()
        );

        let chunked_sweep_expected =
            SweepBuilder::new(target_loop_uid, vec![parameter0.uid], 4).build();

        let root_expected = node_structure!(
            Operation::Root,
            [(Operation::Sweep(chunked_sweep_expected), [])]
        );
        assert_eq!(root, root_expected);
    }
}
