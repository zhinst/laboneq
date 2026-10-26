// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::Operation;
use laboneq_dsl::types::{ParameterUid, SweepParameter};

use crate::error::{Error, Result};

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
}

/// Restricts the experiment's sweep parameters to the current chunk.
///
/// TODO: In future consider chunking the sweep itself directly, instead of
/// working around it like this. Will perhaps need a refactor.
pub(crate) fn slice_chunked_parameters(
    ir: &ExperimentNode,
    parameters: &mut HashMap<ParameterUid, SweepParameter>,
    chunking_info: &ChunkingInfo,
) -> Result<()> {
    let mut chunk_bound_parameters = HashSet::new();
    chunk_experiment_impl(ir, parameters, chunking_info, &mut chunk_bound_parameters)?;
    Ok(())
}

fn chunk_experiment_impl(
    ir: &ExperimentNode,
    parameters: &mut HashMap<ParameterUid, SweepParameter>,
    chunking_info: &ChunkingInfo,
    chunk_bound_parameters: &mut HashSet<ParameterUid>,
) -> Result<()> {
    if let Operation::Sweep(obj) = &ir.kind {
        for param in obj.parameters.iter() {
            // TODO: Currently parameters used in a chunked sweep cannot be used in any other sweeps
            if chunk_bound_parameters.contains(param) {
                let msg = format!(
                    "Parameters used in a chunked sweep cannot be used in multiple sweeps: '{}'",
                    param.0
                );
                return Err(Error::new(msg));
            }
            chunk_bound_parameters.insert(*param);
        }

        if obj.is_chunked() {
            for param in obj.parameters.iter() {
                let old_param = parameters.get(param).unwrap();
                let new_param = old_param.slice(chunking_info.current_chunk_range(obj.count));
                parameters.insert(*param, new_param);
                chunk_bound_parameters.insert(*param);
            }
        }
    }
    for child in ir.children.iter() {
        chunk_experiment_impl(child, parameters, chunking_info, chunk_bound_parameters)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChunkingInfo, slice_chunked_parameters};
    use laboneq_common::named_id::NamedId;
    use laboneq_dsl::node_structure;
    use laboneq_dsl::operation::Operation;
    use laboneq_dsl::operation::builders::SweepBuilder;
    use laboneq_dsl::types::{ParameterUid, SectionUid, SweepParameter};
    use numeric_array::NumericArray;
    use std::collections::HashMap;
    use std::vec;

    #[test]
    fn test_slice_chunked_parameters_first_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12)).unwrap();
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let chunked_sweep = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            (parameter0.len() as u32).try_into().unwrap(),
        )
        .chunk_count(3.try_into().unwrap())
        .build();
        let root = node_structure!(Operation::Root, [(Operation::Sweep(chunked_sweep), [])]);

        let mut parameters = HashMap::from_iter([(parameter0.uid, parameter0.clone())]);
        slice_chunked_parameters(
            &root,
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
        // The sliced length is what lowering uses as the loop's iteration count.
        assert_eq!(parameters[&parameter0.uid].len(), 4);

        // The experiment tree is left untouched; only the parameters are sliced.
        let chunked_sweep_expected = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            12.try_into().unwrap(),
        )
        .chunk_count(3.try_into().unwrap())
        .build();

        let root_expected = node_structure!(
            Operation::Root,
            [(Operation::Sweep(chunked_sweep_expected), [])]
        );
        assert_eq!(root, root_expected);
    }

    #[test]
    fn test_slice_chunked_parameters_last_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12)).unwrap();
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let chunked_sweep = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            (parameter0.len() as u32).try_into().unwrap(),
        )
        .chunk_count(3.try_into().unwrap())
        .build();

        // Source experiment
        let root = node_structure!(Operation::Root, [(Operation::Sweep(chunked_sweep), [])]);

        let mut parameters = HashMap::from_iter([(parameter0.uid, parameter0.clone())]);

        slice_chunked_parameters(
            &root,
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
        // The sliced length is what lowering uses as the loop's iteration count.
        assert_eq!(parameters[&parameter0.uid].len(), 4);

        // The experiment tree is left untouched; only the parameters are sliced.
        let chunked_sweep_expected = SweepBuilder::new(
            target_loop_uid,
            vec![parameter0.uid],
            12.try_into().unwrap(),
        )
        .chunk_count(3.try_into().unwrap())
        .build();

        let root_expected = node_structure!(
            Operation::Root,
            [(Operation::Sweep(chunked_sweep_expected), [])]
        );
        assert_eq!(root, root_expected);
    }
}
