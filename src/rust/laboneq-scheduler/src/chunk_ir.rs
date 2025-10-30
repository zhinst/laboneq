// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::ScheduledNode;
use crate::error::Result;
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{ParameterUid, SectionUid};
use crate::ir::IrKind;

/// This function modifies the IR in place to only include the specified chunk of a sweep loop.
/// When chunking happens, new parameters corresponding to the chunked iterations are created and returned.
///
/// # Arguments
/// * `ir` - The root of the IR to be modified.
/// * `chunk` - A tuple containing:
///     - The UID of the loop to be chunked.
///     - The index of the current chunk.
///     - The total number of chunks the loop is divided into.
/// * `parameters` - A map of existing sweep parameters.
pub fn chunk_ir(
    ir: &mut ScheduledNode,
    chunk: (SectionUid, usize, usize),
    parameters: &HashMap<ParameterUid, SweepParameter>,
) -> Result<Vec<SweepParameter>> {
    let mut chunked_parameters = vec![];
    let _ = chunk_ir_impl(ir, chunk, parameters, &mut chunked_parameters)?;
    Ok(chunked_parameters)
}

/// Recursive implementation of chunking the IR.
///
/// Returns true if the chunk was applied in this subtree.
fn chunk_ir_impl(
    ir: &mut ScheduledNode,
    chunk: (SectionUid, usize, usize),
    parameters: &HashMap<ParameterUid, SweepParameter>,
    chunked_parameters: &mut Vec<SweepParameter>,
) -> Result<bool> {
    match &mut ir.kind {
        IrKind::Loop(obj) => {
            if obj.uid != chunk.0 {
                for child in ir.children.iter_mut() {
                    chunk_ir_impl(child.node.make_mut(), chunk, parameters, chunked_parameters)?;
                }
                return Ok(false);
            }
            let chunk_index = chunk.1;
            let chunk_count = chunk.2;
            let chunk_size = obj.iterations / chunk_count;
            assert!(
                obj.iterations.is_multiple_of(chunk_size),
                "sweep is not evenly divided into chunks"
            );

            let global_iteration_start = chunk_index * chunk_size;
            let global_iteration_end = ((chunk_index + 1) * chunk_size).min(obj.iterations);

            for param in obj.parameters.iter() {
                let old_param = parameters.get(param).unwrap();
                let new_param = old_param.slice(global_iteration_start..global_iteration_end);
                chunked_parameters.push(new_param);
            }
            obj.iterations = chunk_size;
            Ok(true)
        }
        _ => {
            for child in ir.children.iter_mut() {
                if chunk_ir_impl(child.node.make_mut(), chunk, parameters, chunked_parameters)? {
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
    use std::vec;

    use laboneq_common::named_id::NamedId;
    use numeric_array::NumericArray;

    use super::*;
    use crate::experiment::sweep_parameter::SweepParameter;
    use crate::experiment::types::ParameterUid;
    use crate::ir::Loop;
    use crate::scheduled_node::ir_node_structure;

    #[test]
    fn test_chunk_ir_first_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12));
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let loop_top = Loop {
            uid: SectionUid(NamedId::debug_id(0)),
            iterations: 8,
            parameters: vec![],
        };
        let chunked_loop = Loop {
            uid: target_loop_uid,
            iterations: parameter0.len(),
            parameters: vec![parameter0.uid],
        };
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top),
                [(0, IrKind::Loop(chunked_loop), []),]
            )]
        );
        let chunk_index = 0;
        let chunk_count = 3;
        let chunk_info = (target_loop_uid, chunk_index, chunk_count);

        // Run chunking
        let new_params = chunk_ir(
            &mut root,
            chunk_info,
            &HashMap::from_iter([(parameter0.uid, parameter0.clone())]),
        )
        .unwrap();

        // Test that the new parameter has the expected values
        assert_eq!(new_params.len(), 1);
        assert_eq!(new_params[0].uid, parameter0.uid);
        assert_eq!(
            new_params[0].values,
            NumericArray::Integer64(vec![0, 1, 2, 3]).into()
        );

        // Test that the IR has been modified correctly
        let loop_top = Loop {
            uid: SectionUid(NamedId::debug_id(0)),
            iterations: 8,
            parameters: vec![],
        };
        let chunked_loop = Loop {
            uid: target_loop_uid,
            iterations: 4, // Iterations chunked
            parameters: vec![parameter0.uid],
        };
        let root_expected = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top),
                [(0, IrKind::Loop(chunked_loop), []),]
            )]
        );
        assert_eq!(root, root_expected);
    }

    #[test]
    fn test_chunk_ir_last_chunk() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..12));
        let target_loop_uid = SectionUid(NamedId::debug_id(1));
        let loop_top = Loop {
            uid: SectionUid(NamedId::debug_id(0)),
            iterations: 8,
            parameters: vec![],
        };
        let chunked_loop = Loop {
            uid: target_loop_uid,
            iterations: 12,
            parameters: vec![parameter0.uid],
        };
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top),
                [(0, IrKind::Loop(chunked_loop), []),]
            )]
        );
        let chunk_index = 2;
        let chunk_count = 3;
        let chunk_info = (target_loop_uid, chunk_index, chunk_count);
        // Run chunking
        let new_params = chunk_ir(
            &mut root,
            chunk_info,
            &HashMap::from_iter([(parameter0.uid, parameter0.clone())]),
        )
        .unwrap();

        // Test that the new parameter has the expected values
        assert_eq!(new_params.len(), 1);
        assert_eq!(new_params[0].uid, parameter0.uid);
        assert_eq!(
            new_params[0].values,
            NumericArray::Integer64(vec![8, 9, 10, 11]).into()
        );

        // Test that the IR has been modified correctly
        let loop_top = Loop {
            uid: SectionUid(NamedId::debug_id(0)),
            iterations: 8,
            parameters: vec![],
        };
        let chunked_loop = Loop {
            uid: target_loop_uid,
            iterations: 4, // Iterations chunked
            parameters: vec![parameter0.uid],
        };
        let root_expected = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top),
                [(0, IrKind::Loop(chunked_loop), []),]
            )]
        );
        assert_eq!(root, root_expected);
    }
}
