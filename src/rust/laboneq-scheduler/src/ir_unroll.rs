// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ScheduledNode;
use crate::error::Result;
use crate::ir::IrKind;

/// This function modifies the IR to unroll all sweep unconditionally.
pub fn unroll_loops(ir: &mut ScheduledNode) -> Result<()> {
    match &mut ir.kind {
        IrKind::Loop(obj) => {
            // Skip unrolling for loops without parameters or if the loop is already fully unrolled
            if obj.parameters.is_empty() || obj.iterations == ir.children.len() {
                for child in ir.children.iter_mut() {
                    unroll_loops(child.node.make_mut())?;
                }
                return Ok(());
            }
            assert!(
                ir.children.len() == 1,
                "Loop must have exactly one child to unroll."
            );
            // Traverse and unroll the first iteration and clone it for the remaining iterations
            let mut iteration = ir.children.pop().unwrap().clone();
            unroll_loops(iteration.node.make_mut())?;
            ir.children = vec![iteration; obj.iterations];
            Ok(())
        }
        _ => {
            for child in ir.children.iter_mut() {
                unroll_loops(child.node.make_mut())?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::experiment::sweep_parameter::SweepParameter;
    use crate::experiment::types::{ParameterUid, SectionUid};
    use crate::ir::{IrKind, Loop};
    use crate::scheduled_node::ir_node_structure;
    use laboneq_common::named_id::NamedId;

    #[test]
    fn test_unroll_loop() {
        let parameter0 =
            SweepParameter::new(ParameterUid(NamedId::debug_id(1)), Vec::from_iter(0..4));
        let loop_top = Loop {
            uid: SectionUid(NamedId::debug_id(0)),
            iterations: 8,
            parameters: vec![],
        };
        let loop_to_unroll = Loop {
            uid: SectionUid(NamedId::debug_id(1)),
            iterations: parameter0.len(),
            parameters: vec![parameter0.uid],
        };
        let mut root = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top.clone()),
                [(
                    0,
                    IrKind::Loop(loop_to_unroll.clone()),
                    [(0, IrKind::LoopIteration, [])]
                ),]
            )]
        );
        unroll_loops(&mut root).unwrap();
        let root_expected = ir_node_structure!(
            IrKind::Root,
            [(
                0,
                IrKind::Loop(loop_top),
                [(
                    0,
                    IrKind::Loop(loop_to_unroll),
                    [
                        (0, IrKind::LoopIteration, []),
                        (0, IrKind::LoopIteration, []),
                        (0, IrKind::LoopIteration, []),
                        (0, IrKind::LoopIteration, [])
                    ]
                ),]
            )]
        );
        assert_eq!(root, root_expected);
        // Unroll again, should have no effect
        unroll_loops(&mut root).unwrap();
        assert_eq!(root, root_expected);
    }
}
