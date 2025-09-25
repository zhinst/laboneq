// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::Result;
use crate::ir::{IrNode, NodeKind, Samples};

fn transform_phase_reset_nodes(node: &mut IrNode, cut_points: &mut HashSet<Samples>) -> Result<()> {
    match node.data_mut() {
        NodeKind::PhaseReset(data) => {
            let has_hw_modulated_signals = data.signals.iter().any(|s| !s.is_sw_modulated());
            node.replace_data(if has_hw_modulated_signals {
                cut_points.insert(*node.offset());
                NodeKind::ResetPhase()
            } else {
                NodeKind::Nop { length: 0 }
            });
        }
        _ => {
            for child in node.iter_children_mut() {
                transform_phase_reset_nodes(child, cut_points)?;
            }
        }
    }
    Ok(())
}

/// Handle hardware phase resets in the IR tree.
///
/// This function inserts an initial reset phase node at the start of the program
/// and transforms all phase reset nodes in the IR tree to hardware reset nodes.
pub fn handle_hw_phase_resets(
    program: &mut IrNode,
    cut_points: &mut HashSet<Samples>,
) -> Result<()> {
    program.insert_child(0, *program.offset(), NodeKind::InitialResetPhase());
    transform_phase_reset_nodes(program, cut_points)?;
    Ok(())
}
