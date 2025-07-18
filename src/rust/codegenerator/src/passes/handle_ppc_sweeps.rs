// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;

/// Transform ppc sweep step nodes from IR to AWG commands
pub fn handle_ppc_sweep_steps(node: &mut ir::IrNode) -> Result<()> {
    match node.data_mut() {
        ir::NodeKind::PpcSweepStep(ir_mod) => {
            let new_node = ir::NodeKind::PpcStep(ir_mod.clone());
            node.replace_data(new_node);
        }
        _ => {
            for child in node.iter_children_mut() {
                handle_ppc_sweep_steps(child)?;
            }
        }
    }
    Ok(())
}
