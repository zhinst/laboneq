// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{Result, ir};
use anyhow::anyhow;

fn handle_match(node: &mut ir::IrNode, delay: &ir::Samples, state: Option<u16>) -> Result<()> {
    // TODO: Validate that all the pulses inside match does not have HW oscillator switching
    let state = match node.data() {
        ir::NodeKind::Match(_) => {
            *node.offset_mut() += delay;
            None
        }
        ir::NodeKind::Case(ob) => {
            if state.is_some() {
                return Err(anyhow!("Match cases cannot be nested.").into());
            }
            Some(ob.state)
        }
        _ => None,
    };
    for child in node.iter_children_mut() {
        handle_match(child, delay, state)?;
    }
    Ok(())
}

pub fn handle_match_nodes(node: &mut ir::IrNode, delay: &ir::Samples) -> Result<()> {
    handle_match(node, delay, None)
}
