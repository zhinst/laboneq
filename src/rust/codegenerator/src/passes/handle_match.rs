// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_error::bail;

use crate::Result;
use crate::ir::{IrNode, NodeKind};

fn handle_match(node: &IrNode, state: Option<u16>) -> Result<()> {
    let state = match node.data() {
        NodeKind::Case(ob) => {
            if state.is_some() {
                bail!("Match cases cannot be nested.");
            }
            Some(ob.state)
        }
        NodeKind::Section(_) => None,
        _ => {
            if state.is_some() && node.has_children() {
                bail!("No special sections permitted inside 'case()' blocks.",);
            }
            None
        }
    };
    for child in node.iter_children() {
        handle_match(child, state)?;
    }
    Ok(())
}

pub(crate) fn handle_match_nodes(node: &IrNode) -> Result<()> {
    handle_match(node, None)
}
