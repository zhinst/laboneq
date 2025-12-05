// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use std::collections::HashSet;

fn handle_loops_recursive(node: &ir::IrNode, cut_points: &mut HashSet<ir::Samples>) -> Result<()> {
    for child in node.iter_children() {
        if let ir::NodeKind::LoopIteration(data) = child.data() {
            cut_points.insert(*child.offset());
            cut_points.insert(*child.offset() + data.length);
        }
        handle_loops_recursive(child, cut_points)?;
    }
    Ok(())
}

pub(crate) fn handle_loops(node: &ir::IrNode, cut_points: &mut HashSet<ir::Samples>) -> Result<()> {
    handle_loops_recursive(node, cut_points)
}
