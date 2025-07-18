// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use crate::ir;
use log::warn;
use std::collections::HashSet;

pub fn handle_loops_recursive(
    node: &mut ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
    sample_multiple: u16,
    compressed: bool,
) -> Result<()> {
    let mut compressed = compressed;

    for child in node.iter_children_mut() {
        let offset = *child.offset();
        match child.data_mut() {
            ir::NodeKind::Loop(data) => {
                // todo: Remove this layer from tree
                compressed = data.compressed;
            }
            ir::NodeKind::LoopIteration(data) => {
                data.compressed = compressed;
                let start = offset;
                let end = start + data.length;
                if compressed && data.iteration == 0 && (end % sample_multiple as i64 != 0) {
                    warn!(
                        "Loop end time {end} is not divisible by sample multiple {sample_multiple}"
                    );
                }
                cut_points.insert(start);
                cut_points.insert(start + data.length);
                *child.offset_mut() = start;
            }
            _ => {}
        }
        handle_loops_recursive(child, cut_points, sample_multiple, compressed)?;
    }
    Ok(())
}

pub fn handle_loops(
    node: &mut ir::IrNode,
    cut_points: &mut HashSet<ir::Samples>,
    sample_multiple: u16,
) -> Result<()> {
    handle_loops_recursive(node, cut_points, sample_multiple, false)
}
