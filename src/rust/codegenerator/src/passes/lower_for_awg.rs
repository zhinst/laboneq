// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::ir;
use crate::ir::compilation_job as cjob;
use crate::node;
use crate::tinysample;

/// Convert tiny samples to unit samples on target AWG
pub fn convert_to_samples(node: &mut ir::IrNode, awg: &cjob::AwgCore) {
    *node.offset_mut() = tinysample::tinysample_to_samples(*node.offset(), awg.sampling_rate);
    let len_ts = node.data().length();
    node.data_mut()
        .set_length(tinysample::tinysample_to_samples(len_ts, awg.sampling_rate));
    for child in node.iter_children_mut() {
        convert_to_samples(child, awg);
    }
}

/// Convert relative node offsets to absolute values starting from the root
pub fn offset_to_absolute(node: &ir::IrNode, start: ir::Samples) -> ir::IrNode {
    let mut code_node = node::Node::new(node.data().clone(), start);
    for child in node.iter_children() {
        let code_child = offset_to_absolute(child, start + child.offset());
        code_node.add_child_node(code_child);
    }
    code_node
}
