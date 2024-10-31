// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use common::RuntimeError;
use loop_ir::LoopIr;
use section_ir::SectionIr;

mod common;

pub mod interval_ir;
pub mod loop_ir;
pub mod section_ir;

#[derive(Debug)]
pub enum IrNode {
    SectionIr(SectionIr),
    LoopIr(LoopIr),
}

pub fn deep_copy_ir_node(node: &IrNode) -> Result<IrNode, RuntimeError> {
    let node_copy: IrNode = match node {
        IrNode::SectionIr(section) => IrNode::SectionIr(section.deep_copy()?),
        IrNode::LoopIr(ir_loop) => IrNode::LoopIr(ir_loop.deep_copy()?),
    };
    Ok(node_copy)
}

pub trait DeepCopy {
    fn deep_copy(&self) -> Result<Self, RuntimeError>
    where
        Self: Sized;
}
