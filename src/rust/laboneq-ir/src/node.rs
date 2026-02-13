// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_units::tinysample::TinySamples;

use crate::IrKind;

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IrNode {
    pub kind: IrKind,
    pub length: TinySamples,
    pub children: Vec<NodeChild>,
}

impl IrNode {
    pub fn new(kind: IrKind, length: TinySamples) -> Self {
        Self {
            kind,
            length,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, offset: TinySamples, child: IrNode) {
        self.children.push(NodeChild {
            offset,
            node: child,
        });
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeChild {
    pub offset: TinySamples,
    pub node: IrNode,
}
