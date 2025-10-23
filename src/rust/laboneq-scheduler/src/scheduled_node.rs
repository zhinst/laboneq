// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::rc::Rc;

use crate::{TinySample, ir::IrKind, schedule_info::ScheduleInfo};

pub type NodeRef = Rc<Node>;

/// A node that hold the scheduling information of an IR node and its children.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: IrKind,
    pub schedule: ScheduleInfo,
    pub children: Vec<NodeChild>,
}

impl Node {
    pub fn new(kind: IrKind, schedule: ScheduleInfo) -> Self {
        Self {
            kind,
            schedule,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, offset: TinySample, child: Node) {
        self.children.push(NodeChild {
            offset,
            node: Rc::new(child),
        });
    }

    pub fn make_mut(self: &mut Rc<Node>) -> &mut Self {
        Rc::make_mut(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeChild {
    pub offset: TinySample,
    pub node: NodeRef,
}
