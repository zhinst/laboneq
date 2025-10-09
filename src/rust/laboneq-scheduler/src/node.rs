// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

pub type NodeRef<T, M> = Arc<Node<T, M>>;

#[derive(Debug, Clone)]
pub struct NodeChild<T, M> {
    node: NodeRef<T, M>,
    metadata: M,
}

impl<T, M> NodeChild<T, M> {
    pub fn node(&self) -> &NodeRef<T, M> {
        &self.node
    }

    pub fn metadata(&self) -> &M {
        &self.metadata
    }
}

/// A generic node in a tree.
///
/// `T` is the type of the node's kind.
/// `M` is the type of the metadata associated with each child node.
#[derive(Debug, Clone)]
pub struct Node<T, M = ()> {
    kind: T,
    children: Vec<NodeChild<T, M>>,
}

impl<T, M> Node<T, M> {
    pub fn new(kind: T) -> Self {
        Self {
            kind,
            children: Vec::new(),
        }
    }

    pub fn kind(&self) -> &T {
        &self.kind
    }

    pub fn add_child(&mut self, child: NodeRef<T, M>, metadata: M) {
        self.children.push(NodeChild {
            node: child,
            metadata,
        });
    }

    pub fn iter_children(&self) -> impl Iterator<Item = &NodeChild<T, M>> {
        self.children.iter()
    }
}
