// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::mem;

#[derive(Debug)]
pub struct Node<S, T> {
    data: T,
    // Offset of the node relative to the context it is used in,
    // e.g. offset from the root or from the parent.
    offset: S,
    children: Vec<Node<S, T>>,
}

impl<S, T> Node<S, T> {
    pub fn new(data: T, offset: S) -> Self {
        Self {
            data,
            offset,
            children: Vec::new(),
        }
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    pub fn offset(&self) -> &S {
        &self.offset
    }

    pub fn offset_mut(&mut self) -> &mut S {
        &mut self.offset
    }

    pub fn iter_children(&self) -> impl DoubleEndedIterator<Item = &Node<S, T>> {
        self.children.iter()
    }

    pub fn iter_children_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Node<S, T>> {
        self.children.iter_mut()
    }

    pub fn take_children(&mut self) -> Vec<Node<S, T>> {
        std::mem::take(&mut self.children)
    }

    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn replace_data(&mut self, data: T) {
        self.data = data;
    }

    pub fn swap_data(&mut self, data: T) -> T {
        mem::replace(&mut self.data, data)
    }

    pub fn add_child_node(&mut self, node: Node<S, T>) {
        self.children.push(node);
    }

    pub fn add_child(&mut self, offset: S, data: T) {
        self.children.push(Node::new(data, offset));
    }

    pub fn insert_child(&mut self, index: usize, offset: S, data: T) {
        self.children.insert(index, Node::new(data, offset));
    }
}
