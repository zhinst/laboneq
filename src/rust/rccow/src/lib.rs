// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This crate provides [`RcCow`], a smart pointer for traversing and mutating a tree of
//! reference-counted nodes. For the most part, you can think of `RcCow<'a, T>` as a `&'a mut Rc<T>`.
//!
//! To mutate a node, call [`RcCow::promote()`]. This will clone the node and, recursively, its parents, all
//! the way to the root of the tree. The clone is skipped if the node is not actually shared
//! (that is, the reference count is 1).
//!
//! Look at the provided example.

use std::ops::Deref;
use std::rc::Rc;

/// Trait required to be implemented by the inner node type, so that `RcCow` can descend the tree.
pub trait GetChildren {
    fn get_child(&self, index: usize) -> &Rc<Self>;
    fn get_child_mut(&mut self, index: usize) -> &mut Rc<Self>;

    fn iter_children(&self) -> impl Iterator<Item = &Rc<Self>> + '_;
}

/// Smart pointer for copy-on-write tree traversal.
#[derive(Debug)]
pub enum RcCow<'a, T> {
    Root(&'a mut Rc<T>),
    Ref {
        node: &'a Rc<T>,
        parent: *mut RcCow<'a, T>,
        child_index: usize,
    },
}

impl<T> Deref for RcCow<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match &self {
            RcCow::Root(node) => node,
            RcCow::Ref { node, .. } => node,
        }
    }
}

impl<'a, T: Clone + GetChildren> RcCow<'a, T> {
    /// Access the given child wrapped in an `RcCow`.
    pub fn descend<'b>(&'b mut self, child_index: usize) -> RcCow<'b, T>
    where
        'a: 'b,
    {
        let parent_ptr: *mut RcCow<'a, T> = std::ptr::from_mut(self);

        // SAFETY: We are _decreasing_ the lifetime
        let parent_ptr: *mut RcCow<'b, T> = unsafe { std::mem::transmute(parent_ptr) };

        let node: &'b Rc<T> = self.get_child(child_index);
        RcCow::Ref {
            node,
            parent: parent_ptr,
            child_index,
        }
    }

    /// Iterate over all children.
    pub fn iter<'b>(&'b mut self) -> impl Iterator<Item = RcCow<'b, T>> + 'b {
        let parent_ptr: *mut RcCow<'a, T> = std::ptr::from_mut(self);

        // SAFETY: We are _decreasing_ the lifetime
        let parent_ptr: *mut RcCow<'b, T> = unsafe { std::mem::transmute(parent_ptr) };

        self.iter_children().enumerate().map({
            move |(child_index, node)| RcCow::Ref {
                node,
                parent: parent_ptr,
                child_index,
            }
        })
    }

    /// Create a new `RcCow` by anchoring it to the root of the tree.
    pub fn root(inner: &'a mut Rc<T>) -> RcCow<'a, T> {
        RcCow::Root(inner)
    }

    #[must_use]
    pub fn promote(&mut self) -> &mut T {
        match self {
            RcCow::Root(node) => Rc::make_mut(node),
            RcCow::Ref {
                node: _, // We DO NOT access this!
                parent,
                child_index,
            } => {
                // First, recursively promote the parent.
                // SAFETY: By not accessing `self.node` through the parent, the node is not aliased here.
                // No other (3rd party) ref to self can exist, by nature of `promote()` taking `&mut self`.
                // Also, while multiple children may each hold a *mut pointer to the parent, only
                // one of them at any time turns it into a reference, right here inside `promote()`.
                // That reference to the parent is short-lived and does not leak.
                let parent: &mut T = unsafe { (**parent).promote() };

                //Next, descend back into the child and store that in self.
                *self = RcCow::Root(parent.get_child_mut(*child_index));

                // Finally, borrow from the new self we just created.
                let RcCow::Root(node) = self else { panic!() };
                Rc::make_mut(node)
            }
        }
    }
}

#[cfg(test)]
mod tests;
