// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::{common::RuntimeError, deep_copy_ir_node, DeepCopy, IrNode};

#[derive(Debug, Clone, Default)]
pub struct IntervalIr {
    pub children: Vec<Arc<Mutex<IrNode>>>,
    pub length: Option<i64>,
    pub signals: HashSet<String>,
    pub children_start: Vec<i64>,
}

impl DeepCopy for IntervalIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let mut cloned_children: Vec<Arc<Mutex<IrNode>>> = Vec::new();
        for c in &self.children {
            let c_guard = c.lock().map_err(|_| RuntimeError::LockFailed())?;
            let c_clone = deep_copy_ir_node(&c_guard);
            match c_clone {
                Ok(c_clone) => cloned_children.push(Arc::new(Mutex::new(c_clone))),
                Err(err) => return Err(err),
            }
        }
        Ok(IntervalIr {
            children: cloned_children,
            length: self.length,
            signals: self.signals.clone(),
            children_start: self.children_start.clone(),
        })
    }
}
