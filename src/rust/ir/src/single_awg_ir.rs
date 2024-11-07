// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use pyo3::{Py, PyAny};

use crate::{
    common::{py_deep_copy, RuntimeError},
    interval_ir::IntervalIr,
    DeepCopy,
};

#[derive(Debug)]
pub struct SingleAwgIr {
    pub interval: Arc<Mutex<IntervalIr>>,

    pub awg: Py<PyAny>,
}

impl DeepCopy for SingleAwgIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;
        let awg_copy = py_deep_copy(&self.awg).map_err(|_| RuntimeError::PyhonDeepCop())?;
        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
            awg: awg_copy,
        })
    }
}
