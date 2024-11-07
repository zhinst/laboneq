// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use pyo3::{Py, PyAny};

use crate::{
    common::{py_deep_copy, RuntimeError},
    interval_ir::IntervalIr,
    DeepCopy,
};

#[derive(Debug)]
pub struct LoopIr {
    pub interval: Arc<Mutex<IntervalIr>>,

    pub section: String,
    pub trigger_output: HashSet<(String, i64)>,
    pub prng_setup: Option<Py<PyAny>>,
    pub compressed: bool,
    pub iterations: i64,
}

impl DeepCopy for LoopIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let prng_clone = match self.prng_setup {
            Some(ref prng) => match py_deep_copy(prng) {
                Ok(prng) => Some(prng),
                Err(_) => return Err(RuntimeError::PyhonDeepCop()),
            },
            None => None,
        };

        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;
        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
            section: self.section.clone(),
            trigger_output: self.trigger_output.clone(),
            prng_setup: prng_clone,
            compressed: self.compressed,
            iterations: self.iterations,
        })
    }
}
