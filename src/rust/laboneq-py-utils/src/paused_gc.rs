// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::intern;
use pyo3::prelude::*;

/// Holds Python's cyclic garbage collector off while it lives, if it was on.
///
/// Use it around bulk creation of Python container objects that stay alive, such as
/// decoding a large message into Python objects. The collections that the new objects
/// trigger find no garbage, yet they traverse the live objects, and the more objects
/// exist, the longer that takes.
///
/// Keep user code out of its scope. User code can create cycles that only the collector
/// frees, and it can switch the collector on or off itself.
pub struct PausedGc<'py>(Option<Bound<'py, PyModule>>);

impl<'py> PausedGc<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        let gc = py.import(intern!(py, "gc"))?;
        if !gc.call_method0(intern!(py, "isenabled"))?.is_truthy()? {
            return Ok(Self(None));
        }
        gc.call_method0(intern!(py, "disable"))?;
        Ok(Self(Some(gc)))
    }
}

impl Drop for PausedGc<'_> {
    fn drop(&mut self) {
        if let Some(gc) = &self.0
            && let Err(err) = gc.call_method0(intern!(gc.py(), "enable"))
        {
            err.write_unraisable(gc.py(), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    use super::PausedGc;

    #[test]
    fn test_restores_the_collector_state() {
        Python::attach(|py| {
            let gc = py.import("gc").unwrap();
            let enabled = || gc.call_method0("isenabled").unwrap().is_truthy().unwrap();

            gc.call_method0("enable").unwrap();
            {
                let _paused = PausedGc::new(py).unwrap();
                assert!(!enabled());
            }
            assert!(enabled());

            gc.call_method0("disable").unwrap();
            drop(PausedGc::new(py).unwrap());
            assert!(!enabled());

            gc.call_method0("enable").unwrap();
        });
    }
}
