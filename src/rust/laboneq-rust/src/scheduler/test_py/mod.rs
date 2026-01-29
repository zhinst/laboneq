// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module contains tests for Python Experiment to Rust Experiment conversion.

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use std::ffi::CStr;

use crate::scheduler::create_py_module;
mod test_derived_parameters;
mod test_experiment_signals;
mod test_pulse_parameters;

#[macro_export]
macro_rules! include_py_file {
    ($path:literal) => {
        c_str!(include_str!($path))
    };
}

/// Loads a Python module from the given CStr contents.
///
/// The loaded module must use the `laboneq._rust.test_scheduler` module path to access the Rust
/// bindings for the scheduler package.
/// The package python bindings are made available in a separate module to avoid issues with
/// pyo3 type interpretations. The module is named `laboneq._rust.test_scheduler` and it shares
/// the same structure as the main `laboneq._rust.scheduler` module.
///
/// This is due to the fact and `laboneq._rust` is dynamically loaded and the binaries in the environment are not the same as the one used in tests.
/// This causes e.g. pyo3 types downcasting to fail and writing the manual conversion is cumbersome since each nested type must be handled explicitly.
pub(super) fn load_module<'py>(py: Python<'py>, module_contents: &CStr) -> Bound<'py, PyModule> {
    let py_modules = py.import("sys").unwrap().getattr("modules").unwrap();
    py_modules
        .set_item(
            "laboneq._rust.test_compiler",
            create_py_module(py, "test_compiler").unwrap(),
        )
        .unwrap();
    let module: Bound<'_, PyModule> =
        PyModule::from_code(py, module_contents, c_str!("testfile.py"), c_str!("")).unwrap();
    module
}
