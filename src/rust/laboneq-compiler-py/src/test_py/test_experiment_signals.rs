// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::ffi::c_str;

use pyo3::prelude::*;

use crate::include_py_file;
use crate::test_py::load_module;

/// Test for error handling when a signal referenced in the experiment is missing.
#[test]
fn test_missing_signal() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = load_module(py, py_testfile);
        let exp_func = module.getattr("create_missing_signal_experiment").unwrap();
        let py_result = exp_func.call0();

        let err_msg = "Signal 'q1/drive' is not available in the experiment definition. Available signals are: 'q0/drive'";
        let expected = py_result.err().unwrap().to_string();
        assert!(expected.contains(err_msg), "{} != {}", expected, err_msg);
    });
}
