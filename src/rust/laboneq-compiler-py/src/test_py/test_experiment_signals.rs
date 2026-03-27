// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::ffi::c_str;

use pyo3::prelude::*;

use crate::include_py_file;

/// Test for error handling when a signal referenced in the experiment is missing.
#[test]
fn test_missing_signal() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = PyModule::from_code(py, py_testfile, c_str!(""), c_str!("")).unwrap();
        let exp_func = module.getattr("create_missing_signal_experiment").unwrap();
        let py_result = exp_func.call0();

        let err_msg = "Signal 'q1/drive' is not available in the experiment definition. Available signals are: 'q0/drive'";
        let expected = py_result.err().unwrap().to_string();
        assert!(expected.contains(err_msg), "{} != {}", expected, err_msg);
    });
}
