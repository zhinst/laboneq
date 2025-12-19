// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::ffi::c_str;

use pyo3::prelude::*;

use crate::include_py_file;
use crate::scheduler::build_experiment_py;

/// Test for error handling when a signal referenced in the experiment is missing.
#[test]
fn test_missing_signal() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = PyModule::from_code(py, py_testfile, c_str!(""), c_str!("")).unwrap();
        let exp_func = module.getattr("create_missing_signal_experiment").unwrap();
        let experiment_signals = exp_func.call0().unwrap();
        let experiment = experiment_signals.get_item(0).unwrap();
        let result = build_experiment_py(&experiment, vec![], vec![]);
        assert!(result.is_err());
        let err_msg = "Signal 'q1/drive' is not available in the experiment definition. Available signals are: 'q0/drive'";
        let err = result.err().unwrap();
        let expected = err.to_string();
        assert!(
            err.to_string().contains(err_msg),
            "{} != {}",
            expected,
            err_msg
        );
    });
}
