// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::ffi::c_str;

use super::load_module;
use crate::scheduler::experiment_py_to_experiment;
use pyo3::prelude::*;

use crate::include_py_file;

#[test]
fn test_pulse_parameters() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = load_module(py, py_testfile);
        let run_experiment = module.getattr("run_experiment").unwrap();
        let experiment = run_experiment.call0().unwrap();

        // Test Experiment building
        let experiment = experiment_py_to_experiment(&experiment, vec![]).unwrap();
        let id_store = experiment.id_store;
        // Test sweep parameter collection
        let parameter = experiment.parameters.iter().next().unwrap().1;
        assert_eq!(id_store.resolve(parameter.uid).unwrap(), "sweep_param123");
        // Test experiment signals
        let experiment_signals: Vec<&str> = experiment
            .experiment_signals
            .iter()
            .map(|s_uid| id_store.resolve(*s_uid).unwrap())
            .collect();
        assert_eq!(experiment_signals[0], "q0/drive");
        // Test external pulse parameters
        // Sweep parameter is not external, therefore not stored
        assert_eq!(experiment.external_parameters.len(), 1);
        let external_pulse_parameter = experiment.external_parameters.values().next().unwrap();
        assert!(
            external_pulse_parameter
                .bind(py)
                .eq((0.5).into_pyobject(py).unwrap())
                .unwrap(),
        );
    });
}
