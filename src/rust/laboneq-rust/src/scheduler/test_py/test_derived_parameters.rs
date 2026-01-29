// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Operation, Sweep};
use laboneq_dsl::types::{ParameterUid, SectionUid};
use pyo3::ffi::c_str;

use pyo3::prelude::*;

use crate::include_py_file;
use crate::scheduler::experiment::{DeviceSetup, Experiment};
use crate::scheduler::experiment_py_to_experiment;
use crate::scheduler::py_device::DevicePy;
use crate::scheduler::py_signal::SignalPy;

/// Test for derived parameter registration in experiments defined via calibration fields.
#[test]
fn test_derived_parameters_calibration() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = PyModule::from_code(py, py_testfile, c_str!(""), c_str!("")).unwrap();
        let exp_func = module
            .getattr("create_derived_param_experiment_calibration")
            .unwrap();
        let (experiment, _) = parse_experiment_output(exp_func);
        let id_store = &experiment.id_store;

        // Test that the derived parameter is registered in the experiment parameters
        let target_param_uid = ParameterUid(id_store.get("derived_param").unwrap());
        assert!(experiment.parameters.contains_key(&target_param_uid));

        // Test that the sweep contains the derived parameter, as well as the original parameter
        fn find_sweep<'a>(node: &'a ExperimentNode, target: &SectionUid) -> Option<&'a Sweep> {
            if let Operation::Sweep(sweep) = &node.kind
                && &sweep.uid == target
            {
                return Some(sweep);
            }
            for child in &node.children {
                if let Some(sweep) = find_sweep(child, target) {
                    return Some(sweep);
                }
            }
            None
        }

        let target_sweep =
            find_sweep(&experiment.root, &id_store.get("sweep").unwrap().into()).unwrap();
        assert_eq!(target_sweep.parameters.len(), 2); // Both original and derived
        assert!(target_sweep.parameters.contains(&target_param_uid));
    });
}

/// Test for derived parameter registration in experiments defined via operation fields.
#[test]
fn test_derived_parameters_operation_field() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = PyModule::from_code(py, py_testfile, c_str!(""), c_str!("")).unwrap();
        let exp_func = module
            .getattr("create_derived_param_experiment_operation_field")
            .unwrap();
        let (experiment, _) = parse_experiment_output(exp_func);
        let id_store = &experiment.id_store;
        let target_param_uid = ParameterUid(id_store.get("derived_param").unwrap());
        // Test that the derived parameter is registered in the experiment parameters
        assert!(experiment.parameters.contains_key(&target_param_uid));

        // Test that the sweep contains the derived parameter, as well as the original parameter
        fn find_sweep<'a>(node: &'a ExperimentNode, target: &SectionUid) -> Option<&'a Sweep> {
            if let Operation::Sweep(sweep) = &node.kind
                && &sweep.uid == target
            {
                return Some(sweep);
            }
            for child in &node.children {
                if let Some(sweep) = find_sweep(child, target) {
                    return Some(sweep);
                }
            }
            None
        }

        let target_sweep =
            find_sweep(&experiment.root, &id_store.get("sweep").unwrap().into()).unwrap();
        assert_eq!(target_sweep.parameters.len(), 2); // Both original and derived
        assert!(target_sweep.parameters.contains(&target_param_uid));
    });
}

/// Helper function to parse experiment and device setup from a Python function.
fn parse_experiment_output(exp_func: Bound<'_, PyAny>) -> (Experiment, DeviceSetup) {
    // Call the function to get experiment, signals, and devices
    let function_output = exp_func.call0().unwrap();

    let experiment = function_output.get_item(0).unwrap();
    let signals = function_output.get_item(1).unwrap();
    let signals = signals
        .try_iter()
        .unwrap()
        .map(|py_signal| {
            let binding: Bound<'_, PyAny> = py_signal.unwrap().clone();
            let signal = binding.cast::<SignalPy>().unwrap();
            signal.clone()
        })
        .collect::<Vec<_>>();
    let devices = function_output.get_item(2).unwrap();
    let devices = devices
        .try_iter()
        .unwrap()
        .map(|py_device| {
            let binding: Bound<'_, PyAny> = py_device.unwrap().clone();
            let signal = binding.cast::<DevicePy>().unwrap();
            signal.clone()
        })
        .collect::<Vec<_>>();
    experiment_py_to_experiment(&experiment, signals, devices).unwrap()
}
