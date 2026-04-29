// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{Operation, Sweep};
use laboneq_dsl::types::{ParameterUid, SectionUid};
use pyo3::ffi::c_str;

use pyo3::prelude::*;

use crate::capnp_deserializer::deserialize_experiment;
use crate::include_py_file;
use crate::test_py::load_module;

/// Test for derived parameter registration in experiments defined via calibration fields.
#[test]
fn test_derived_parameters_calibration() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = load_module(py, py_testfile);
        let exp_func = module
            .getattr("create_derived_param_experiment_calibration")
            .unwrap();
        let capnp_data = exp_func.call0().unwrap();
        let deserialized =
            deserialize_experiment(py, capnp_data.extract::<&[u8]>().unwrap(), false).unwrap();
        let id_store = deserialized.id_store;

        // Test that the derived parameter is registered in the experiment parameters
        let target_param_uid = ParameterUid(id_store.get("derived_param").unwrap());
        assert!(deserialized.parameters.contains_key(&target_param_uid));

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
            find_sweep(&deserialized.root, &id_store.get("sweep").unwrap().into()).unwrap();
        assert_eq!(target_sweep.parameters.len(), 2); // Both original and derived
        assert!(target_sweep.parameters.contains(&target_param_uid));
    });
}

/// Test for derived parameter registration in experiments defined via operation fields.
#[test]
fn test_derived_parameters_operation_field() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = load_module(py, py_testfile);
        let exp_func = module
            .getattr("create_derived_param_experiment_operation_field")
            .unwrap();
        let capnp_data = exp_func.call0().unwrap();
        let deserialized =
            deserialize_experiment(py, capnp_data.extract::<&[u8]>().unwrap(), false).unwrap();
        let id_store = deserialized.id_store;

        let target_param_uid = ParameterUid(id_store.get("derived_param").unwrap());
        // Test that the derived parameter is registered in the experiment parameters
        assert!(deserialized.parameters.contains_key(&target_param_uid));

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
            find_sweep(&deserialized.root, &id_store.get("sweep").unwrap().into()).unwrap();
        assert_eq!(target_sweep.parameters.len(), 2); // Both original and derived
        assert!(target_sweep.parameters.contains(&target_param_uid));
    });
}
