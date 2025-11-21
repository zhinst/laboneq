// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{Operation, ParameterUid, SectionUid, Sweep};
use pyo3::ffi::c_str;

use pyo3::prelude::*;

use crate::include_py_file;
use crate::scheduler::experiment_py_to_experiment;
use crate::scheduler::signal::SignalPy;

/// Test for derived parameters in experiments.
#[test]
fn test_derived_parameters() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = PyModule::from_code(py, py_testfile, c_str!(""), c_str!("")).unwrap();
        let exp_func = module.getattr("create_derived_param_experiment").unwrap();
        let experiment_signals = exp_func.call0().unwrap();
        let _experiment = experiment_signals.get_item(0).unwrap();
        let signals = experiment_signals.get_item(1).unwrap();
        let signals = signals
            .try_iter()
            .unwrap()
            .map(|py_signal| {
                let binding: Bound<'_, PyAny> = py_signal.unwrap().clone();
                let signal = binding.cast::<SignalPy>().unwrap();
                signal.clone()
            })
            .collect::<Vec<_>>();
        let experiment = experiment_py_to_experiment(&_experiment, signals).unwrap();
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

        let target_sweep = find_sweep(
            &experiment.sections[0],
            &id_store.get("sweep").unwrap().into(),
        )
        .unwrap();
        assert_eq!(target_sweep.parameters.len(), 2);
        assert!(target_sweep.parameters.contains(&target_param_uid));
    });
}
