// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::{
    ExperimentNode,
    operation::{Operation, PulseParameterValue},
    types::{NumericLiteral, ValueOrParameter},
};
use pyo3::ffi::c_str;

use super::load_module;
use crate::capnp_deserializer::deserialize_experiment;
use pyo3::prelude::*;

use crate::include_py_file;

fn visit_node(node: &ExperimentNode, f: &mut impl FnMut(&ExperimentNode)) {
    f(node);
    for child in &node.children {
        visit_node(child, f);
    }
}

/// Test pulse parameter handling in Python to Rust Experiment conversion.
#[test]
fn test_pulse_parameters_handling() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = load_module(py, py_testfile);
        let run_experiment = module.getattr("run_experiment").unwrap();
        let capnp_data = run_experiment.call0().unwrap();

        let deserialized =
            deserialize_experiment(py, capnp_data.extract::<&[u8]>().unwrap(), false).unwrap();
        let id_store = deserialized.id_store;
        // Test sweep parameter collection
        let parameter = deserialized.parameters.iter().next().unwrap().1;
        assert_eq!(id_store.resolve(parameter.uid).unwrap(), "sweep_param123");

        // Test pulse parameters
        let mut asserted_sigma = false;
        let mut asserted_beta = false;

        visit_node(&deserialized.root, &mut |node: &ExperimentNode| {
            if let Operation::PlayPulse(play) = &node.kind {
                for (param_uid, param) in &play.parameters {
                    let param_name = id_store.resolve(*param_uid).unwrap();
                    match param_name {
                        "sigma" => {
                            if let PulseParameterValue::ValueOrParameter(value_or_param) = &param {
                                assert_eq!(
                                    *value_or_param,
                                    ValueOrParameter::Parameter(parameter.uid)
                                );
                                asserted_sigma = true;
                            }
                        }
                        "beta" => {
                            if let PulseParameterValue::ValueOrParameter(value_or_param) = &param {
                                assert_eq!(
                                    *value_or_param,
                                    ValueOrParameter::Value(NumericLiteral::Float(0.5))
                                );
                                asserted_beta = true;
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
        });
        assert!(
            asserted_sigma && asserted_beta,
            "No PlayPulse node found in the experiment"
        );
    });
}
