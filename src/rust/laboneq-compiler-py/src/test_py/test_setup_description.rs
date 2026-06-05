// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Roundtrip test for the ZQCS setup-description blob across the Python/Rust
//! Cap'n Proto boundary. A companion Python integration test
//! (`tests/zqcs/integration/test_setup_description_roundtrip.py`) verifies the
//! upstream half — controller-side mock SCM through to `compat.build_rs_experiment`.

use laboneq_dsl::device_setup::SetupDescription;
use pyo3::ffi::c_str;
use pyo3::prelude::*;

use crate::capnp_deserializer::deserialize_experiment;
use crate::include_py_file;
use crate::test_py::load_module;

#[test]
fn test_zqcs_setup_description_round_trip() {
    let py_testfile = include_py_file!("./test_dsl_experiment.py");
    Python::attach(|py| {
        let module = load_module(py, py_testfile);
        let expected = module
            .getattr("ZQCS_SETUP_DESCRIPTION_BLOB")
            .unwrap()
            .extract::<Vec<u8>>()
            .unwrap();
        let capnp_data = module
            .getattr("create_experiment_with_zqcs_setup_description")
            .unwrap()
            .call0()
            .unwrap();
        let deserialized =
            deserialize_experiment(py, capnp_data.extract::<&[u8]>().unwrap(), false).unwrap();

        let SetupDescription::Zqcs(zqcs) = &deserialized.setup_description else {
            panic!(
                "expected ZQCS setup description, got {:?}",
                deserialized.setup_description
            );
        };
        assert_eq!(zqcs.data, expected);
    });
}
