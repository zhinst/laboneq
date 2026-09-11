// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Cap'n Proto `CodegenArtifactsQccs` -> Python deserializer.
//!
//! `CodegenArtifactsQccs` is a thin wrapper around a JSON blob
//! rather than a structured schema, so almost nothing is deserialized here,
//! only the JSON payload is deserialized into a Python object and returned.

use capnp::any_pointer;
use pyo3::types::PyBytes;
use pyo3::{intern, prelude::*};

use laboneq_compiled_experiment_py::DeserializerContext;
use laboneq_error::{LabOneQError, WithContext, laboneq_error};
use laboneq_qccs_capnp::payload::v1::payload_capnp::codegen_artifacts_qccs;

/// Deserializes `CodegenArtifactsQccs` into a
/// `laboneq.data.artifacts_qccs.ArtifactsCodegen` Python object.
///
/// Decodes the JSON payload's slice out of the message, then delegates the
/// actual deserialization to Python.
pub fn qccs_artifacts_from_capnp<'py>(
    py: Python<'py>,
    reader: any_pointer::Reader<'_>,
    _ctx: &DeserializerContext,
) -> Result<Bound<'py, PyAny>, LabOneQError> {
    qccs_artifacts_from_capnp_impl(py, reader)
        .with_context(|| "Failed to deserialize QCCS artifacts from Compiler output.")
}

fn qccs_artifacts_from_capnp_impl<'py>(
    py: Python<'py>,
    reader: any_pointer::Reader<'_>,
) -> Result<Bound<'py, PyAny>, LabOneQError> {
    const DESERIALIZER_MODULE_PY: &str =
        "laboneq.serializers.implementations._models._artifacts_qccs";
    const DESERIALIZER_FUNC_PY: &str = "artifacts_codegen_from_json_bytes";

    // Read the Cap'n Proto data and extract the JSON bytes.
    let root: codegen_artifacts_qccs::Reader<'_> = reader
        .get_as()
        .map_err(|e| laboneq_error!("Malformed QCCS artifacts bytes: {e}"))?;

    let json_bytes = root
        .get_json()
        .map_err(|e| laboneq_error!("Malformed QCCS artifacts bytes: {e}"))?;

    // Call the Python deserializer function with the JSON bytes.
    let module = py.import(intern!(py, DESERIALIZER_MODULE_PY))?;
    let artifacts_py =
        module.call_method1(DESERIALIZER_FUNC_PY, (PyBytes::new(py, json_bytes),))?;

    Ok(artifacts_py)
}
