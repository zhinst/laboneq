// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Writes `CompilationArtifactsQccs`'s Python `ArtifactsCodegen` object into
//! `CodegenArtifactsQccs`, the QCCS-specific capnp struct plugged into
//! `CompiledExperiment.artifacts` via [`ArtifactsWriter`].

use capnp::any_pointer;
use pyo3::prelude::*;

use laboneq_capnp::artifacts::ArtifactsWriter;
use laboneq_error::{LabOneQError, WithContext};
use laboneq_qccs_capnp::payload::v1::payload_capnp::codegen_artifacts_qccs;

pub(crate) struct QccsArtifactsCapnpWriter<'a> {
    pub artifacts: &'a Py<PyAny>,
}

impl ArtifactsWriter for QccsArtifactsCapnpWriter<'_> {
    fn write(&self, builder: any_pointer::Builder<'_>) -> capnp::Result<()> {
        Python::attach(|py| {
            // Use QCCS artifact serialization code to convert the Python `ArtifactsCodegen` object into JSON bytes.
            let json_bytes: Vec<u8> = py
                .import("laboneq.serializers.implementations._models._artifacts_qccs")
                .and_then(|m| {
                    m.call_method1(
                        "artifacts_codegen_to_json_bytes",
                        (self.artifacts.bind(py),),
                    )
                })
                .and_then(|obj| obj.extract())
                .map_err(LabOneQError::from)
                .with_context(|| "Failed to serialize QCCS artifacts to JSON")
                .map_err(err_to_capnp)?;

            let mut root: codegen_artifacts_qccs::Builder = builder.init_as();
            root.set_json(&json_bytes);
            Ok(())
        })
    }
}

fn err_to_capnp(err: LabOneQError) -> capnp::Error {
    capnp::Error::failed(err.to_string())
}
