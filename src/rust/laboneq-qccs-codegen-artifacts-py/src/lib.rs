// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Cap'n Proto `CodegenArtifactsQccs` -> Python deserializer.
//!
//! `CodegenArtifactsQccs` is a thin wrapper around a JSON blob and the
//! sequencer programs rather than a structured schema, so only the programs
//! are decoded here; Python deserializes the JSON payload.

use capnp::any_pointer;
use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};

use laboneq_compiled_experiment_py::DeserializerContext;
use laboneq_error::{LabOneQError, WithContext, laboneq_error};
use laboneq_qccs_capnp::payload::v1::payload_capnp::{codegen_artifacts_qccs, sequencer_program};

/// Deserializes `CodegenArtifactsQccs` into a
/// `laboneq.data.artifacts_qccs.ArtifactsCodegen` Python object.
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
    let (json, programs) =
        read_parts(reader).map_err(|e| laboneq_error!("Malformed QCCS artifacts bytes: {e}"))?;
    let artifacts = py
        .import(intern!(py, "laboneq.qccs.serializers.artifacts"))?
        .call_method1(intern!(py, "artifacts_from_capnp_parts"), (json, programs))?;
    Ok(artifacts)
}

fn read_parts(
    reader: any_pointer::Reader<'_>,
) -> capnp::Result<(&[u8], Vec<SequencerProgram<'_>>)> {
    let root: codegen_artifacts_qccs::Reader<'_> = reader.get_as()?;
    let programs = root
        .get_programs()?
        .iter()
        .map(SequencerProgram::read)
        .collect::<capnp::Result<_>>()?;
    Ok((root.get_json()?, programs))
}

/// An entry of `ArtifactsCodegen.src`.
struct SequencerProgram<'a> {
    uid: &'a str,
    src: &'a str,
    bin: Option<&'a [u8]>,
}

impl<'a> SequencerProgram<'a> {
    fn read(program: sequencer_program::Reader<'a>) -> capnp::Result<Self> {
        Ok(Self {
            uid: program.get_uid()?.to_str()?,
            src: program.get_src()?.to_str()?,
            bin: if program.has_bin() {
                Some(program.get_bin()?)
            } else {
                None
            },
        })
    }
}

/// Converts into a `{"filename", "text", "elf"}` dict, without `elf` when there is no binary.
impl<'py> IntoPyObject<'py> for SequencerProgram<'_> {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let entry = PyDict::new(py);
        entry.set_item(intern!(py, "filename"), self.uid)?;
        entry.set_item(intern!(py, "text"), self.src)?;
        if let Some(bin) = self.bin {
            entry.set_item(intern!(py, "elf"), bin)?;
        }
        Ok(entry)
    }
}
