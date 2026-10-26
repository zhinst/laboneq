// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Writes `CompilationArtifactsQccs`'s Python `ArtifactsCodegen` object into
//! `CodegenArtifactsQccs`, the QCCS-specific capnp struct plugged into
//! `CompiledExperiment.artifacts` via [`ArtifactsWriter`].

use capnp::any_pointer;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};

use laboneq_capnp::artifacts::ArtifactsWriter;
use laboneq_capnp::convert::{checked_data, checked_text, narrow};
use laboneq_error::{LabOneQError, WithContext};
use laboneq_qccs_capnp::payload::v1::payload_capnp::codegen_artifacts_qccs;

pub(crate) struct QccsArtifactsCapnpWriter<'a> {
    pub artifacts: &'a Py<PyAny>,
}

impl ArtifactsWriter for QccsArtifactsCapnpWriter<'_> {
    fn write(&self, builder: any_pointer::Builder<'_>) -> capnp::Result<()> {
        Python::attach(|py| {
            let (json, programs): (PyBackedBytes, Vec<SequencerProgram>) = py
                .import(intern!(py, "laboneq.qccs.serializers.artifacts"))
                .and_then(|m| {
                    m.call_method1(
                        intern!(py, "artifacts_to_capnp_parts"),
                        (self.artifacts.bind(py),),
                    )
                })
                .and_then(|parts| parts.extract())
                .map_err(LabOneQError::from)
                .with_context(|| "Failed to serialize QCCS artifacts")
                .map_err(err_to_capnp)?;

            let mut root: codegen_artifacts_qccs::Builder = builder.init_as();
            root.set_json(checked_data(&json, "json")?);
            write_programs(&programs, root)
        })
    }
}

/// An entry of `ArtifactsCodegen.src`.
#[derive(FromPyObject)]
struct SequencerProgram {
    #[pyo3(item("filename"))]
    uid: PyBackedStr,
    #[pyo3(item("text"))]
    src: PyBackedStr,
    #[pyo3(item("elf"), default)]
    bin: Option<PyBackedBytes>,
}

fn write_programs(
    programs: &[SequencerProgram],
    root: codegen_artifacts_qccs::Builder<'_>,
) -> capnp::Result<()> {
    let mut programs_builder = root.init_programs(narrow(programs.len(), "programs")?);
    for (i, program) in programs.iter().enumerate() {
        let mut program_builder = programs_builder.reborrow().get(i as u32);
        program_builder.set_uid(&*program.uid);
        program_builder.set_src(checked_text(&program.src, "programs.src")?);
        if let Some(bin) = &program.bin {
            program_builder.set_bin(checked_data(bin, "programs.bin")?);
        }
    }
    Ok(())
}

fn err_to_capnp(err: LabOneQError) -> capnp::Error {
    capnp::Error::failed(err.to_string())
}
