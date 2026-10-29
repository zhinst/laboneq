// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use laboneq_common::near_time::NtStepKey;
use laboneq_compiler_py::compiler_backend::{CompilerBackendResult, Linker};
use laboneq_error::{LabOneQError, laboneq_error};

use crate::backend::{CodeGenArtifactQccs, QccsCombinedOutput};

/// A [`Linker`] for QCCS variants, backed by the Python `SeqCLinker`.
///
/// The first job starts the combined output. The first job of each later variant
/// merges the variant into the combined output, against the previous job's
/// variant. Every further job of a variant repeats the variant's run.
#[derive(Default)]
pub struct QccsLinker {
    /// The output linked so far, from the first job on.
    linked: Option<Linked>,
}

struct Linked {
    /// The `CombinedRTOutputSeqC`.
    combined: Py<PyAny>,
    /// The `SeqCGenOutput` of the latest job's variant.
    previous: Py<PyAny>,
}

/// A QCCS variant: its `SeqCGenOutput`.
pub struct QccsVariant {
    output: Py<PyAny>,
}

impl Linker for QccsLinker {
    type Artifact = CodeGenArtifactQccs;
    type Variant = QccsVariant;
    type Output = QccsCombinedOutput;

    fn add_artifact(
        &mut self,
        artifact: CodeGenArtifactQccs,
    ) -> CompilerBackendResult<QccsVariant> {
        Ok(QccsVariant {
            output: artifact.inner,
        })
    }

    fn add_job(
        &mut self,
        key: &NtStepKey,
        variant: &QccsVariant,
        first_run: bool,
    ) -> CompilerBackendResult<()> {
        Python::attach(|py| {
            self.link_job(py, key, &variant.output, first_run)
                .map_err(|e| LabOneQError::from_pyerr(py, e))
        })
    }

    fn finish(self) -> CompilerBackendResult<QccsCombinedOutput> {
        self.linked
            .map(|linked| QccsCombinedOutput(linked.combined))
            .ok_or_else(|| laboneq_error!("Internal error: No real-time job was linked"))
    }
}

impl QccsLinker {
    fn link_job(
        &mut self,
        py: Python<'_>,
        key: &NtStepKey,
        output: &Py<PyAny>,
        first_run: bool,
    ) -> PyResult<()> {
        static SEQC_LINKER: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        let linker = SEQC_LINKER.import(py, "laboneq.compiler.seqc.linker", "SeqCLinker")?;
        let previous = output.clone_ref(py);
        let output = output.bind(py);
        let step_indices = key.0.clone();
        match &mut self.linked {
            None => {
                let combined = linker.call_method1(
                    intern!(py, "combined_from_single_run"),
                    (output, step_indices),
                )?;
                self.linked = Some(Linked {
                    combined: combined.unbind(),
                    previous,
                });
            }
            Some(linked) => {
                let combined = linked.combined.bind(py);
                if first_run {
                    linker.call_method1(
                        intern!(py, "merge_combined_compiler_runs"),
                        (combined, output, linked.previous.bind(py), step_indices),
                    )?;
                } else {
                    linker.call_method1(
                        intern!(py, "repeat_run"),
                        (combined, output, step_indices),
                    )?;
                }
                linked.previous = previous;
            }
        }
        Ok(())
    }
}
