// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! The Python leg of the compilation.
//!
//! Everything that crosses into `laboneq.compiler.workflow.compiler` is confined
//! to this module: the pass holds its Python operands as GIL-independent
//! [`Py`] handles and attaches to the interpreter only for the duration of a
//! [`CompilePass::compile`] call, so callers drive chunking without a
//! [`Python`] token of their own.
//!
//! TODO: Remove once the compilation workflow is driven from Rust.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{PyTypeInfo, intern};

use std::sync::Arc;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_error::{
    LabOneQError, PyErrorWithContext, ResourceLimitationError, WithContext, bail_resource_usage,
};
use laboneq_ir::pulse_sheet_schedule::PulseSheetSchedule;
use laboneq_ir::signal::Signal;

use crate::compiler_backend::CompilerBackend;
use crate::execution::Execution;
use crate::experiment::Experiment;
use crate::py_execution::create_py_execution;
use crate::py_experiment::ExperimentPy;
use crate::py_pulse_sheet_schedule::PulseSheetSchedulePy;

/// One run of the Python compilation workflow, in this backend's own types.
pub(crate) struct CompilePassOutput<C> {
    pub(crate) combined_output: C,
    pub(crate) pulse_sheet_schedule: Option<PulseSheetSchedule>,
}

/// The Python compilation workflow, bound to one experiment.
///
/// Built once and then run per chunk count, so the Python operands are
/// converted a single time no matter how often auto-chunking retries.
pub(crate) struct CompilePass<B: CompilerBackend> {
    backend: B,
    experiment: Py<ExperimentPy>,
    execution: Py<PyAny>,
    compiler_settings: Py<PyAny>,
}

impl<B: CompilerBackend> CompilePass<B> {
    /// Convert the compilation operands to their Python counterparts.
    pub(crate) fn new(
        backend: B,
        experiment: Experiment,
        signals: Arc<[Signal]>,
        backend_data: B::Output,
        compiler_settings: CompilerSettings,
        execution: &Execution,
    ) -> Result<Self, LabOneQError> {
        Python::attach(|py| {
            let settings_py = create_compiler_settings_py(py, &compiler_settings)?.unbind();
            let execution = create_py_execution(
                py,
                execution,
                &experiment.parameters,
                &experiment.id_store,
                &experiment.external_parameters,
            )?
            .unbind();
            let experiment = ExperimentPy {
                inner: experiment,
                signals,
                compiler_settings,
                backend_data: Arc::new(backend_data),
                backend: Arc::new(backend.clone()),
            };
            Ok(Self {
                backend,
                experiment: Py::new(py, experiment)?,
                execution,
                compiler_settings: settings_py,
            })
        })
    }

    /// Run the workflow, optionally splitting the sweep into `chunk_count` chunks.
    pub(crate) fn compile(
        &self,
        chunk_count: Option<u32>,
    ) -> Result<CompilePassOutput<B::CombinedOutput>, LabOneQError> {
        Python::attach(|py| {
            let output = self.call_compile(py, chunk_count)?;
            Ok(CompilePassOutput {
                combined_output: self.backend.combined_output_from_py(&output.artifacts)?,
                pulse_sheet_schedule: output
                    .schedule
                    .map(|obj| -> PyResult<_> {
                        Ok(obj
                            .extract::<PyRefMut<'_, PulseSheetSchedulePy>>()?
                            .take_inner())
                    })
                    .transpose()?,
            })
        })
    }

    fn call_compile<'py>(
        &self,
        py: Python<'py>,
        chunk_count: Option<u32>,
    ) -> Result<CompilationOutputPy<'py>, LabOneQError> {
        let compile_fn = py
            .import(intern!(py, "laboneq.compiler.workflow.compiler"))?
            .getattr(intern!(py, "compile_whole_or_with_chunks"))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "experiment"), &self.experiment)?;
        kwargs.set_item(intern!(py, "execution"), &self.execution)?;
        kwargs.set_item(intern!(py, "chunk_count"), chunk_count)?;
        kwargs.set_item(intern!(py, "backend"), self.backend.name())?;
        kwargs.set_item(intern!(py, "compiler_settings"), &self.compiler_settings)?;

        match compile_fn.call((), Some(&kwargs)) {
            Ok(result) => Ok(result.extract()?),
            Err(e) if e.is_instance(py, &ResourceLimitationError::type_object(py)) => {
                let usage = e
                    .value(py)
                    .getattr(intern!(py, "usage"))
                    .ok()
                    .and_then(|v: Bound<'_, PyAny>| v.extract().ok());
                bail_resource_usage!("{}", e.to_string(), usage = usage);
            }
            Err(e) => Err(rebuild_error_chain(py, e)),
        }
    }
}

/// Re-root a Python exception raised on top of a Rust error.
///
/// The error chain is Python (the workflow) - Rust (rt compiler) - Python (pulse
/// sampler). Dropping the outermost Python cause turns it into Rust - Python
/// (pulse sampler), so errors raised by custom Python pulse sampling functions
/// are reported properly.
///
/// TODO: Python is removed as middle man, this workaround is no longer needed.
fn rebuild_error_chain(py: Python<'_>, error: PyErr) -> LabOneQError {
    let Some(cause) = error.cause(py) else {
        return error.into();
    };
    error.set_cause(py, None);
    let mut cause: PyErrorWithContext = cause.into();
    cause.add_context(|| error.to_string());
    LabOneQError::PythonException(cause)
}

fn create_compiler_settings_py<'py>(
    py: Python<'py>,
    compiler_settings: &CompilerSettings,
) -> PyResult<Bound<'py, PyAny>> {
    let compiler_settings_py = py
        .import(intern!(py, "laboneq.compiler.common.compiler_settings"))?
        .getattr(intern!(py, "CompilerSettings"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "LOG_REPORT"), compiler_settings.log_report)?;
    kwargs.set_item(
        intern!(py, "IGNORE_RESOURCE_LIMITATION_ERRORS"),
        compiler_settings.ignore_resource_exhaustion,
    )?;
    compiler_settings_py.call((), Some(&kwargs))
}

/// The output of the Python function `compile_whole_or_with_chunks`.
#[derive(FromPyObject, Debug)]
struct CompilationOutputPy<'py> {
    artifacts: Bound<'py, PyAny>,
    schedule: Option<Bound<'py, PyAny>>,
}
