// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Entry points the backend binding crates wrap as `#[pyfunction]`s.
//!
//! These shims own the Python-side concerns — propagating the caller's OpenTelemetry
//! context and extracting Python arguments — and delegate the work to the pipeline.

use capnp::any_pointer;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use laboneq_opentelemetry_python::attach_otel_context;
use laboneq_tracing::{tracing_is_enabled, with_tracing};

use crate::capnp_py_types::ExperimentCapnpPy;
use crate::capnp_serializer;
use crate::compiler;
use crate::compiler_backend::CompilerBackend;
use crate::error::{Error, Result, create_error_message};
use crate::py_helpers;

/// Serialize a Python experiment object to Cap'n Proto bytes.
///
/// `setup_description` is the root of the target backend's own encoding of the device setup,
/// which the core serializer forwards into an opaque pointer tagged with `backend`.
pub fn serialize_experiment(
    py: Python,
    experiment: &Bound<'_, PyAny>,
    backend: &str,
    setup_description: any_pointer::Reader<'_>,
    packed: bool,
) -> Result<Vec<u8>> {
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;
    with_tracing(|| {
        let experiment: ExperimentCapnpPy = experiment.extract()?;
        capnp_serializer::serialize_experiment(py, experiment, backend, setup_description, packed)
    })
    .map_err(|e| Error::new(create_error_message(e)))
}

/// Compile an experiment from Cap'n Proto bytes plus device/signal configuration.
///
/// `packed` describes `capnp_data` only; the returned `CompiledExperiment` bytes are
/// always unpacked.
pub fn compile_experiment<B: CompilerBackend>(
    py: Python<'_>,
    capnp_data: &[u8],
    packed: bool,
    compiler_settings: Option<Bound<'_, PyDict>>,
    backend: B,
) -> PyResult<Vec<u8>> {
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;

    let compiler_settings =
        py_helpers::compiler_settings_from_py_dict(backend.settings(), compiler_settings.as_ref())?;

    with_tracing(|| {
        compiler::compile_experiment(py, capnp_data, packed, compiler_settings, backend)
    })
}
