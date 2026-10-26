// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python DSL experiment -> Cap'n Proto `Experiment` serializer.

mod dsl_types;
mod py_helpers;
mod py_types;
mod serializer;

use capnp::any_pointer;
use pyo3::prelude::*;

use laboneq_error::LabOneQError;
use laboneq_opentelemetry_python::attach_otel_context;
use laboneq_tracing::{tracing_is_enabled, with_tracing};

use crate::py_types::ExperimentCapnpPy;

type Result<T, E = LabOneQError> = std::result::Result<T, E>;

/// Serialize a Python experiment object to Cap'n Proto bytes.
///
/// `setup_description` is the root of the target backend's own encoding of the device setup,
/// which the serializer forwards into an opaque pointer tagged with `backend`.
pub fn serialize_experiment(
    py: Python,
    experiment: &Bound<'_, PyAny>,
    backend: &str,
    setup_description: any_pointer::Reader<'_>,
    packed: bool,
) -> PyResult<Vec<u8>> {
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;
    with_tracing(|| {
        let experiment: ExperimentCapnpPy = experiment.extract()?;
        serializer::serialize_experiment(py, experiment, backend, setup_description, packed)
    })
    .map_err(PyErr::from)
}
