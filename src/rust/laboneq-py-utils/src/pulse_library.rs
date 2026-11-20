// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::BuiltinPulse;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

/// Module that defines LabOne Q standard functional pulses.
const PULSE_LIBRARY_MODULE: &str = "laboneq.dsl.experiment.pulse_library";

/// Returns the names of the standard pulses whose registered sampler is not a user replacement.
pub fn builtin_names(py: Python<'_>) -> PyResult<HashSet<&'static str>> {
    let pulse_samplers = py
        .import(intern!(py, "laboneq.core.utilities.pulse_sampler"))?
        .getattr(intern!(py, "pulse_function_library"))?
        .cast_into::<PyDict>()?;
    let mut builtin_names = HashSet::new();
    for (name, sampler) in pulse_samplers.iter() {
        let Some(pulse) = BuiltinPulse::from_name(name.extract::<&str>()?) else {
            continue;
        };
        let module: Option<String> = sampler.getattr(intern!(py, "__module__"))?.extract()?;
        if module.as_deref() == Some(PULSE_LIBRARY_MODULE) {
            builtin_names.insert(pulse.name());
        }
    }
    Ok(builtin_names)
}
