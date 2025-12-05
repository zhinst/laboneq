// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::scheduler::{
    pulse::{PulseDef, PulseFunction, PulseKind},
    py_export::numeric_literal_to_py,
};

pub(super) fn pulse_def_to_py(
    py: Python,
    id_store: &NamedIdStore,
    pulse_def: &PulseDef,
) -> PyResult<Py<PyAny>> {
    let m = py.import("laboneq")?;
    let pulse_def_py_cls = m
        .getattr("data")
        .and_then(|m| m.getattr("compilation_job"))
        .and_then(|m| m.getattr("PulseDef"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("uid", id_store.resolve(pulse_def.uid).unwrap())?;
    kwargs.set_item(
        "amplitude",
        numeric_literal_to_py(py, &pulse_def.amplitude)?,
    )?;
    kwargs.set_item("can_compress", pulse_def.can_compress)?;

    match &pulse_def.kind {
        PulseKind::Functional(func) => {
            let function_name = match func.function {
                PulseFunction::Constant => "const",
                PulseFunction::Custom { ref function } => function.as_str(),
            };
            kwargs.set_item("function", function_name)?;
            kwargs.set_item("length", func.length.value())?;
            kwargs.set_item("samples", py.None())?;
        }
        PulseKind::Sampled(sampled) => {
            kwargs.set_item("samples", sampled.samples.clone_ref(py))?;
            kwargs.set_item("function", py.None())?;
            kwargs.set_item("length", py.None())?;
        }
        PulseKind::LengthOnly { length } => {
            kwargs.set_item("samples", py.None())?;
            kwargs.set_item("function", py.None())?;
            kwargs.set_item("length", length.value())?;
        }
    }
    let pulse_def_py = pulse_def_py_cls.call((), Some(&kwargs))?;
    Ok(pulse_def_py.into())
}
