// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};

use laboneq_dsl::signal_calibration::Precompensation;

use codegenerator::ir::compilation_job::MixerType;

pub(crate) fn mixer_type_to_py<'py>(
    py: Python<'py>,
    mixer_type: &MixerType,
) -> PyResult<Bound<'py, PyAny>> {
    let mixer_type_cls = py
        .import(intern!(py, "laboneq.core.types.enums"))?
        .getattr(intern!(py, "MixerType"))?;
    let mixer_type_str = match mixer_type {
        MixerType::IQ => "IQ",
        MixerType::UhfqaEnvelope => "UHFQA_ENVELOPE",
    };
    mixer_type_cls.getattr(mixer_type_str)
}

/// Convert a [`Precompensation`] into the dictionary form the artifacts use.
pub(crate) fn precompensation_to_py<'py>(
    py: Python<'py>,
    precomp: &Precompensation,
) -> PyResult<Bound<'py, PyDict>> {
    let high_pass = precomp
        .high_pass
        .as_ref()
        .map(|hp| -> PyResult<_> {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "timeconstant"), hp.timeconstant)?;
            Ok(d)
        })
        .transpose()?;

    let exponential: Option<Vec<Bound<'py, PyDict>>> = if precomp.exponential.is_empty() {
        None
    } else {
        Some(
            precomp
                .exponential
                .iter()
                .map(|exp| -> PyResult<_> {
                    let d = PyDict::new(py);
                    d.set_item(intern!(py, "timeconstant"), exp.timeconstant)?;
                    d.set_item(intern!(py, "amplitude"), exp.amplitude)?;
                    Ok(d)
                })
                .collect::<PyResult<_>>()?,
        )
    };

    let fir = precomp
        .fir
        .as_ref()
        .map(|fir| -> PyResult<_> {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "coefficients"), fir.coefficients.clone())?;
            Ok(d)
        })
        .transpose()?;

    let bounce = precomp
        .bounce
        .as_ref()
        .map(|bounce| -> PyResult<_> {
            let d = PyDict::new(py);
            d.set_item(intern!(py, "delay"), bounce.delay)?;
            d.set_item(intern!(py, "amplitude"), bounce.amplitude)?;
            Ok(d)
        })
        .transpose()?;

    let out = PyDict::new(py);
    out.set_item(intern!(py, "exponential"), exponential)?;
    out.set_item(intern!(py, "high_pass"), high_pass)?;
    out.set_item(intern!(py, "bounce"), bounce)?;
    out.set_item(intern!(py, "FIR"), fir)?;
    Ok(out)
}
