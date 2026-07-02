// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{intern, prelude::*};

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
