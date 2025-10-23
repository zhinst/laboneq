// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for CompilerSettings

use pyo3::{prelude::*, types::PyDict};

use codegenerator::CodeGeneratorSettings;

/// Creates code generator settings from a Python dictionary
pub(crate) fn code_generator_settings_from_dict(
    ob: &Bound<PyDict>,
) -> PyResult<CodeGeneratorSettings> {
    let settings_py = ob.as_any();
    let hdawg_min_playwave_hint = settings_py
        .get_item("HDAWG_MIN_PLAYWAVE_HINT")?
        .extract::<u16>()?;
    let hdawg_min_playzero_hint = settings_py
        .get_item("HDAWG_MIN_PLAYZERO_HINT")?
        .extract::<u16>()?;

    let uhfqa_min_playwave_hint = settings_py
        .get_item("UHFQA_MIN_PLAYWAVE_HINT")?
        .extract::<u16>()?;
    let uhfqa_min_playzero_hint = settings_py
        .get_item("UHFQA_MIN_PLAYZERO_HINT")?
        .extract::<u16>()?;
    let shfsg_min_playwave_hint = settings_py
        .get_item("SHFSG_MIN_PLAYWAVE_HINT")?
        .extract::<u16>()?;
    let shfsg_min_playzero_hint = settings_py
        .get_item("SHFSG_MIN_PLAYZERO_HINT")?
        .extract::<u16>()?;
    let amplitude_resolution_bits = settings_py
        .get_item("AMPLITUDE_RESOLUTION_BITS")?
        .extract::<u64>()?;
    let phase_resolution_bits = settings_py
        .get_item("PHASE_RESOLUTION_BITS")?
        .extract::<u64>()?;
    let use_amplitude_increment = settings_py
        .get_item("USE_AMPLITUDE_INCREMENT")?
        .extract::<bool>()?;
    let emit_timing_comments = settings_py
        .get_item("EMIT_TIMING_COMMENTS")?
        .extract::<bool>()?;
    let shf_output_mute_min_duration = settings_py
        .get_item("SHF_OUTPUT_MUTE_MIN_DURATION")?
        .extract::<f64>()?;
    let out = CodeGeneratorSettings::new(
        hdawg_min_playwave_hint,
        hdawg_min_playzero_hint,
        shfsg_min_playwave_hint,
        shfsg_min_playzero_hint,
        uhfqa_min_playwave_hint,
        uhfqa_min_playzero_hint,
        amplitude_resolution_bits,
        phase_resolution_bits,
        use_amplitude_increment,
        emit_timing_comments,
        shf_output_mute_min_duration,
    );
    Ok(out)
}
