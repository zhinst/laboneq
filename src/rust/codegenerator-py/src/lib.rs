// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::CodeGeneratorSettings;
use codegenerator::generate_code;
use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::named_id::resolve_ids;
use laboneq_error::LabOneQError;
use laboneq_opentelemetry_python::attach_otel_context;
use laboneq_py_utils::experiment_ir::ExperimentIrPy;
use laboneq_tracing::tracing_is_enabled;
use laboneq_tracing::with_tracing;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod common_types;
mod result;

use codegenerator::ir_to_codegen_ir;

use crate::result::FeedbackRegisterConfigPy;
use crate::result::MeasurementPy;
use crate::result::ResultSourcePy;
use crate::waveform_sampler::WaveformSamplerPy;
use result::{AwgCodeGenerationResultPy, SampledWaveformPy, SeqCGenOutputPy};

mod waveform_sampler;

/// Convert LabOneQError to PyErr, while resolving [`NamedId`]s
pub(crate) fn to_pyerr(error: LabOneQError, id_store: &NamedIdStore) -> PyErr {
    error.to_pyerr(|s| resolve_ids(&s, id_store))
}

#[pyfunction(name = "generate_code")]
fn generate_code_py(py: Python, ir_experiment: &ExperimentIrPy) -> PyResult<SeqCGenOutputPy> {
    let _context_guard = tracing_is_enabled()
        .then(|| attach_otel_context(py))
        .transpose()?;
    with_tracing(|| generate_code_py_impl(py, ir_experiment))
}

fn generate_code_py_impl(py: Python, ir_experiment: &ExperimentIrPy) -> PyResult<SeqCGenOutputPy> {
    let id_store = &ir_experiment.inner.id_store;
    let codegen_ir = ir_to_codegen_ir(&ir_experiment.inner).map_err(|e| to_pyerr(e, id_store))?;
    let sampler = WaveformSamplerPy::new(
        py,
        &ir_experiment.inner.pulses,
        codegen_ir.acquisition_type.clone(),
        &codegen_ir.pulse_parameters,
        &ir_experiment.py_object_store,
        &ir_experiment.inner.id_store,
    );
    let settings = compiler_setting_to_codegenerator_settings(&ir_experiment.compiler_settings);
    let result = py
        .detach(|| generate_code(codegen_ir, &sampler, settings))
        .map_err(|e| to_pyerr(e, id_store))?;
    Python::attach(|py| {
        let result = SeqCGenOutputPy::new(py, result, id_store);
        Ok(result)
    })
}

fn compiler_setting_to_codegenerator_settings(
    compiler_settings: &CompilerSettings,
) -> CodeGeneratorSettings {
    CodeGeneratorSettings::new(
        compiler_settings.hdawg_min_playwave_hint,
        compiler_settings.hdawg_min_playzero_hint,
        compiler_settings.shfsg_min_playwave_hint,
        compiler_settings.shfsg_min_playzero_hint,
        compiler_settings.uhfqa_min_playwave_hint,
        compiler_settings.uhfqa_min_playzero_hint,
        compiler_settings.amplitude_resolution_bits,
        compiler_settings.phase_resolution_bits,
        compiler_settings.use_amplitude_increment,
        compiler_settings.emit_timing_comments,
        compiler_settings.shf_output_mute_min_duration,
        compiler_settings.ignore_resource_exhaustion,
    )
}

pub fn create_py_module<'a>(py: Python<'a>, name: &str) -> PyResult<Bound<'a, PyModule>> {
    use crate::result::ChannelPropertiesPy;

    let m = PyModule::new(py, name)?;
    // Common types
    // Move up the compiler stack as we need the common types
    m.add_class::<common_types::SignalTypePy>()?;
    m.add_class::<common_types::DeviceTypePy>()?;
    m.add_class::<common_types::MixerTypePy>()?;
    // AWG Code generation
    m.add_function(wrap_pyfunction!(generate_code_py, &m)?)?;
    // Waveform sampling
    m.add_class::<PlaySamplesPy>()?;
    m.add_class::<PlayHoldPy>()?;
    m.add_class::<SampledWaveformPy>()?;
    // Result
    m.add_class::<AwgCodeGenerationResultPy>()?;
    m.add_class::<FeedbackRegisterConfigPy>()?;
    m.add_class::<MeasurementPy>()?;
    m.add_class::<ResultSourcePy>()?;
    m.add_class::<ChannelPropertiesPy>()?;
    Ok(m)
}
