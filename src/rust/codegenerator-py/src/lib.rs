// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::IrNode;
use codegenerator::ir::compilation_job::AwgCore;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use signature::{PulseSignaturePy, WaveformSignaturePy};
use std::vec;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod py_conversions;
mod waveform_sampler;
use codegenerator::generate_code;
mod common_types;
mod pulse_parameters;
mod result;
mod settings;
mod signature;
use crate::pulse_parameters::PulseParameters;
use crate::result::FeedbackRegisterConfigPy;
use crate::settings::code_generator_settings_from_dict;
use codegenerator::ir::experiment::PulseParametersId;
use result::{AwgCodeGenerationResultPy, SampledWaveformPy, SeqCGenOutputPy};

mod error;
use crate::error::Result;
use crate::waveform_sampler::WaveformSamplerPy;
use std::collections::HashMap;

fn transform_ir_and_awg(
    ir_tree: &Bound<PyAny>,
    awgs: &Bound<PyList>,
) -> Result<(
    IrNode,
    Vec<AwgCore>,
    HashMap<PulseParametersId, PulseParameters>,
)> {
    let root_ir = ir_tree.getattr("root")?;
    let ir_signals = ir_tree.getattr("signals")?;
    let mut awg_cores = vec![];
    for awg in awgs.try_iter()? {
        let mut awg = py_conversions::extract_awg(&awg?, &ir_signals)?;
        // Sort the signals for deterministic ordering
        awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
        awg_cores.push(awg);
    }
    let (root, pulse_parameters) = py_conversions::transform_py_ir(&root_ir, &awg_cores)?;
    Ok((root, awg_cores, pulse_parameters))
}

// NOTE: When changing the API, update the stub in 'laboneq/_rust/codegenerator'
#[pyfunction(name = "generate_code")]
#[allow(clippy::too_many_arguments)]
fn generate_code_py(
    py: Python,
    // IRTree
    ir: &Bound<PyAny>,
    // list[AwgInfo]
    awgs: &Bound<PyList>,
    feedback_register_layout: &Bound<PyDict>,
    acquisition_type: &Bound<'_, PyAny>,
    // Dictionary with compiler settings
    settings: &Bound<PyDict>,
    waveform_sampler: Py<PyAny>,
) -> Result<SeqCGenOutputPy> {
    let settings = code_generator_settings_from_dict(settings)?;
    let acquisition_type = py_conversions::extract_acquisition_type(acquisition_type)?;
    let feedback_register_layout =
        py_conversions::extract_feedback_register_layout(feedback_register_layout)?;
    let (ir_root, awgs, pulse_parameters) = transform_ir_and_awg(ir, awgs)?;
    let sampler = WaveformSamplerPy::new(&waveform_sampler, &pulse_parameters);
    let result = py.detach(|| {
        generate_code(
            &ir_root,
            &awgs,
            &acquisition_type,
            &feedback_register_layout,
            settings,
            &sampler,
        )
    })?;
    Python::attach(|py| {
        let result = SeqCGenOutputPy::new(py, result);
        Ok(result)
    })
}

pub fn create_py_module<'a>(py: Python<'a>, name: &str) -> PyResult<Bound<'a, PyModule>> {
    let m = PyModule::new(py, name)?;
    // Common types
    // Move up the compiler stack as we need the common types
    m.add_class::<common_types::SignalTypePy>()?;
    m.add_class::<common_types::DeviceTypePy>()?;
    m.add_class::<common_types::MixerTypePy>()?;
    // AWG Code generation
    m.add_function(wrap_pyfunction!(generate_code_py, &m)?)?;
    m.add_class::<PulseSignaturePy>()?;
    m.add_class::<WaveformSignaturePy>()?;
    // Waveform sampling
    m.add_class::<PlaySamplesPy>()?;
    m.add_class::<PlayHoldPy>()?;
    m.add_class::<SampledWaveformPy>()?;
    // Result
    m.add_class::<AwgCodeGenerationResultPy>()?;
    m.add_class::<FeedbackRegisterConfigPy>()?;
    Ok(m)
}
