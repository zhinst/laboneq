// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::AwgInfo;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::named_id::resolve_ids;
use laboneq_error::LabOneQError;
use laboneq_py_utils::experiment_ir::ExperimentIrPy;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod py_conversions;
use codegenerator::generate_code;
mod common_types;
mod ir_compat;
mod result;
mod settings;

use crate::ir_compat::ir_to_code_compat;
use crate::py_conversions::extract_awg;
use crate::result::FeedbackRegisterConfigPy;
use crate::result::MeasurementPy;
use crate::result::ResultSourcePy;
use crate::settings::code_generator_settings_from_dict;
use crate::waveform_sampler::WaveformSamplerPy;
use result::{AwgCodeGenerationResultPy, SampledWaveformPy, SeqCGenOutputPy};

mod waveform_sampler;

/// Convert LabOneQError to PyErr, while resolving [`NamedId`]s
pub(crate) fn to_pyerr(error: LabOneQError, id_store: &NamedIdStore) -> PyErr {
    error.to_pyerr(|s| resolve_ids(&s, id_store))
}

#[pyfunction(name = "generate_code")]
#[allow(clippy::too_many_arguments)]
fn generate_code_py(
    py: Python,
    ir_experiment: &ExperimentIrPy,
    // list[AwgInfo]
    awgs: &Bound<PyList>,
    // Dictionary with compiler settings
    settings: &Bound<PyDict>,
) -> PyResult<SeqCGenOutputPy> {
    let id_store = &ir_experiment.inner.id_store;
    let settings = code_generator_settings_from_dict(settings)?;

    let awg_infos = awgs
        .try_iter()?
        .map(|awg| extract_awg(&awg.unwrap(), id_store).unwrap())
        .collect::<Vec<AwgInfo>>();

    let compat = ir_to_code_compat(ir_experiment, awg_infos).map_err(|e| to_pyerr(e, id_store))?;
    let sampler = WaveformSamplerPy::new(
        py,
        &ir_experiment.pulses,
        compat.acquisition_type.clone(),
        &compat.pulse_parameters,
        &ir_experiment.py_object_store,
        &ir_experiment.inner.id_store,
    );
    let result = py
        .detach(|| {
            generate_code(
                &compat.root,
                compat.awg_cores,
                compat.acquisition_type,
                settings,
                &sampler,
            )
        })
        .map_err(|e| to_pyerr(e, id_store))?;
    Python::attach(|py| {
        let result = SeqCGenOutputPy::new(py, result, id_store);
        Ok(result)
    })
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
