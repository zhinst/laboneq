// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::compilation_job::Marker;
use codegenerator::signature::{PulseSignature, WaveformSignature};
use pyo3::prelude::*;

use crate::waveform_sampler::WaveformSamplerPy;

use super::pulse_parameters::PulseParametersPy;

#[pyclass(name = "WaveformSamplingDesc", frozen)]
pub(crate) struct WaveformSamplingDescPy {
    #[pyo3(get)]
    pub length: i64,
    #[pyo3(get)]
    pub pulses: Vec<Py<PulseSamplingDescPy>>,
}

#[pyclass(name = "PulseSamplingDesc", frozen)]
pub(crate) struct PulseSamplingDescPy {
    #[pyo3(get)]
    pub pulse: Py<PyAny>,
    #[pyo3(get)]
    pub start: i64,
    #[pyo3(get)]
    pub length: i64,
    #[pyo3(get)]
    pub amplitude: f64,
    #[pyo3(get)]
    pub oscillator_frequency: Option<f64>,
    #[pyo3(get)]
    pub phase: f64,
    #[pyo3(get)]
    pub channel: Option<u16>,
    #[pyo3(get)]
    pub markers: Vec<Py<MarkerSamplingDescPy>>,
    #[pyo3(get)]
    pub pulse_parameters: Option<Py<PulseParametersPy>>,
}

#[pyclass(name = "MarkerSamplingDesc", frozen)]
pub(crate) struct MarkerSamplingDescPy {
    #[pyo3(get)]
    pub marker_selector: String,
    #[pyo3(get)]
    pub enable: bool,
    #[pyo3(get)]
    pub start: Option<f64>,
    #[pyo3(get)]
    pub length: Option<f64>,
    #[pyo3(get)]
    pub pulse: Option<Py<PyAny>>,
}

pub(crate) fn create_waveform_description(
    py: Python,
    waveform: &WaveformSignature,
    sampler: &WaveformSamplerPy<'_>,
) -> WaveformSamplingDescPy {
    let pulses = if let WaveformSignature::Pulses { pulses, .. } = waveform {
        pulses
            .iter()
            .filter_map(|sig| {
                create_pulse_description(py, sig, sampler).map(|desc| Py::new(py, desc).unwrap())
            })
            .collect()
    } else {
        vec![]
    };
    WaveformSamplingDescPy {
        length: waveform.length(),
        pulses,
    }
}

fn create_pulse_description(
    py: Python,
    pulse: &PulseSignature,
    sampler: &WaveformSamplerPy<'_>,
) -> Option<PulseSamplingDescPy> {
    // If there is no pulse associated with the signature, no sampling needed
    let pulse_uid = if let Some(p) = &pulse.pulse {
        &p.uid
    } else {
        return None;
    };
    PulseSamplingDescPy {
        pulse: sampler.pulse_def_py(py, pulse_uid),
        start: pulse.start,
        length: pulse.length,
        amplitude: pulse.amplitude.unwrap_or(0.0),
        oscillator_frequency: pulse.oscillator_frequency,
        phase: pulse.phase,
        channel: pulse.channel,
        markers: create_marker_description(py, &pulse.markers, sampler),
        pulse_parameters: pulse.id_pulse_params.map(|uid| {
            sampler
                .pulse_parameters_to_py(py, uid)
                .expect("Internal error: Failed to resolve pulse parameters.")
                .unbind()
        }),
    }
    .into()
}

fn create_marker_description(
    py: Python,
    markers: &[Marker],
    sampler: &WaveformSamplerPy<'_>,
) -> Vec<Py<MarkerSamplingDescPy>> {
    markers
        .iter()
        .map(|m| {
            let desc_py = MarkerSamplingDescPy {
                marker_selector: m.marker_selector.clone(),
                enable: m.enable,
                start: m.start,
                length: m.length,
                pulse: m.pulse_id.as_ref().map(|pid| sampler.pulse_def_py(py, pid)),
            };
            Py::new(py, desc_py).unwrap()
        })
        .collect()
}
