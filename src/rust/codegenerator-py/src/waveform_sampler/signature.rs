// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::ir::compilation_job::Marker;
use codegenerator::ir::experiment::PulseParametersId;
use codegenerator::signature::{PulseSignature, WaveformSignature};
use pyo3::prelude::*;
use std::collections::HashMap;

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
    pulse_defs: &HashMap<String, Py<PyAny>>,
    pulse_parameters: &HashMap<PulseParametersId, Py<PulseParametersPy>>,
) -> WaveformSamplingDescPy {
    let pulses = if let WaveformSignature::Pulses { pulses, .. } = waveform {
        pulses
            .iter()
            .filter_map(|sig| {
                create_pulse_description(py, sig, pulse_defs, pulse_parameters)
                    .map(|desc| Py::new(py, desc).unwrap())
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
    pulse_defs: &HashMap<String, Py<PyAny>>,
    pulse_parameters: &HashMap<PulseParametersId, Py<PulseParametersPy>>,
) -> Option<PulseSamplingDescPy> {
    // If there is no pulse associated with the signature, no sampling needed
    let pulse_uid = if let Some(p) = &pulse.pulse {
        &p.uid
    } else {
        return None;
    };
    PulseSamplingDescPy {
        pulse: pulse_defs.get(pulse_uid.as_str()).unwrap().clone_ref(py),
        start: pulse.start,
        length: pulse.length,
        amplitude: pulse.amplitude.unwrap_or(0.0),
        oscillator_frequency: pulse.oscillator_frequency,
        phase: pulse.phase,
        channel: pulse.channel,
        markers: create_marker_description(py, &pulse.markers, pulse_defs),
        pulse_parameters: pulse
            .id_pulse_params
            .map(|uid| pulse_parameters.get(&uid).unwrap().clone_ref(py)),
    }
    .into()
}

fn create_marker_description(
    py: Python,
    markers: &[Marker],
    pulse_defs: &HashMap<String, Py<PyAny>>,
) -> Vec<Py<MarkerSamplingDescPy>> {
    markers
        .iter()
        .map(|m| {
            let desc_py = MarkerSamplingDescPy {
                marker_selector: m.marker_selector.clone(),
                enable: m.enable,
                start: m.start,
                length: m.length,
                pulse: m.pulse_id.as_ref().map(|pid| pulse_defs[pid].clone_ref(py)),
            };
            Py::new(py, desc_py).unwrap()
        })
        .collect()
}
