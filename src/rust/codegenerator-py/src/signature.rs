// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::signature::{PulseSignature, SamplesSignatureID, WaveformSignature};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};
use std::hash::Hash;

#[pyclass]
#[derive(Debug, Clone)]
pub struct HwOscillator {
    #[pyo3(get, set)]
    pub uid: String,
    #[pyo3(get, set)]
    pub index: u16,
}

/// Lightweight Python wrapper for [`PulseSignature`].
#[pyclass(name = "PulseSignature", frozen, hash, eq)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PulseSignaturePy {
    signature: PulseSignature,
}

impl PulseSignaturePy {
    pub fn new(ob: PulseSignature) -> Self {
        PulseSignaturePy { signature: ob }
    }
}

#[pymethods]
impl PulseSignaturePy {
    fn __deepcopy__(&self, py: Python, _memo: Py<PyAny>) -> PyResult<Py<Self>> {
        let new = PulseSignaturePy::new(self.signature.clone());
        Py::new(py, new)
    }

    #[getter]
    fn start(&self) -> i64 {
        self.signature.start
    }

    #[getter]
    fn pulse(&self) -> Option<&String> {
        self.signature.pulse.as_ref().map(|x| &x.uid)
    }

    #[getter]
    fn length(&self) -> i64 {
        self.signature.length
    }

    #[getter]
    fn amplitude(&self) -> Option<f64> {
        self.signature.amplitude
    }

    #[getter]
    fn phase(&self) -> f64 {
        self.signature.phase
    }

    #[getter]
    fn oscillator_frequency(&self) -> Option<f64> {
        self.signature.oscillator_frequency
    }

    #[getter]
    fn channel(&self) -> Option<u16> {
        self.signature.channel
    }

    #[getter]
    fn sub_channel(&self) -> Option<u8> {
        self.signature.sub_channel
    }

    #[getter]
    fn id_pulse_params(&self) -> Option<usize> {
        self.signature.id_pulse_params
    }

    #[getter]
    fn markers(&self, py: Python) -> PyResult<Py<PyTuple>> {
        let mut out: Vec<PyObject> = vec![];
        for marker in self.signature.markers.iter() {
            let d = PyDict::new(py);
            d.set_item("marker_selector", marker.marker_selector.clone())?;
            d.set_item("enable", marker.enable)?;
            d.set_item("start", marker.start)?;
            d.set_item("length", marker.length)?;
            d.set_item("pulse_id", marker.pulse_id.clone())?;
            out.push(d.into());
        }
        let out = PyTuple::new(py, out)?;
        Ok(out.into())
    }
}

/// Lightweight Python wrapper for [`WaveformSignature`].
#[pyclass(name = "WaveformSignature", frozen, hash, eq)]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct WaveformSignaturePy {
    waveform: WaveformSignature,
}

impl WaveformSignaturePy {
    pub fn new(ob: WaveformSignature) -> Self {
        WaveformSignaturePy { waveform: ob }
    }
}

#[pymethods]
impl WaveformSignaturePy {
    #[classmethod]
    #[pyo3(signature = (length, uid, label, has_i, has_q=None, has_marker1=None, has_marker2=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_samples_id(
        _cls: &Bound<'_, PyType>,
        length: i64,
        uid: u64,
        label: String,
        has_i: bool,
        has_q: Option<bool>,
        has_marker1: Option<bool>,
        has_marker2: Option<bool>,
    ) -> Self {
        let id = SamplesSignatureID {
            uid,
            label: label.to_string(),
            has_i,
            has_q: has_q.unwrap_or(false),
            has_marker1: has_marker1.unwrap_or(false),
            has_marker2: has_marker2.unwrap_or(false),
        };
        let waveform = WaveformSignature::Samples {
            length,
            samples_id: id,
        };
        WaveformSignaturePy { waveform }
    }

    fn __deepcopy__(&self, _memo: Py<PyAny>) -> Self {
        self.clone()
    }

    #[getter]
    fn length(&self) -> i64 {
        self.waveform.length()
    }

    #[getter]
    fn pulses(&self, py: Python) -> Vec<Py<PulseSignaturePy>> {
        if let WaveformSignature::Pulses { pulses, .. } = &self.waveform {
            return pulses
                .iter()
                .map(|sig| Py::new(py, PulseSignaturePy::new(sig.clone())).unwrap())
                .collect();
        }
        vec![]
    }

    fn is_playzero(&self) -> bool {
        self.waveform.is_playzero()
    }

    fn signature_string(&self) -> String {
        self.waveform.signature_string()
    }
}
