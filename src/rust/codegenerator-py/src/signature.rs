// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use codegenerator::signature::PulseSignature as PulseSignatureRs;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

#[pyclass]
#[derive(Debug, Clone)]
pub struct HwOscillator {
    #[pyo3(get, set)]
    pub uid: String,
    #[pyo3(get, set)]
    pub index: u16,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PulseSignature {
    signature: PulseSignatureRs,
}

impl PulseSignature {
    pub fn new(ob: PulseSignatureRs) -> Self {
        PulseSignature { signature: ob }
    }
}

#[pymethods]
impl PulseSignature {
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
        self.signature.amplitude.as_ref().map(|value| value.re)
    }

    #[getter]
    fn phase(&self) -> f64 {
        self.signature.phase
    }

    #[getter]
    fn oscillator_phase(&self) -> Option<f64> {
        self.signature.oscillator_phase
    }

    #[getter]
    fn oscillator_frequency(&self) -> Option<f64> {
        self.signature.oscillator_frequency
    }

    #[getter]
    fn increment_oscillator_phase(&self) -> Option<f64> {
        self.signature.increment_oscillator_phase
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

    #[getter]
    fn incr_phase_params(&self, py: Python) -> PyResult<Py<PyTuple>> {
        let out = PyTuple::new(
            py,
            self.signature.incr_phase_params.iter().map(|s| s.as_str()),
        )?;
        Ok(out.into())
    }
}
