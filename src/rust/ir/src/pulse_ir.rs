// Copyright 2024 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use pyo3::{
    types::{PyComplex, PyDict},
    Py, PyAny, PyObject, Python,
};

use crate::{
    common::{py_deep_copy, RuntimeError},
    interval_ir::IntervalIr,
    DeepCopy,
};

#[derive(Debug)]
pub struct PulseIr {
    pub interval: Arc<Mutex<IntervalIr>>,

    pub pulse: PyObject,
    pub amplitude: Py<PyComplex>,
    pub amp_param_name: Option<String>,
    pub phase: f64,
    pub offset: i64,
    pub set_oscillator_phase: Option<f64>,
    pub increment_oscillator_phase: Option<f64>,
    pub incr_phase_param_name: Option<String>,
    pub section: String,
    pub play_pulse_params: Option<Py<PyDict>>,
    pub pulse_pulse_params: Option<Py<PyDict>>,
    pub is_acquire: bool,
    pub markers: Option<Py<PyAny>>,
}

impl DeepCopy for PulseIr {
    fn deep_copy(&self) -> Result<Self, RuntimeError> {
        let guard = self.interval.lock().map_err(|_| RuntimeError::Lock())?;

        let pulse_clone = py_deep_copy(&self.pulse).map_err(|_| RuntimeError::PyhonDeepCop())?;

        let amplitude_clone =
            py_deep_copy(self.amplitude.as_any()).map_err(|_| RuntimeError::PyhonDeepCop())?;
        let amplitude_clone = Python::with_gil(|py| {
            amplitude_clone
                .extract::<Py<PyComplex>>(py)
                .map_err(|_| RuntimeError::PyhonExtraction())
        })?;

        let play_pulse_params_clone = match self.play_pulse_params {
            Some(ref play_pulse_params) => match py_deep_copy(play_pulse_params.as_any()) {
                Ok(play_pulse_params) => {
                    let play_pulse_params_dict = Python::with_gil(|py| {
                        play_pulse_params
                            .extract::<Py<PyDict>>(py)
                            .map_err(|_| RuntimeError::PyhonExtraction())
                    })?;
                    Some(play_pulse_params_dict)
                }
                Err(_) => return Err(RuntimeError::PyhonDeepCop()),
            },
            None => None,
        };

        let pulse_pulse_params_clone = match self.pulse_pulse_params {
            Some(ref pulse_pulse_params) => match py_deep_copy(pulse_pulse_params.as_any()) {
                Ok(pulse_pulse_params) => {
                    let pulse_pulse_params_dict = Python::with_gil(|py| {
                        pulse_pulse_params
                            .extract::<Py<PyDict>>(py)
                            .map_err(|_| RuntimeError::PyhonExtraction())
                    })?;
                    Some(pulse_pulse_params_dict)
                }
                Err(_) => return Err(RuntimeError::PyhonDeepCop()),
            },
            None => None,
        };

        Ok(Self {
            interval: Arc::new(Mutex::new(guard.deep_copy()?)),
            pulse: pulse_clone,
            amplitude: amplitude_clone,
            amp_param_name: self.amp_param_name.clone(),
            phase: self.phase,
            offset: self.offset,
            set_oscillator_phase: self.set_oscillator_phase,
            increment_oscillator_phase: self.increment_oscillator_phase,
            incr_phase_param_name: self.incr_phase_param_name.clone(),
            section: self.section.clone(),
            play_pulse_params: play_pulse_params_clone,
            pulse_pulse_params: pulse_pulse_params_clone,
            is_acquire: self.is_acquire,
            markers: None,
        })
    }
}
