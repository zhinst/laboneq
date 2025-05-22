// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::signature;
use codegenerator::ir::{InitAmplitudeRegister, Match, ParameterOperation, Samples};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PlayWaveEvent {
    pub signals: HashSet<String>,
    pub pulses: Vec<signature::PulseSignature>,
    #[pyo3(get)]
    pub state: Option<u16>,
    #[pyo3(get)]
    pub hw_oscillator: Option<signature::HwOscillator>,
    #[pyo3(get)]
    pub amplitude_register: u16,
    pub amplitude: Option<ParameterOperation<f64>>,
    #[pyo3(get)]
    pub increment_phase: Option<f64>,
    #[pyo3(get)]
    pub increment_phase_params: Vec<Option<String>>,
}

#[pymethods]
impl PlayWaveEvent {
    #[getter]
    fn set_amplitude(&self) -> Option<f64> {
        if let Some(ParameterOperation::SET(amp)) = self.amplitude {
            return Some(amp);
        }
        None
    }

    #[getter]
    fn increment_amplitude(&self) -> Option<f64> {
        if let Some(ParameterOperation::INCREMENT(amp)) = self.amplitude {
            return Some(amp);
        }
        None
    }

    #[getter]
    fn signals(&self) -> PyResult<&HashSet<String>> {
        Ok(&self.signals)
    }

    #[getter]
    fn pulses(&self) -> PyResult<Vec<signature::PulseSignature>> {
        Ok(self.pulses.clone())
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct MatchEvent {
    #[pyo3(get)]
    pub handle: Option<String>,
    #[pyo3(get)]
    pub local: bool,
    #[pyo3(get)]
    pub user_register: Option<i64>,
    #[pyo3(get)]
    pub prng_sample: Option<String>,
    #[pyo3(get)]
    pub section: String,
}

impl MatchEvent {
    pub fn from_ir(event: Match) -> Self {
        MatchEvent {
            handle: event.handle,
            local: event.local,
            user_register: event.user_register,
            prng_sample: event.prng_sample,
            section: event.section,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ChangeHwOscPhase {
    #[pyo3(get)]
    pub signal: String,
    #[pyo3(get)]
    pub phase: f64,
    #[pyo3(get)]
    pub hw_oscillator: Option<signature::HwOscillator>,
    #[pyo3(get)]
    pub parameter: Option<String>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct InitAmplitudeRegisterPy {
    obj: InitAmplitudeRegister,
}

impl InitAmplitudeRegisterPy {
    pub fn new(obj: InitAmplitudeRegister) -> Self {
        InitAmplitudeRegisterPy { obj }
    }
}

#[pymethods]
impl InitAmplitudeRegisterPy {
    // The amplitude can be either SET or INCREMENT, but we are not modelling the amplitude enum
    // for Python at this moment so we have these 2 elegant methods.
    #[getter]
    fn set_amplitude(&self) -> Option<f64> {
        if let ParameterOperation::SET(amp) = self.obj.value {
            return Some(amp);
        }
        None
    }

    #[getter]
    fn increment_amplitude(&self) -> Option<f64> {
        if let ParameterOperation::INCREMENT(amp) = self.obj.value {
            return Some(amp);
        }
        None
    }

    #[getter]
    fn register(&self) -> u16 {
        self.obj.register
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum EventType {
    PlayWave(PlayWaveEvent),
    Match(MatchEvent),
    ChangeHwOscPhase(ChangeHwOscPhase),
    InitAmplitudeRegister(InitAmplitudeRegisterPy),
}

#[pyclass]
#[derive(Debug)]
pub struct AwgEvent {
    #[pyo3(get)]
    pub start: Samples,
    #[pyo3(get)]
    pub end: Samples,
    pub kind: EventType,
    #[pyo3(get)]
    pub position: Option<u64>,
}

#[pymethods]
impl AwgEvent {
    const fn event_type(&self) -> i64 {
        match self.kind {
            EventType::PlayWave(_) => 0,
            EventType::Match(_) => 1,
            EventType::ChangeHwOscPhase(_) => 2,
            EventType::InitAmplitudeRegister(_) => 3,
        }
    }

    fn data(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.kind {
            EventType::PlayWave(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::Match(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::ChangeHwOscPhase(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::InitAmplitudeRegister(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
        }
    }
}

pub fn sort_events(events: &mut [AwgEvent]) {
    events.sort_by(|a, b| a.start.cmp(&b.start));
}
