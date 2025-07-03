// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::signature::{self, WaveformSignaturePy};
use codegenerator::ir::{InitAmplitudeRegister, Match, ParameterOperation, Samples};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PlayWaveEvent {
    pub signals: HashSet<String>,
    #[pyo3(get)]
    pub waveform: WaveformSignaturePy,
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
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PlayHoldEvent {
    #[pyo3(get)]
    pub length: i64,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct AcquireEvent {
    #[pyo3(get)]
    pub signal_id: String,
    #[pyo3(get)]
    pub pulse_defs: Vec<String>,
    #[pyo3(get)]
    pub id_pulse_params: Vec<Option<usize>>,
    #[pyo3(get)]
    pub oscillator_frequency: f64,
    #[pyo3(get)]
    pub channels: Vec<u8>,
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
pub struct PpcSweepStepStart {
    #[pyo3(get)]
    pub pump_power: Option<f64>,
    #[pyo3(get)]
    pub pump_frequency: Option<f64>,
    #[pyo3(get)]
    pub probe_power: Option<f64>,
    #[pyo3(get)]
    pub probe_frequency: Option<f64>,
    #[pyo3(get)]
    pub cancellation_phase: Option<f64>,
    #[pyo3(get)]
    pub cancellation_attenuation: Option<f64>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct SetOscillatorFrequencyPy {
    obj: codegenerator::ir::OscillatorFrequencySweepStep,
}

impl SetOscillatorFrequencyPy {
    pub fn new(obj: codegenerator::ir::OscillatorFrequencySweepStep) -> Self {
        SetOscillatorFrequencyPy { obj }
    }
}

#[pymethods]
impl SetOscillatorFrequencyPy {
    #[getter]
    pub fn osc_index(&self) -> u16 {
        self.obj.osc_index
    }

    #[getter]
    pub fn iteration(&self) -> usize {
        self.obj.iteration
    }

    #[getter]
    pub fn start_frequency(&self) -> f64 {
        self.obj.parameter.start
    }

    #[getter]
    pub fn step_frequency(&self) -> f64 {
        self.obj.parameter.step
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.obj.parameter.count
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum EventType {
    PlayWave(PlayWaveEvent),
    PlayHold(PlayHoldEvent),
    Match(MatchEvent),
    ChangeHwOscPhase(ChangeHwOscPhase),
    InitAmplitudeRegister(InitAmplitudeRegisterPy),
    ResetPrecompensationFilters { signature: WaveformSignaturePy },
    AcquireEvent(AcquireEvent),
    PpcSweepStepStart(PpcSweepStepStart),
    PpcSweepStepEnd(),
    SetOscillatorFrequency(SetOscillatorFrequencyPy),
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
            EventType::ResetPrecompensationFilters { .. } => 4,
            EventType::AcquireEvent(_) => 5,
            EventType::PpcSweepStepStart(_) => 6,
            EventType::PpcSweepStepEnd() => 7,
            EventType::PlayHold(_) => 8,
            EventType::SetOscillatorFrequency(_) => 9,
        }
    }

    fn data(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.kind {
            EventType::PlayWave(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::Match(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::ChangeHwOscPhase(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::InitAmplitudeRegister(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::ResetPrecompensationFilters { signature } => {
                Ok(signature.clone().into_pyobject(py)?.into())
            }
            EventType::AcquireEvent(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::PpcSweepStepStart(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::PpcSweepStepEnd() => Ok(None::<()>.into_pyobject(py)?.into()),
            EventType::PlayHold(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::SetOscillatorFrequency(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
        }
    }
}

pub fn sort_events(events: &mut [AwgEvent]) {
    events.sort_by(|a, b| a.start.cmp(&b.start));
}
