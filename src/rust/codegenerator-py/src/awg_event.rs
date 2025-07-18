// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use crate::signature::{self, WaveformSignaturePy};
use codegenerator::ir::{
    InitAmplitudeRegister, Match, ParameterOperation, PlayAcquire, PlayWave, QaEvent, Samples,
};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PlayWaveEventPy {
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

impl PlayWaveEventPy {
    pub fn from_ir(
        event: PlayWave,
        state: Option<u16>,
        hw_oscillator: Option<signature::HwOscillator>,
    ) -> Self {
        PlayWaveEventPy {
            signals: event.signals.iter().map(|sig| sig.uid.clone()).collect(),
            waveform: WaveformSignaturePy::new(event.waveform),
            state,
            hw_oscillator,
            amplitude_register: event.amplitude_register,
            amplitude: event.amplitude,
            increment_phase: event.increment_phase,
            increment_phase_params: event.increment_phase_params,
        }
    }
}

#[pymethods]
impl PlayWaveEventPy {
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

impl AcquireEvent {
    pub fn from_ir(event: PlayAcquire) -> Self {
        let channels = event.signal().channels.to_vec();
        let pulse_defs = event.pulse_defs().iter().map(|x| x.uid.clone()).collect();
        let id_pulse_params = event.id_pulse_params().to_vec();
        let oscillator_frequency = event.oscillator_frequency();
        AcquireEvent {
            signal_id: event.signal().uid.clone(),
            pulse_defs,
            id_pulse_params,
            oscillator_frequency,
            channels,
        }
    }
}

#[pymethods]
impl AcquireEvent {
    #[getter]
    fn channels(&self) -> Vec<i64> {
        self.channels.iter().map(|&ch| ch as i64).collect()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct QaEventPy {
    #[pyo3(get)]
    pub acquire_events: Vec<AcquireEvent>,
    #[pyo3(get)]
    pub play_wave_events: Vec<PlayWaveEventPy>,
}

impl QaEventPy {
    pub fn from_ir(event: QaEvent) -> Self {
        let (acquires, waveforms) = event.into_parts();
        QaEventPy {
            acquire_events: acquires.into_iter().map(AcquireEvent::from_ir).collect(),
            play_wave_events: waveforms
                .into_iter()
                .map(|wf| PlayWaveEventPy::from_ir(wf, None, None))
                .collect(),
        }
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
pub struct PushLoop {
    #[pyo3(get)]
    pub num_repeats: u64,
    pub compressed: bool,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Iterate {
    #[pyo3(get)]
    pub num_repeats: u64,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PrngSetup {
    #[pyo3(get)]
    pub range: u16,
    #[pyo3(get)]
    pub seed: u32,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PrngSample {
    #[pyo3(get)]
    pub sample_name: String,
    #[pyo3(get)]
    pub section_name: String,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PrngDropSample {
    #[pyo3(get)]
    pub sample_name: String,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct TriggerOutput {
    #[pyo3(get)]
    pub state: u16,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct TriggerOutputBit {
    pub bit: u8,
    pub set: bool,
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
    PlayWave(PlayWaveEventPy),
    PlayHold(PlayHoldEvent),
    Match(MatchEvent),
    ChangeHwOscPhase(ChangeHwOscPhase),
    InitAmplitudeRegister(InitAmplitudeRegisterPy),
    ResetPrecompensationFilters { signature: WaveformSignaturePy },
    AcquireEvent(AcquireEvent),
    PpcSweepStepStart(PpcSweepStepStart),
    PpcSweepStepEnd(),
    SetOscillatorFrequency(SetOscillatorFrequencyPy),
    ResetPhase(),
    InitialResetPhase(),
    LoopStepStart(),
    LoopStepEnd(),
    PushLoop(PushLoop),
    Iterate(Iterate),
    PrngSetup(PrngSetup),
    PrngSample(PrngSample),
    // todo: Only for assertions, to make sure sampling is not used outside
    // of a setup; consider testing already
    // when building the tree instead of creating a separate event.
    PrngDropSample(PrngDropSample),
    // This is a bit of a hack, but we need to be able to consolidate
    // the trigger output events after flattening the tree.
    // The TriggerOutputBit never appears in the final event list.
    TriggerOutputBit(TriggerOutputBit),
    TriggerOutput(TriggerOutput),
    QaEvent(QaEventPy),
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
            EventType::ResetPhase() => 8,
            EventType::InitialResetPhase() => 9,
            EventType::LoopStepStart() => 10,
            EventType::LoopStepEnd() => 11,
            EventType::PushLoop(_) => 12,
            EventType::Iterate(_) => 13,
            EventType::PrngSetup(_) => 14,
            EventType::PrngSample(_) => 15,
            EventType::PrngDropSample(_) => 16,
            EventType::TriggerOutput(_) => 17,
            EventType::TriggerOutputBit(_) => panic!("Internal error: Unresolved TriggerOutputBit"),
            EventType::PlayHold(_) => 18,
            EventType::SetOscillatorFrequency(_) => 19,
            EventType::QaEvent(_) => 20,
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
            EventType::LoopStepStart() => Ok(None::<()>.into_pyobject(py)?.into()),
            EventType::LoopStepEnd() => Ok(None::<()>.into_pyobject(py)?.into()),
            EventType::ResetPhase() => Ok(None::<()>.into_pyobject(py)?.into()),
            EventType::InitialResetPhase() => Ok(None::<()>.into_pyobject(py)?.into()),
            EventType::PushLoop(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::Iterate(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::PrngSetup(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::PrngSample(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::PrngDropSample(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::TriggerOutput(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::TriggerOutputBit(_) => {
                // At this point of the workflow, the single bits must have been
                // combined into words and replaced with TriggerOutput.
                panic!("Internal error: Unresolved TriggerOutputBit")
            }
            EventType::PlayHold(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::SetOscillatorFrequency(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
            EventType::QaEvent(ob) => Ok(ob.clone().into_pyobject(py)?.into()),
        }
    }
}

pub fn sort_events(events: &mut [AwgEvent]) {
    events.sort_by(|a, b| a.start.cmp(&b.start));
}
