// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use awg_event::{AcquireEvent, AwgEvent, EventType, InitAmplitudeRegisterPy};
use codegenerator::ir::compilation_job::AwgCore;
use codegenerator::signature::WaveformSignature;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PySet;
use pyo3::wrap_pyfunction;
use signature::{PulseSignaturePy, WaveformSignaturePy};
use std::collections::HashSet;
mod code_generator;
mod py_conversions;
use codegenerator::generate_code;
use codegenerator::ir::{self, NodeKind};
mod awg_event;
mod result;
mod signature;

use codegenerator::Error as CodeError;
use result::AwgCodeGenerationResultPy;

use crate::code_generator::{
    SeqCGeneratorPy, SeqCTrackerPy, WaveIndexTrackerPy, merge_generators_py,
    seqc_generator_from_device_and_signal_type_py, string_sanitize_py,
};

import_exception!(laboneq.core.exceptions, LabOneQException);

fn translate_error(err: CodeError) -> PyErr {
    match err {
        CodeError::Anyhow(x) => LabOneQException::new_err(x.to_string()),
    }
}

/// Generate Python compatible AWG sampled events from the IR tree
fn generate_output(
    mut node: ir::IrNode,
    awg: &AwgCore,
    state: &mut Option<u16>,
    pos: &mut u64,
) -> Vec<awg_event::AwgEvent> {
    *pos += 1;
    match node.swap_data(ir::NodeKind::Nop { length: 0 }) {
        ir::NodeKind::PlayWave(ob) => {
            let end = node.offset() + ob.length();
            let hw_osc = match ob.oscillator {
                Some(uid) => {
                    let index = *awg.osc_allocation.get(&uid).expect("Missing index");
                    let out = signature::HwOscillator { uid, index };
                    Some(out)
                }
                None => None,
            };
            vec![awg_event::AwgEvent {
                start: *node.offset(),
                end,
                kind: awg_event::EventType::PlayWave(awg_event::PlayWaveEvent {
                    signals: ob.signals.iter().map(|sig| sig.uid.clone()).collect(),
                    waveform: signature::WaveformSignaturePy::new(ob.waveform),
                    state: *state,
                    hw_oscillator: hw_osc,
                    amplitude_register: ob.amplitude_register,
                    amplitude: ob.amplitude,
                    increment_phase: ob.increment_phase,
                    increment_phase_params: ob.increment_phase_params,
                }),
                position: None,
            }]
        }
        ir::NodeKind::Match(ob) => {
            let end = node.offset() + ob.length;
            let obj = awg_event::MatchEvent::from_ir(ob);
            let event = awg_event::AwgEvent {
                start: *node.offset(),
                end,
                kind: awg_event::EventType::Match(obj),
                position: None,
            };
            let mut out = vec![event];
            out.extend(
                node.take_children()
                    .into_iter()
                    .flat_map(|x| generate_output(x, awg, state, pos)),
            );
            out
        }
        ir::NodeKind::FrameChange(ob) => {
            let mut hw_osc = None;
            if awg.device_kind.traits().supports_oscillator_switching {
                if let Some(osc) = &ob.signal.oscillator {
                    let out = signature::HwOscillator {
                        uid: osc.uid.clone(),
                        index: *awg.osc_allocation.get(&osc.uid).expect("Missing index"),
                    };
                    hw_osc = Some(out);
                }
            }
            vec![awg_event::AwgEvent {
                start: *node.offset(),
                end: node.offset() + ob.length,
                position: Some(*pos),
                kind: awg_event::EventType::ChangeHwOscPhase(awg_event::ChangeHwOscPhase {
                    signal: ob.signal.uid.clone(),
                    phase: ob.phase,
                    hw_oscillator: hw_osc,
                    parameter: ob.parameter,
                }),
            }]
        }
        ir::NodeKind::Case(ob) => {
            *state = Some(ob.state);
            let out = node
                .take_children()
                .into_iter()
                .flat_map(|x| generate_output(x, awg, state, pos))
                .collect();
            *state = None;
            out
        }
        NodeKind::ResetPrecompensationFilters(ob) => {
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + ob.length,
                position: None,
                kind: EventType::ResetPrecompensationFilters {
                    signature: WaveformSignaturePy::new(WaveformSignature::Pulses {
                        length: ob.length,
                        pulses: vec![],
                    }),
                },
            };
            vec![event]
        }
        NodeKind::PpcStep(ob) => {
            let start = *node.offset();
            let end = *node.offset() + ob.length;

            let start_event = AwgEvent {
                start,
                end: start,
                position: None,
                kind: EventType::PpcSweepStepStart(awg_event::PpcSweepStepStart {
                    pump_power: ob.sweep_command.pump_power,
                    pump_frequency: ob.sweep_command.pump_frequency,
                    probe_power: ob.sweep_command.probe_power,
                    probe_frequency: ob.sweep_command.probe_frequency,
                    cancellation_phase: ob.sweep_command.cancellation_phase,
                    cancellation_attenuation: ob.sweep_command.cancellation_attenuation,
                }),
            };
            let end_event = AwgEvent {
                start: end,
                end,
                position: None,
                kind: EventType::PpcSweepStepEnd(),
            };
            vec![start_event, end_event]
        }
        NodeKind::InitAmplitudeRegister(ob) => {
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset(),
                position: None,
                kind: EventType::InitAmplitudeRegister(InitAmplitudeRegisterPy::new(ob)),
            };
            vec![event]
        }
        NodeKind::Acquire(ob) => {
            let length: i64 = ob.length();
            let channels = ob.signal().channels.to_vec();
            let pulse_defs = ob.pulse_defs().iter().map(|x| x.uid.clone()).collect();
            let id_pulse_params = ob.id_pulse_params().to_vec();
            let oscillator_frequency = ob.oscillator_frequency();
            let event = AcquireEvent {
                signal_id: ob.signal().uid.clone(),
                pulse_defs,
                id_pulse_params,
                oscillator_frequency,
                channels,
            };
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset() + length,
                position: Some(*pos),
                kind: EventType::AcquireEvent(event),
            };
            vec![event]
        }
        _ => node
            .take_children()
            .into_iter()
            .flat_map(|x| generate_output(x, awg, state, pos))
            .collect(),
    }
}

// NOTE: When changing the API, update the stub in 'laboneq/_rust/codegenerator'
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generate_code_for_awg(
    py: Python,
    // SingleAwgIR
    ob: Py<PyAny>,
    // SignalIR
    signals: Py<PyAny>,
    cut_points: &Bound<PySet>,
    play_wave_size_hint: u16,
    play_zero_size_hint: u16,
    amplitude_resolution_range: u64,
    use_amplitude_increment: bool,
    phase_resolution_range: u64,
    global_delay_samples: ir::Samples,
) -> PyResult<AwgCodeGenerationResultPy> {
    let mut awg = py_conversions::extract_awg(&ob.bind(py).getattr("awg")?, signals.bind(py))?;
    if awg.signals.is_empty() {
        return Ok(AwgCodeGenerationResultPy::default());
    }

    let root = py_conversions::transform_py_ir(ob.bind(py), &awg.signals)?;
    if !root.has_children() {
        return Ok(AwgCodeGenerationResultPy::default());
    }
    let awg_node = generate_code::generate_code_for_awg(
        &root,
        &mut awg,
        cut_points.extract::<HashSet<ir::Samples>>()?,
        play_wave_size_hint,
        play_zero_size_hint,
        amplitude_resolution_range,
        use_amplitude_increment,
        phase_resolution_range,
        global_delay_samples,
    )
    .map_err(translate_error)?;
    let mut awg_events = generate_output(awg_node, &awg, &mut None, &mut 0);
    awg_event::sort_events(&mut awg_events);
    let output = AwgCodeGenerationResultPy::create(awg_events)?;
    Ok(output)
}

pub fn create_py_module<'a>(
    parent: &Bound<'a, PyModule>,
    name: &str,
) -> PyResult<Bound<'a, PyModule>> {
    let m = PyModule::new(parent.py(), name)?;
    m.add_function(wrap_pyfunction!(generate_code_for_awg, &m)?)?;
    m.add_class::<PulseSignaturePy>()?;
    m.add_class::<WaveformSignaturePy>()?;
    // Sampled event handler
    m.add_class::<WaveIndexTrackerPy>()?;
    m.add_class::<SeqCGeneratorPy>()?;
    m.add_class::<SeqCTrackerPy>()?;
    m.add_function(wrap_pyfunction!(
        seqc_generator_from_device_and_signal_type_py,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(merge_generators_py, &m)?)?;
    m.add_function(wrap_pyfunction!(string_sanitize_py, &m)?)?;
    // Result
    m.add_class::<AwgCodeGenerationResultPy>()?;
    Ok(m)
}
