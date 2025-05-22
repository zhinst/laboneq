// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use awg_event::{AwgEvent, EventType, InitAmplitudeRegisterPy};
use codegenerator::ir::compilation_job::AwgCore;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PySet;
use pyo3::wrap_pyfunction;
use signature::PulseSignature;
use std::collections::HashSet;
mod code_generator;
mod py_conversions;
use codegenerator::generate_code;
use codegenerator::ir::{self, NodeKind};
mod awg_event;
mod signature;
use codegenerator::Error as CodeError;

use crate::code_generator::{
    SeqCGeneratorPy, SeqCTrackerPy, WaveIndexTrackerPy, merge_generators_py,
    seqc_generator_from_device_and_signal_type_py,
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
            let pulses = ob.waveform.pulses.into_iter().map(PulseSignature::new);

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
                    pulses: pulses.collect(),
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
        NodeKind::InitAmplitudeRegister(ob) => {
            let event = AwgEvent {
                start: *node.offset(),
                end: *node.offset(),
                position: None,
                kind: EventType::InitAmplitudeRegister(InitAmplitudeRegisterPy::new(ob)),
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
) -> PyResult<Vec<awg_event::AwgEvent>> {
    let mut awg = py_conversions::extract_awg(&ob.bind(py).getattr("awg")?, signals.bind(py))?;
    if awg.signals.is_empty() {
        return Ok(vec![]);
    }

    let root = py_conversions::transform_py_ir(ob.bind(py), &awg.signals)?;
    if !root.has_children() {
        return Ok(vec![]);
    }
    let result = generate_code::generate_code_for_awg(
        &root,
        &mut awg,
        cut_points.extract::<HashSet<ir::Samples>>()?,
        play_wave_size_hint,
        play_zero_size_hint,
        amplitude_resolution_range,
        use_amplitude_increment,
        phase_resolution_range,
    );
    let mut out = match result {
        Ok(program) => generate_output(program, &awg, &mut None, &mut 0),
        Err(e) => {
            return Err(translate_error(e));
        }
    };
    awg_event::sort_events(&mut out);
    Ok(out)
}

pub fn create_py_module<'a>(
    parent: &Bound<'a, PyModule>,
    name: &str,
) -> PyResult<Bound<'a, PyModule>> {
    let m = PyModule::new(parent.py(), name)?;
    m.add_function(wrap_pyfunction!(generate_code_for_awg, &m)?)?;
    m.add_class::<WaveIndexTrackerPy>()?;
    m.add_class::<SeqCGeneratorPy>()?;
    m.add_class::<SeqCTrackerPy>()?;
    m.add_function(wrap_pyfunction!(
        seqc_generator_from_device_and_signal_type_py,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(merge_generators_py, &m)?)?;
    Ok(m)
}
