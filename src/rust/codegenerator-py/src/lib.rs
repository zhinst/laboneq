// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use awg_event::{AcquireEvent, AwgEvent, EventType, InitAmplitudeRegisterPy};
use codegenerator::ir::compilation_job::AwgCore;
use codegenerator::signature::WaveformSignature;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PySet;
use pyo3::wrap_pyfunction;
use signature::{PulseSignaturePy, WaveformSignaturePy};
use std::collections::HashSet;
use waveform_sampler::PlayHoldPy;
use waveform_sampler::PlaySamplesPy;
mod code_generator;
mod py_conversions;
mod waveform_sampler;
use codegenerator::ir::{self, NodeKind};
use codegenerator::{AwgWaveforms, collect_and_finalize_waveforms, transform_ir_to_awg_events};
mod awg_event;
mod result;
mod signature;
use codegenerator::Error as CodeError;
mod settings;
use crate::settings::code_generator_settings_from_dict;
use result::{AwgCodeGenerationResultPy, SampledWaveformPy};
mod common_types;

use crate::awg_event::SetOscillatorFrequencyPy;
use crate::code_generator::{
    SeqCGeneratorPy, SeqCTrackerPy, WaveIndexTrackerPy, merge_generators_py,
    seqc_generator_from_device_and_signal_type_py, string_sanitize_py,
};
use crate::waveform_sampler::WaveformSamplerPy;

import_exception!(laboneq.core.exceptions, LabOneQException);

fn translate_error(py: Python, err: CodeError) -> PyErr {
    match err {
        CodeError::Anyhow(x) => LabOneQException::new_err(x.to_string()),
        CodeError::External(py_err) => {
            let original_cause = *py_err.downcast::<PyErr>().expect("Expected PyErr");
            if original_cause.is_instance_of::<LabOneQException>(py) {
                return original_cause;
            }
            let error = LabOneQException::new_err("Error when calling external code.");
            error.set_cause(py, Some(original_cause));
            error
        }
    }
}

/// Generate Python compatible AWG sampled events from the IR tree
fn generate_output(
    mut node: ir::IrNode,
    awg: &AwgCore,
    state: &mut Option<u16>,
    pos: &mut u64,
) -> Vec<AwgEvent> {
    *pos += 1;
    match node.swap_data(NodeKind::Nop { length: 0 }) {
        NodeKind::PlayWave(ob) => {
            let end = node.offset() + ob.length();
            let hw_osc = match ob.oscillator {
                Some(uid) => {
                    let index = *awg.osc_allocation.get(&uid).expect("Missing index");
                    let out = signature::HwOscillator { uid, index };
                    Some(out)
                }
                None => None,
            };
            vec![AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::PlayWave(awg_event::PlayWaveEvent {
                    signals: ob.signals.iter().map(|sig| sig.uid.clone()).collect(),
                    waveform: WaveformSignaturePy::new(ob.waveform),
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
        NodeKind::PlayHold(ob) => {
            let end = node.offset() + ob.length;
            vec![AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::PlayHold(awg_event::PlayHoldEvent { length: ob.length }),
                position: None,
            }]
        }
        NodeKind::Match(ob) => {
            let end = node.offset() + ob.length;
            let obj = awg_event::MatchEvent::from_ir(ob);
            let event = AwgEvent {
                start: *node.offset(),
                end,
                kind: EventType::Match(obj),
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
        NodeKind::FrameChange(ob) => {
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
            vec![AwgEvent {
                start: *node.offset(),
                end: node.offset() + ob.length,
                position: Some(*pos),
                kind: EventType::ChangeHwOscPhase(awg_event::ChangeHwOscPhase {
                    signal: ob.signal.uid.clone(),
                    phase: ob.phase,
                    hw_oscillator: hw_osc,
                    parameter: ob.parameter,
                }),
            }]
        }
        NodeKind::Case(ob) => {
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
        NodeKind::SetOscillatorFrequencySweep(ob) => {
            let start = *node.offset();
            let end = *node.offset() + ob.length;
            let mut awg_events = vec![];
            for osc in ob.oscillators.into_iter() {
                let event = AwgEvent {
                    start,
                    end,
                    position: None,
                    kind: EventType::SetOscillatorFrequency(SetOscillatorFrequencyPy::new(osc)),
                };
                awg_events.push(event);
            }
            awg_events
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
    // Compiler settings as Python dictionary
    settings: Py<PyDict>,
    cut_points: &Bound<PySet>,
    global_delay_samples: ir::Samples,
    waveform_sampler: Py<PyAny>,
) -> PyResult<AwgCodeGenerationResultPy> {
    let mut settings = code_generator_settings_from_dict(settings.bind(py))?;
    for msg in settings
        .sanitize()
        .map_err(|err| translate_error(py, err))?
    {
        log::warn!(
            "Compiler setting `{}` is sanitized from {} to {}. Reason: {}",
            msg.field.to_uppercase(),
            msg.original,
            msg.sanitized,
            msg.reason
        );
    }
    let mut awg = py_conversions::extract_awg(&ob.bind(py).getattr("awg")?, signals.bind(py))?;
    if awg.signals.is_empty() {
        return Ok(AwgCodeGenerationResultPy::default());
    }

    let root = py_conversions::transform_py_ir(ob.bind(py), &awg.signals)?;
    if !root.has_children() {
        return Ok(AwgCodeGenerationResultPy::default());
    }
    // Sort the signals for deterministic ordering
    awg.signals.sort_by(|a, b| a.channels.cmp(&b.channels));
    let mut awg_node = transform_ir_to_awg_events(
        root,
        &awg,
        cut_points.extract::<HashSet<ir::Samples>>()?,
        &settings,
        global_delay_samples,
    )
    .map_err(|err| translate_error(py, err))?;
    let waveforms = if WaveformSamplerPy::supports_waveform_sampling(&awg) {
        collect_and_finalize_waveforms(
            &mut awg_node,
            WaveformSamplerPy::new(waveform_sampler, &awg),
        )
    } else {
        Ok(AwgWaveforms::default())
    }
    .map_err(|err| translate_error(py, err))?;
    let mut awg_events = generate_output(awg_node, &awg, &mut None, &mut 0);
    awg_event::sort_events(&mut awg_events);
    let (sampled_waveforms, wave_declarations) = waveforms.into_inner();
    let output =
        AwgCodeGenerationResultPy::create(awg_events, sampled_waveforms, wave_declarations)?;
    Ok(output)
}

pub fn create_py_module<'a>(
    parent: &Bound<'a, PyModule>,
    name: &str,
) -> PyResult<Bound<'a, PyModule>> {
    let m = PyModule::new(parent.py(), name)?;
    // Common types
    // Move up the compiler stack as we need the common types
    m.add_class::<common_types::SignalTypePy>()?;
    m.add_class::<common_types::DeviceTypePy>()?;
    m.add_class::<common_types::MixerTypePy>()?;
    // AWG Code generation
    m.add_function(wrap_pyfunction!(generate_code_for_awg, &m)?)?;
    m.add_class::<PulseSignaturePy>()?;
    m.add_class::<WaveformSignaturePy>()?;
    // Waveform sampling
    m.add_class::<PlaySamplesPy>()?;
    m.add_class::<PlayHoldPy>()?;
    m.add_class::<SampledWaveformPy>()?;
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
