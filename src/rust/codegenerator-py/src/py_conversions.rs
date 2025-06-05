// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Translations from Python IR into code generator IR
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::{PyComplex, PyString};

use num_complex::Complex;

use crate::ir::experiment::SweepCommand;
use codegenerator::ir;
use codegenerator::ir::compilation_job as cjob;
use codegenerator::node::Node;
use codegenerator::tinysample::length_to_samples;
use numeric_array::NumericArray;
struct Deduplicator<'a> {
    // Signals have an unique UID
    signal_dedup: HashMap<&'a str, &'a Rc<cjob::Signal>>,
    // Loop parameters are unique by the UID
    loop_params: HashMap<String, Arc<cjob::SweepParameter>>,
}

impl<'a> Deduplicator<'a> {
    fn get_signal(&self, uid: &'a str) -> Option<Rc<cjob::Signal>> {
        self.signal_dedup.get(uid).cloned().cloned()
    }

    fn get_parameter(&self, uid: &'a str) -> Option<Arc<cjob::SweepParameter>> {
        self.loop_params.get(uid).cloned()
    }

    fn set_parameter(&mut self, value: Arc<cjob::SweepParameter>) {
        self.loop_params.insert(value.uid.clone(), value);
    }
}

fn py_to_nodekind(ob: &Bound<PyAny>, dedup: &mut Deduplicator) -> Result<ir::NodeKind, PyErr> {
    let py = ob.py();
    match ob
        .getattr(intern!(py, "__class__"))?
        .getattr(intern!(py, "__name__"))?
        .downcast::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "PulseIR" => {
            let is_acquire = ob.getattr(intern!(py, "is_acquire"))?.extract::<bool>()?;
            if is_acquire {
                if let Some(acquire_pulse) = extract_acquire_pulse(ob, dedup)? {
                    return Ok(ir::NodeKind::AcquirePulse(acquire_pulse));
                }
            }
            Ok(ir::NodeKind::PlayPulse(extract_pulse(ob, dedup)?))
        }
        "AcquireGroupIR" => Ok(ir::NodeKind::AcquirePulse(
            extract_acquire_pulse_group(ob, dedup)?.expect("Expected acquire pulse"),
        )),
        "MatchIR" => Ok(ir::NodeKind::Match(extract_match(ob)?)),
        "CaseIR" => Ok(ir::NodeKind::Case(extract_case(ob, dedup)?)),
        "SetOscillatorFrequencyIR" | "InitialOscillatorFrequencyIR" => Ok(
            ir::NodeKind::SetOscillatorFrequency(extract_set_oscillator_frequency(ob, dedup)?),
        ),
        "PhaseResetIR" => Ok(ir::NodeKind::PhaseReset(extract_reset_oscillator_phase(
            ob,
        )?)),
        "PrecompClearIR" => Ok(ir::NodeKind::PrecompensationFilterReset()),
        "PPCStepIR" => Ok(ir::NodeKind::PpcSweepStep(extract_ppc_step(ob)?)),
        "LoopIR" => Ok(ir::NodeKind::Loop(extract_loop(ob)?)),
        "LoopIterationIR" => Ok(ir::NodeKind::LoopIteration(extract_loop_iteration(
            ob, dedup,
        )?)),
        _ => {
            let length = ob.getattr(intern!(py, "length"))?.extract::<i64>()?;
            Ok(ir::NodeKind::Nop { length })
        }
    }
}

fn extract_node(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
    start: ir::Samples,
) -> Result<ir::IrNode, PyErr> {
    // ir.IntervalIR
    let py = ob.py();
    let mut ir_obj = py_to_nodekind(ob, dedup)?;
    match &mut ir_obj {
        ir::NodeKind::PlayPulse(pulse) => {
            let mut start = start;
            if pulse.pulse_def.is_some() {
                // Offset is legacy value used by tests, and should be
                // safely removed when the tests are refactored.
                // It cannot be set from the DSL API, defaulting to 0.
                let offset = ob
                    .getattr(intern!(py, "offset"))?
                    .extract::<ir::Samples>()?;
                pulse.length -= offset;
                start += offset;
            }
            let node = Node::new(ir_obj, start);
            Ok(node)
        }
        _ => {
            let mut node = Node::new(py_to_nodekind(ob, dedup)?, start);
            let children_start = ob.getattr(intern!(py, "children_start"))?.try_iter()?.map(
                |x| -> Result<i64, PyErr> {
                    match x {
                        Ok(val) => val.extract::<i64>(),
                        Err(e) => Err(e),
                    }
                },
            );
            let children = ob.getattr(intern!(py, "children"))?.try_iter()?;
            for (py_child, start_child) in children.into_iter().zip(children_start) {
                let child = extract_node(&py_child?, dedup, start_child?)?;
                node.add_child_node(child);
            }
            Ok(node)
        }
    }
}

fn extract_pulse_def(ob: &Bound<'_, PyAny>) -> Result<Option<cjob::PulseDef>, PyErr> {
    // ir.PulseDef
    if ob.is_none() {
        return Ok(None);
    };
    let py = ob.py();
    let uid = ob.getattr(intern!(py, "uid"))?.extract::<String>()?;
    let kind = if uid.starts_with("__marker__") {
        cjob::PulseDefKind::Marker
    } else {
        cjob::PulseDefKind::Pulse
    };
    let pdef = cjob::PulseDef { uid, kind };
    Ok(Some(pdef))
}

pub fn extract_markers(ob: &Bound<'_, PyAny>) -> Result<Vec<cjob::Marker>, PyErr> {
    // compilation_job.Marker
    if ob.is_none() {
        return Ok(vec![]);
    };
    let py = ob.py();
    ob.try_iter()?
        .map(|x| {
            let elem = x?;
            let marker = cjob::Marker {
                marker_selector: elem
                    .getattr(intern!(py, "marker_selector"))?
                    .extract::<String>()?,
                // NOTE: Currently in Python IR `enable` can be `None`, which is same as `False`,
                // So we default to `False`, this should probably be fixed upstream.
                enable: elem
                    .getattr(intern!(py, "enable"))?
                    .extract::<Option<bool>>()?
                    .unwrap_or(false),
                start: elem
                    .getattr(intern!(py, "start"))?
                    .extract::<Option<f64>>()?,
                length: elem
                    .getattr(intern!(py, "length"))?
                    .extract::<Option<f64>>()?,
                pulse_id: elem
                    .getattr(intern!(py, "pulse_id"))?
                    .extract::<Option<String>>()?,
            };
            Ok(marker)
        })
        .collect()
}

fn extract_set_oscillator_frequency(
    ob: &Bound<'_, PyAny>,
    dedup: &Deduplicator,
) -> Result<ir::SetOscillatorFrequency, PyErr> {
    // ir.SetOscillatorFrequency
    let py = ob.py();
    let values = ob.getattr(intern!(py, "values"))?.try_iter()?;
    let oscillators = ob.getattr(intern!(py, "oscillators"))?.try_iter()?;
    let mut osc_values: Vec<(Rc<cjob::Signal>, f64)> = vec![];
    for (osc, value) in oscillators.into_iter().zip(values) {
        let value = value?.extract::<f64>()?;
        for sig in osc?.getattr(intern!(py, "signals"))?.try_iter()? {
            let signal_uid = sig?.extract::<String>()?;
            let signal = dedup
                .get_signal(&signal_uid)
                .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", &signal_uid));
            osc_values.push((signal.clone(), value));
        }
    }
    let out = ir::SetOscillatorFrequency { values: osc_values };
    Ok(out)
}

fn extract_reset_oscillator_phase(ob: &Bound<'_, PyAny>) -> Result<ir::PhaseReset, PyErr> {
    // ir.PhaseResetIR
    let py = ob.py();
    let reset_sw_oscillators = ob
        .getattr(intern!(py, "reset_sw_oscillators"))?
        .extract::<bool>()?;
    let out = ir::PhaseReset {
        reset_sw_oscillators,
    };
    Ok(out)
}

fn extract_pulse(ob: &Bound<'_, PyAny>, dedup: &mut Deduplicator) -> Result<ir::PlayPulse, PyErr> {
    // ir.PulseIR
    let py = ob.py();
    let length = ob.getattr(intern!(py, "length"))?.extract::<i64>()?;
    let py_section_signal_pulse = ob.getattr(intern!(py, "pulse"))?;
    let py_signal = py_section_signal_pulse.getattr(intern!(py, "signal"))?;
    let signal_uid = py_signal.getattr(intern!(py, "uid"))?;
    let signal = dedup
        .get_signal(signal_uid.downcast::<PyString>()?.to_cow()?.as_ref())
        .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", &signal_uid));
    // TODO: PulseDef should be taken via deduplicator
    let py_pulse_def =
        extract_pulse_def(&py_section_signal_pulse.getattr(intern!(py, "pulse"))?)?.map(Arc::new);
    let phase = ob.getattr(intern!(py, "phase"))?.extract::<Option<f64>>()?;
    let increment_oscillator_phase = ob
        .getattr(intern!(py, "increment_oscillator_phase"))?
        .extract::<Option<f64>>()?;
    let incr_phase_param_name = ob
        .getattr(intern!(py, "incr_phase_param_name"))?
        .extract::<Option<String>>()?;
    let set_oscillator_phase = ob
        .getattr(intern!(py, "set_oscillator_phase"))?
        .extract::<Option<f64>>()?;
    // NOTE: This works only after pulse parameters are indexes!
    let id_pulse_params = ob
        .getattr(intern!(py, "play_pulse_params"))?
        .extract::<Option<usize>>()?;
    let amp_param_name = ob
        .getattr(intern!(py, "amp_param_name"))?
        .extract::<Option<String>>()?;

    Ok(ir::PlayPulse {
        length,
        signal,
        amplitude: extract_maybe_complex(&ob.getattr(intern!(py, "amplitude"))?)?,
        pulse_def: py_pulse_def,
        amp_param_name,
        phase: phase.unwrap_or(0.0),
        set_oscillator_phase,
        increment_oscillator_phase,
        incr_phase_param_name,
        id_pulse_params,
        markers: extract_markers(&ob.getattr(intern!(py, "markers"))?)?,
    })
}

fn extract_acquire_pulse(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<Option<ir::AcquirePulse>, PyErr> {
    // ir.PulseIR(is_acquire=True)
    let py = ob.py();
    let py_section_signal_pulse = ob.getattr(intern!(py, "pulse"))?;
    let py_signal = py_section_signal_pulse.getattr(intern!(py, "signal"))?;
    let maybe_pulse_def = &py_section_signal_pulse.getattr(intern!(py, "pulse"))?;
    let pulse_def = extract_pulse_def(maybe_pulse_def)?.map(Arc::new);
    // Acquires without pulse are not yet needed. Used for measurement calculator!
    // If PulseDef = None: No `integration_length`
    if pulse_def.is_none() {
        return Ok(None);
    }
    let length = ob
        .getattr(intern!(py, "integration_length"))?
        .extract::<i64>()?;
    let signal_uid = py_signal.getattr(intern!(py, "uid"))?;
    let signal = dedup
        .get_signal(signal_uid.downcast::<PyString>()?.to_cow()?.as_ref())
        .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", &signal_uid));

    let pulse_def = if let Some(pulse_def_py) =
        extract_pulse_def(&py_section_signal_pulse.getattr(intern!(py, "pulse"))?)?.map(Arc::new)
    {
        vec![pulse_def_py]
    } else {
        vec![]
    };
    // NOTE: This works only after pulse parameters are indexes!
    let id_pulse_params = ob
        .getattr(intern!(py, "play_pulse_params"))?
        .extract::<Option<usize>>()?;

    let acquire_pulse = ir::AcquirePulse {
        length,
        signal,
        pulse_defs: pulse_def,
        id_pulse_params: vec![id_pulse_params],
    };
    Ok(Some(acquire_pulse))
}

fn extract_acquire_pulse_group(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<Option<ir::AcquirePulse>, PyErr> {
    // ir.AcquireGroupIR
    let py = ob.py();
    let mut pulse_defs = vec![];
    // Acquire group can have multiple pulses, which share identical parameters,
    // except for pulse parameters ID.
    let pulses_bound = ob.getattr(intern!(py, "pulses"))?;
    let pulses_py = pulses_bound.downcast::<PyList>()?;
    let py_section_signal_pulse_base = pulses_py.get_item(0)?;
    let py_signal = py_section_signal_pulse_base.getattr(intern!(py, "signal"))?;
    let signal_uid = py_signal.getattr(intern!(py, "uid"))?;
    let signal = dedup
        .get_signal(signal_uid.downcast::<PyString>()?.to_cow()?.as_ref())
        .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", &signal_uid));

    // Single acquire group can consist of multiple individual pulses
    // Length of pulse defs must match pulse params ID
    for py_section_signal_pulse in pulses_py.try_iter()? {
        let maybe_pulse_def = &py_section_signal_pulse?.getattr(intern!(py, "pulse"))?;
        let pulse_def = extract_pulse_def(maybe_pulse_def)?
            .expect("Internal error: Acquire group pulse def is None");
        pulse_defs.push(Arc::new(pulse_def));
    }
    let length = ob.getattr(intern!(py, "length"))?.extract::<i64>()?;
    // NOTE: This works only after pulse parameters are indexes!
    let id_pulse_params = ob
        .getattr(intern!(py, "play_pulse_params"))?
        .extract::<Vec<Option<usize>>>()?;
    let acquire_pulse = ir::AcquirePulse {
        length,
        signal,
        pulse_defs,
        id_pulse_params,
    };
    Ok(Some(acquire_pulse))
}

fn extract_maybe_complex(ob: &Bound<'_, PyAny>) -> Result<Option<Complex<f64>>, PyErr> {
    if ob.is_none() {
        return Ok(None);
    }
    let py = ob.py();
    let complex_func = py
        .import(intern!(py, "builtins"))?
        .getattr(intern!(py, "complex"))?;
    let value = complex_func.call1((ob.into_pyobject(py)?,))?;
    let py_complex = value.downcast::<PyComplex>()?;
    let out = Complex {
        re: py_complex.real(),
        im: py_complex.imag(),
    };
    Ok(Some(out))
}

fn extract_match(ob: &Bound<'_, PyAny>) -> Result<ir::Match, PyErr> {
    // ir.MatchIR
    let py = ob.py();
    let section = ob.getattr(intern!(py, "section"))?.extract::<String>()?;
    let handle = ob
        .getattr(intern!(py, "handle"))?
        .extract::<Option<String>>()?;
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    let user_register = ob
        .getattr(intern!(py, "user_register"))?
        .extract::<Option<i64>>()?;
    let local = ob
        .getattr(intern!(py, "local"))?
        .extract::<Option<bool>>()?;
    let prng_sample = ob
        .getattr(intern!(py, "prng_sample"))?
        .extract::<Option<String>>()?;
    let obj = ir::Match {
        section,
        length,
        handle,
        user_register,
        local: local.unwrap_or(false),
        prng_sample,
    };
    Ok(obj)
}

fn extract_case(ob: &Bound<'_, PyAny>, dedup: &mut Deduplicator) -> Result<ir::Case, PyErr> {
    // ir.CaseIR
    let py = ob.py();
    let state = ob.getattr(intern!(py, "state"))?.extract::<u16>()?;
    let length = ob.getattr(intern!(py, "length"))?.extract::<i64>()?;
    let py_signals = ob.getattr(intern!(py, "signals"))?;
    let mut signals_out = vec![];
    py_signals
        .try_iter()?
        .try_for_each(|item| -> Result<(), PyErr> {
            let item = item?;
            let sig_ref: Cow<'_, str> = item.downcast::<PyString>()?.to_cow()?;
            let out = dedup
                .get_signal(sig_ref.as_ref())
                .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", sig_ref.as_ref()));
            signals_out.push(out);
            Ok(())
        })?;
    Ok(ir::Case {
        state,
        length,
        signals: signals_out,
    })
}

/// Extract loop
fn extract_loop(ob: &Bound<'_, PyAny>) -> Result<ir::Loop, PyErr> {
    // ir.LoopIR
    let py = ob.py();
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    let compressed = ob.getattr(intern!(py, "compressed"))?.extract::<bool>()?;
    Ok(ir::Loop { length, compressed })
}

/// Extract loop iteration
///
/// Fills the deduplicator with sweep parameters if there are any
fn extract_loop_iteration(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<ir::LoopIteration, PyErr> {
    // ir.LoopIterationIR
    let py = ob.py();
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    let iteration = ob.getattr(intern!(py, "iteration"))?.extract::<u64>()?;
    let parameters = ob.getattr(intern!(py, "sweep_parameters"))?;
    let mut params_out = vec![];
    for param in parameters.try_iter()? {
        let param = extract_parameter(&param?, dedup)?;
        params_out.push(param);
    }
    Ok(ir::LoopIteration {
        length,
        iteration,
        parameters: params_out,
    })
}

/// Extract PPC Sweep Step
fn extract_ppc_step(ob: &Bound<'_, PyAny>) -> Result<ir::PpcSweepStep, PyErr> {
    // ir.PPCStepIR
    let py = ob.py();
    let trigger_duration = ob
        .getattr(intern!(py, "trigger_duration"))?
        .extract::<ir::Samples>()?;
    let pump_power = ob
        .getattr(intern!(py, "pump_power"))?
        .extract::<Option<f64>>()?;
    let pump_frequency = ob
        .getattr(intern!(py, "pump_frequency"))?
        .extract::<Option<f64>>()?;
    let probe_power = ob
        .getattr(intern!(py, "probe_power"))?
        .extract::<Option<f64>>()?;
    let probe_frequency = ob
        .getattr(intern!(py, "probe_frequency"))?
        .extract::<Option<f64>>()?;
    let cancellation_phase = ob
        .getattr(intern!(py, "cancellation_phase"))?
        .extract::<Option<f64>>()?;
    let cancellation_attenuation = ob
        .getattr(intern!(py, "cancellation_attenuation"))?
        .extract::<Option<f64>>()?;
    Ok(ir::PpcSweepStep {
        length: trigger_duration,
        sweep_command: SweepCommand {
            pump_power,
            pump_frequency,
            probe_power,
            probe_frequency,
            cancellation_phase,
            cancellation_attenuation,
        },
    })
}

fn extract_oscillator(ob: &Bound<'_, PyAny>) -> Result<Option<cjob::Oscillator>, PyErr> {
    // compilation_job.OscillatorInfo
    if ob.is_none() {
        return Ok(None);
    }
    let py = ob.py();
    let uid = ob.getattr(intern!(py, "uid"))?.extract::<String>()?;
    let osc = match ob
        .getattr(intern!(py, "is_hardware"))?
        .extract::<Option<bool>>()?
    {
        Some(x) => match x {
            true => Some(cjob::Oscillator {
                uid,
                kind: cjob::OscillatorKind::HARDWARE,
            }),
            false => Some(cjob::Oscillator {
                uid,
                kind: cjob::OscillatorKind::SOFTWARE,
            }),
        },
        None => None,
    };
    Ok(osc)
}

fn extract_awg_signal(ob: &Bound<'_, PyAny>, sampling_rate: f64) -> Result<cjob::Signal, PyErr> {
    // compilation_job.SignalObj
    let py = ob.py();
    let signal_type = match ob
        .getattr(intern!(py, "signal_type"))?
        .downcast_into::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "integration" => cjob::SignalKind::INTEGRATION,
        "iq" => cjob::SignalKind::IQ,
        "single" => cjob::SignalKind::SINGLE,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown signal type: {}",
                ob
            )));
        }
    };
    let delay = ob
        .getattr(intern!(py, "total_delay"))?
        .extract::<f64>()
        .unwrap_or(0.0);
    let signal = cjob::Signal {
        uid: ob.getattr(intern!(py, "id"))?.extract::<String>()?,
        kind: signal_type,
        channels: ob.getattr(intern!(py, "channels"))?.extract::<Vec<u8>>()?,
        delay: length_to_samples(delay, sampling_rate),
        // AWG SignalObj does not have full oscillator info, we take it from elsewhere
        oscillator: None,
    };
    Ok(signal)
}

pub fn extract_device_kind(ob: &Bound<'_, PyAny>) -> Result<cjob::DeviceKind, PyErr> {
    // device_type.DeviceType
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let kind = match py_name.downcast::<PyString>()?.to_cow()?.as_ref() {
        "HDAWG" => cjob::DeviceKind::HDAWG,
        "SHFQA" => cjob::DeviceKind::SHFQA,
        "SHFSG" => cjob::DeviceKind::SHFSG,
        "UHFQA" => cjob::DeviceKind::UHFQA,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown device type: {}",
                ob
            )));
        }
    };
    Ok(kind)
}

pub fn extract_oscillator_from_ir_signal(
    ob: &Bound<'_, PyAny>,
) -> Result<HashMap<String, Option<cjob::Oscillator>>, PyErr> {
    // ir.SignalIR
    let py = ob.py();
    let mut osc_map = HashMap::new();
    for sig in ob.try_iter()? {
        let sig = sig?;
        let sig_uid = sig.getattr(intern!(py, "uid"))?.extract::<String>()?;
        let osc = extract_oscillator(&sig.getattr(intern!(py, "oscillator"))?)?;
        osc_map.insert(sig_uid, osc);
    }
    Ok(osc_map)
}

fn extract_awg_kind(ob: &Bound<'_, PyAny>) -> Result<cjob::AwgKind, PyErr> {
    // awg_info.AWGSignalType
    let py = ob.py();
    let out = match ob
        .getattr(intern!(py, "name"))?
        .downcast_into::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "IQ" => cjob::AwgKind::IQ,
        "MULTI" => cjob::AwgKind::MULTI,
        "SINGLE" => cjob::AwgKind::SINGLE,
        "DOUBLE" => cjob::AwgKind::DOUBLE,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown awg signal type: {}",
                ob
            )));
        }
    };
    Ok(out)
}

fn extract_awg_oscs(ob: &Bound<'_, PyAny>) -> Result<HashMap<String, u16>, PyErr> {
    // awg_info.AWGInfo.oscs
    let out = ob.extract::<HashMap<String, u16>>()?;
    Ok(out)
}

pub fn extract_awg<'py>(
    ob: &Bound<'py, PyAny>,
    ir_signals: &Bound<'py, PyAny>,
) -> Result<cjob::AwgCore, PyErr> {
    // awg_info.AWGInfo
    let py = ob.py();
    let sampling_rate = ob.getattr(intern!(py, "sampling_rate"))?.extract::<f64>()?;
    // AWG signal must be combined from `SignalIR` and `SignalObj`
    let mut osc_map = extract_oscillator_from_ir_signal(ir_signals)?;
    let signals: Result<Vec<Rc<cjob::Signal>>, PyErr> = ob
        .getattr(intern!(py, "signals"))?
        .try_iter()?
        .map(|x| {
            let mut sig = extract_awg_signal(&x?, sampling_rate)?;
            sig.oscillator = osc_map.remove(&sig.uid).unwrap_or_else(|| {
                panic!(
                    "Internal error: No oscillator found for signal: '{}'",
                    &sig.uid
                )
            });
            Ok(Rc::new(sig))
        })
        .collect();
    let awg = cjob::AwgCore {
        kind: extract_awg_kind(&ob.getattr(intern!(py, "signal_type"))?)?,
        signals: signals?,
        sampling_rate,
        device_kind: extract_device_kind(&ob.getattr(intern!(py, "device_type"))?)?,
        osc_allocation: extract_awg_oscs(&ob.getattr(intern!(py, "oscs"))?)?,
    };
    Ok(awg)
}

/// Extract parameter
fn extract_parameter(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<Arc<cjob::SweepParameter>, PyErr> {
    // compilation_job.ParameterInfo
    let py = ob.py();
    let uid = ob.getattr(intern!(py, "uid"))?.extract::<String>()?;
    if let Some(param) = dedup.get_parameter(&uid) {
        return Ok(param);
    }
    let numeric_array = match NumericArray::from_py(&ob.getattr(intern!(py, "values"))?) {
        Ok(arr) => arr,
        Err(e) => {
            let msg = format!("Invalid array type on sweep parameter '{}'. {}", uid, e);
            return Err(PyValueError::new_err(msg));
        }
    };
    let obj = Arc::new(cjob::SweepParameter {
        uid,
        values: numeric_array,
    });
    dedup.set_parameter(obj.clone());
    Ok(obj)
}

/// Transform Python IR to code IR
///
/// While the main purpose of this function is to translate Python IR
/// structs into Rust, it also lowers the source IR into code IR as we
/// do not (yet) have Rust models for the original IR.
pub fn transform_py_ir(
    ob: &Bound<'_, PyAny>,
    signals: &[Rc<cjob::Signal>],
) -> Result<ir::IrNode, PyErr> {
    let lookup: HashMap<&str, &Rc<cjob::Signal>> =
        signals.iter().map(|x| (x.uid.as_str(), x)).collect();
    let mut deduplicator = Deduplicator {
        signal_dedup: lookup,
        loop_params: HashMap::new(),
    };
    extract_node(ob, &mut deduplicator, 0)
}
