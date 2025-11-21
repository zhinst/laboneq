// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Translations from Python IR into code generator IR
use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::RandomState;
use std::sync::Arc;
use std::vec;

use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use pyo3::types::{PyComplex, PyString};

use num_complex::Complex;
use numeric_array::NumericArray;

use codegenerator::FeedbackRegister;
use codegenerator::FeedbackRegisterLayout;
use codegenerator::SingleFeedbackRegisterLayoutItem;
use codegenerator::ir;
use codegenerator::ir::PpcDevice;
use codegenerator::ir::PrngSetup;
use codegenerator::ir::SectionId;
use codegenerator::ir::SignalFrequency;
use codegenerator::ir::compilation_job as cjob;
use codegenerator::ir::compilation_job::AwgKey;
use codegenerator::ir::compilation_job::Device;
use codegenerator::ir::compilation_job::{AwgCore, Signal};
use codegenerator::ir::experiment::SectionInfo;
use codegenerator::ir::experiment::SweepCommand;
use codegenerator::ir::experiment::{AcquisitionType, Handle, PulseParametersId, UserRegister};
use codegenerator::node::Node;
use codegenerator::utils::length_to_samples;

use crate::error::Error;
use crate::pulse_parameters::{PulseParameters, create_pulse_parameters};

struct Deduplicator<'a> {
    // Signals have an unique UID
    signal_dedup: HashMap<&'a str, &'a Arc<Signal>>,
    // Loop parameters are unique by the UID
    loop_params: HashMap<String, Arc<cjob::SweepParameter>>,
    // Section Info
    section_info: HashMap<String, Arc<SectionInfo>>,
    // PPC device per signal (channel)
    ppc_devices: HashMap<String, Arc<PpcDevice>>,
    pulse_parameters: HashMap<PulseParametersId, PulseParameters>,
}

impl<'a> Deduplicator<'a> {
    fn new(signals: Vec<&'a Arc<Signal>>) -> Self {
        let signal_dedup: HashMap<&str, &Arc<Signal>> =
            signals.iter().map(|&x| (x.uid.as_str(), x)).collect();
        Self {
            signal_dedup,
            loop_params: HashMap::new(),
            ppc_devices: HashMap::new(),
            section_info: HashMap::new(),
            pulse_parameters: HashMap::new(),
        }
    }

    fn get_signal(&self, uid: &'a str) -> Option<Arc<Signal>> {
        self.signal_dedup.get(uid).cloned().cloned()
    }

    fn get_parameter(&self, uid: &'a str) -> Option<Arc<cjob::SweepParameter>> {
        self.loop_params.get(uid).cloned()
    }

    fn set_parameter(&mut self, value: Arc<cjob::SweepParameter>) {
        self.loop_params.insert(value.uid.clone(), value);
    }

    fn get_or_create_section_info(&mut self, uid: &str) -> Arc<SectionInfo> {
        if let Some(info) = self.section_info.get(uid) {
            Arc::clone(info)
        } else {
            let info = Arc::new(SectionInfo {
                id: self.section_info.len() as SectionId + 1,
                name: uid.to_string(),
            });
            self.section_info.insert(uid.to_string(), Arc::clone(&info));
            info
        }
    }

    fn set_ppc_device(&mut self, signal: &str, device: String, channel: u16) -> Arc<PpcDevice> {
        if let Some(ppc_device) = self.ppc_devices.get(signal) {
            Arc::clone(ppc_device)
        } else {
            let ppc = Arc::new(PpcDevice { device, channel });
            self.ppc_devices
                .insert(signal.to_string(), Arc::clone(&ppc));
            ppc
        }
    }

    fn register_pulse_parameters(
        &mut self,
        py: Python,
        pulse_parameters: Option<&Bound<'_, PyDict>>,
        play_parameters: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<PulseParametersId>> {
        if pulse_parameters.is_none() && play_parameters.is_none() {
            return Ok(None);
        }
        let parameters = create_pulse_parameters(py, pulse_parameters, play_parameters)?;
        let parameters_id = parameters.id();
        self.pulse_parameters
            .entry(parameters_id)
            .or_insert(parameters);
        Ok(Some(parameters_id))
    }

    fn take_pulse_parameters(&mut self) -> HashMap<PulseParametersId, PulseParameters> {
        std::mem::take(&mut self.pulse_parameters)
    }
}

fn py_to_nodekind(ob: &Bound<PyAny>, dedup: &mut Deduplicator) -> Result<ir::NodeKind, PyErr> {
    let py = ob.py();
    match ob
        .getattr(intern!(py, "__class__"))?
        .getattr(intern!(py, "__name__"))?
        .cast::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "PulseIR" => {
            let is_acquire = ob.getattr(intern!(py, "is_acquire"))?.extract::<bool>()?;
            if is_acquire && let Some(acquire_pulse) = extract_acquire_pulse(ob, dedup)? {
                return Ok(ir::NodeKind::AcquirePulse(acquire_pulse));
            }
            Ok(ir::NodeKind::PlayPulse(extract_pulse(ob, dedup)?))
        }
        "AcquireGroupIR" => Ok(ir::NodeKind::AcquirePulse(
            extract_acquire_pulse_group(ob, dedup)?.expect("Expected acquire pulse"),
        )),
        "MatchIR" => Ok(ir::NodeKind::Match(extract_match(ob, dedup)?)),
        "CaseIR" => Ok(ir::NodeKind::Case(extract_case(ob, dedup)?)),
        "SetOscillatorFrequencyIR" => Ok(ir::NodeKind::SetOscillatorFrequency(
            extract_set_oscillator_frequency(ob, dedup)?,
        )),
        "InitialOscillatorFrequencyIR" => Ok(ir::NodeKind::InitialOscillatorFrequency(
            extract_initial_oscillator_frequency(ob, dedup)?,
        )),
        "PhaseResetIR" => Ok(ir::NodeKind::PhaseReset(extract_reset_oscillator_phase(
            ob, dedup,
        )?)),
        "PrecompClearIR" => Ok(ir::NodeKind::PrecompensationFilterReset {
            signal: extract_precompensation_clear_signals(ob, dedup)?,
        }),
        "PPCStepIR" => Ok(ir::NodeKind::PpcSweepStep(extract_ppc_step(ob, dedup)?)),
        "LoopIR" => Ok(ir::NodeKind::Loop(extract_loop(ob, dedup)?)),
        "LoopIterationIR" => Ok(ir::NodeKind::LoopIteration(extract_loop_iteration(ob)?)),
        "SectionIR" => Ok(ir::NodeKind::Section(extract_section(ob, dedup)?)),
        _ => {
            let length = ob.getattr(intern!(py, "length"))?.extract::<i64>()?;
            Ok(ir::NodeKind::Nop { length })
        }
    }
}

fn child_nodes(
    ir_data: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<Vec<ir::IrNode>, PyErr> {
    let py = ir_data.py();
    let children_start = ir_data
        .getattr(intern!(py, "children_start"))?
        .try_iter()?
        .map(|x| -> Result<i64, PyErr> {
            match x {
                Ok(val) => val.extract::<i64>(),
                Err(e) => Err(e),
            }
        });
    let children_vec = ir_data.getattr(intern!(py, "children"))?;
    let mut child_nodes = Vec::with_capacity(children_vec.len()?);
    for (py_child, start_child) in children_vec.try_iter()?.zip(children_start) {
        let child = extract_node(&py_child?, dedup, start_child?)?;
        child_nodes.extend(child);
    }
    Ok(child_nodes)
}

fn extract_node(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
    start: ir::Samples,
) -> Result<Vec<ir::IrNode>, PyErr> {
    // ir.IntervalIR
    let node_kind = py_to_nodekind(ob, dedup)?;
    let mut node = Node::new(node_kind, start);
    let children = child_nodes(ob, dedup)?;
    node.add_child_nodes(children);
    Ok(vec![node])
}

fn signals_set_to_signals(
    // set[str]
    signals_set: &Bound<'_, PyAny>,
    dedup: &Deduplicator<'_>,
) -> Result<Vec<Arc<Signal>>, PyErr> {
    if signals_set.is_none() {
        return Ok(vec![]);
    }
    let mut signals = Vec::with_capacity(signals_set.len()?);
    signals_set
        .try_iter()?
        .try_for_each(|item| -> Result<(), PyErr> {
            let item = item?;
            let sig_ref: Cow<'_, str> = item.cast::<PyString>()?.to_cow()?;
            let out = dedup
                .get_signal(sig_ref.as_ref())
                .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", sig_ref.as_ref()));
            signals.push(out);
            Ok(())
        })?;
    Ok(signals)
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
    let function = ob.getattr(intern!(py, "function"))?;
    let samples = ob.getattr(intern!(py, "samples"))?;
    let pulse_type = if !function.is_none() {
        Some(cjob::PulseType::Function)
    } else if !samples.is_none() {
        Some(cjob::PulseType::Samples)
    } else {
        None
    };
    let pdef = cjob::PulseDef {
        uid,
        kind,
        pulse_type,
    };
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
    let values = ob.getattr(intern!(ob.py(), "values"))?.try_iter()?;
    let values = values.into_iter().map(|v| {
        let (signal, value) = v.unwrap().extract::<(String, f64)>().unwrap();
        SignalFrequency {
            signal: dedup.get_signal(&signal).unwrap(),
            frequency: value,
        }
    });
    let out = ir::SetOscillatorFrequency::new(values.collect());
    Ok(out)
}

fn extract_precompensation_clear_signals(
    ob: &Bound<'_, PyAny>,
    dedup: &Deduplicator,
) -> Result<Arc<Signal>, Error> {
    // ir.PrecompClearIR
    let py = ob.py();
    signals_set_to_signals(&ob.getattr(intern!(py, "signals"))?, dedup)?
        .pop()
        .ok_or_else(|| Error::new("Expected at least one signal in `PrecompClearIR`"))
}

fn extract_initial_oscillator_frequency(
    ob: &Bound<'_, PyAny>,
    dedup: &Deduplicator,
) -> Result<ir::InitialOscillatorFrequency, PyErr> {
    let values = ob.getattr(intern!(ob.py(), "values"))?.try_iter()?;
    let values = values.into_iter().map(|v| {
        let (signal, value) = v.unwrap().extract::<(String, f64)>().unwrap();
        SignalFrequency {
            signal: dedup.get_signal(&signal).unwrap(),
            frequency: value,
        }
    });
    let out = ir::InitialOscillatorFrequency::new(values.collect());
    Ok(out)
}

fn extract_reset_oscillator_phase(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<ir::PhaseReset, PyErr> {
    // ir.PhaseResetIR
    let py = ob.py();
    let signals = signals_set_to_signals(&ob.getattr(intern!(py, "signals"))?, dedup)?;
    let out = ir::PhaseReset { signals };
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
        .get_signal(signal_uid.cast::<PyString>()?.to_cow()?.as_ref())
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

    // Index pulse parameters
    let pulse_pulse_params = ob.getattr(intern!(py, "pulse_pulse_params"))?;
    let play_pulse_params = ob.getattr(intern!(py, "play_pulse_params"))?;
    let pulse_parameters = pulse_pulse_params.cast::<PyDict>().ok();
    let play_parameters = play_pulse_params.cast::<PyDict>().ok();
    let id_pulse_params = dedup.register_pulse_parameters(py, pulse_parameters, play_parameters)?;

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
        .get_signal(signal_uid.cast::<PyString>()?.to_cow()?.as_ref())
        .unwrap_or_else(|| panic!("Internal error: Missing signal: {}", &signal_uid));

    let pulse_def = if let Some(pulse_def_py) =
        extract_pulse_def(&py_section_signal_pulse.getattr(intern!(py, "pulse"))?)?.map(Arc::new)
    {
        vec![pulse_def_py]
    } else {
        vec![]
    };

    // Index pulse parameters
    let pulse_pulse_params = ob.getattr(intern!(py, "pulse_pulse_params"))?;
    let play_pulse_params = ob.getattr(intern!(py, "play_pulse_params"))?;
    let pulse_parameters = pulse_pulse_params.cast::<PyDict>().ok();
    let play_parameters = play_pulse_params.cast::<PyDict>().ok();
    let id_pulse_params = dedup.register_pulse_parameters(py, pulse_parameters, play_parameters)?;

    let acq_params = py_section_signal_pulse.getattr(intern!(py, "acquire_params"))?;
    let handle = acq_params
        .getattr(intern!(py, "handle"))?
        .extract::<String>()?;
    let acquire_pulse = ir::AcquirePulse {
        length,
        signal,
        pulse_defs: pulse_def,
        id_pulse_params: vec![id_pulse_params],
        handle: handle.into(),
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
    let pulses_py = pulses_bound.cast::<PyList>()?;
    let py_section_signal_pulse_base = pulses_py.get_item(0)?;
    let py_signal = py_section_signal_pulse_base.getattr(intern!(py, "signal"))?;
    let signal_uid = py_signal.getattr(intern!(py, "uid"))?;
    let signal = dedup
        .get_signal(signal_uid.cast::<PyString>()?.to_cow()?.as_ref())
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

    // Index pulse parameters
    let pulse_pulse_params = ob.getattr(intern!(py, "pulse_pulse_params"))?;
    let play_pulse_params = ob.getattr(intern!(py, "play_pulse_params"))?;
    let mut id_pulse_params = vec![];
    for (pulse, play) in pulse_pulse_params
        .try_iter()?
        .zip(play_pulse_params.try_iter()?)
    {
        let pulse_parameters_py = pulse?;
        let play_parameters_py = play?;
        let pulse = pulse_parameters_py.cast::<PyDict>().ok();
        let play = play_parameters_py.cast::<PyDict>().ok();
        let parameters_id = dedup.register_pulse_parameters(py, pulse, play)?;
        id_pulse_params.push(parameters_id);
    }

    let acq_params = py_section_signal_pulse_base.getattr(intern!(py, "acquire_params"))?;
    let handle = acq_params
        .getattr(intern!(py, "handle"))?
        .extract::<String>()?;
    let acquire_pulse = ir::AcquirePulse {
        length,
        signal,
        pulse_defs,
        id_pulse_params,
        handle: handle.into(),
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
    let py_complex = value.cast::<PyComplex>()?;
    let out = Complex {
        re: py_complex.real(),
        im: py_complex.imag(),
    };
    Ok(Some(out))
}

fn extract_match(ob: &Bound<'_, PyAny>, dedup: &mut Deduplicator) -> Result<ir::Match, PyErr> {
    // ir.MatchIR
    let py = ob.py();
    let section = ob.getattr(intern!(py, "section"))?.extract::<String>()?;
    let section_info = dedup.get_or_create_section_info(&section);
    let handle = ob
        .getattr(intern!(py, "handle"))?
        .extract::<Option<String>>()?;
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    let user_register = ob
        .getattr(intern!(py, "user_register"))?
        .extract::<Option<UserRegister>>()?;
    let local = ob
        .getattr(intern!(py, "local"))?
        .extract::<Option<bool>>()?;
    let prng_sample = ob
        .getattr(intern!(py, "prng_sample"))?
        .extract::<Option<String>>()?;
    let obj = ir::Match {
        section_info,
        length,
        handle: handle.map(Handle::from),
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
    let signals = signals_set_to_signals(&ob.getattr(intern!(py, "signals"))?, dedup)?;
    Ok(ir::Case {
        state,
        length,
        signals,
        section_info: dedup.get_or_create_section_info(
            ob.getattr(intern!(py, "section"))?
                .extract::<String>()?
                .as_str(),
        ),
    })
}

/// Extract loop
fn extract_loop(ob: &Bound<'_, PyAny>, dedup: &mut Deduplicator) -> Result<ir::Loop, PyErr> {
    // ir.LoopIR
    let py = ob.py();
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    let compressed = ob.getattr(intern!(py, "compressed"))?.extract::<bool>()?;
    let section_info = dedup.get_or_create_section_info(
        ob.getattr(intern!(py, "section"))?
            .extract::<String>()?
            .as_str(),
    );
    let count = ob.getattr(intern!(py, "iterations"))?.extract::<u64>()?;
    // NOTE: Currently PRNG information is only available in loop iterations.
    // This is a workaround to get the PRNG sample from the first iteration.
    let first_iteration = ob
        .getattr(intern!(py, "children"))?
        .get_item(0)
        .expect("Internal Error: Loop has no children");
    let prng_sample = first_iteration
        .getattr(intern!(py, "prng_sample"))?
        .extract::<Option<String>>()?;
    let parameters = ob.getattr(intern!(py, "sweep_parameters"))?;
    let parameters = parameters
        .try_iter()?
        .map(|param| extract_parameter(&param?, dedup))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ir::Loop {
        length,
        compressed,
        section_info,
        count,
        prng_sample,
        parameters,
    })
}

/// Extract loop iteration
fn extract_loop_iteration(ob: &Bound<'_, PyAny>) -> Result<ir::LoopIteration, PyErr> {
    // ir.LoopIterationIR
    let py = ob.py();
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    Ok(ir::LoopIteration { length })
}

/// Extract PPC Sweep Step
fn extract_ppc_step(
    ob: &Bound<'_, PyAny>,
    dedup: &mut Deduplicator,
) -> Result<ir::PpcSweepStep, Error> {
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
    let ppc_device = ob.getattr(intern!(py, "ppc_device"))?.extract::<String>()?;
    let ppc_channel = ob.getattr(intern!(py, "ppc_channel"))?.extract::<u16>()?;
    let signal = signals_set_to_signals(&ob.getattr(intern!(py, "signals"))?, dedup)?
        .pop()
        .ok_or_else(|| Error::new("Expected at least one signal in `PPCStepIR`"))?;
    let ppc_device = dedup.set_ppc_device(&signal.uid, ppc_device, ppc_channel);
    Ok(ir::PpcSweepStep {
        signal,
        length: trigger_duration,
        sweep_command: SweepCommand {
            pump_power,
            pump_frequency,
            probe_power,
            probe_frequency,
            cancellation_phase,
            cancellation_attenuation,
        },
        ppc_device,
    })
}

/// Extract section, along with trigger and PRNG data
fn extract_section(ob: &Bound<'_, PyAny>, dedup: &mut Deduplicator) -> Result<ir::Section, Error> {
    // ir.SectionIR
    let py = ob.py();
    let uid = ob.getattr(intern!(py, "section"))?.extract::<String>()?;
    let section_info = dedup.get_or_create_section_info(&uid);
    let length = ob
        .getattr(intern!(py, "length"))?
        .extract::<ir::Samples>()?;
    let prng_setup = ob.getattr(intern!(py, "prng_setup"))?;
    let prng_setup = if prng_setup.is_none() {
        None
    } else {
        Some(PrngSetup {
            range: prng_setup.getattr(intern!(py, "range"))?.extract::<u16>()?,
            seed: prng_setup.getattr(intern!(py, "seed"))?.extract::<u32>()?,
            section_info: Arc::clone(&section_info),
        })
    };
    let trigger_output: Result<Vec<(Arc<Signal>, u8)>, Error> = ob
        .getattr(intern!(py, "trigger_output"))?
        .extract::<HashSet<(String, u8)>>()?
        .iter()
        .map(|(sig, bit)| -> Result<(Arc<Signal>, u8), _> {
            let signal = dedup.get_signal(sig).ok_or_else(|| {
                Error::new(&format!(
                    "Internal error: Missing signal for trigger: {sig}"
                ))
            })?;
            Ok((signal, *bit))
        })
        .collect();
    let section = ir::Section {
        length,
        prng_setup,
        trigger_output: trigger_output?,
        section_info: Arc::clone(&section_info),
    };
    Ok(section)
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

pub fn extract_mixer_type(ob: &Bound<'_, PyAny>) -> Result<Option<cjob::MixerType>, PyErr> {
    // schedued_experiment.MixerType
    if ob.is_none() {
        return Ok(None);
    }
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let kind = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "IQ" => cjob::MixerType::IQ,
        "UHFQA_ENVELOPE" => cjob::MixerType::UhfqaEnvelope,
        _ => {
            return Err(PyRuntimeError::new_err(format!("Unknown mixer type: {ob}")));
        }
    };
    Ok(Some(kind))
}

fn extract_awg_signal(ob: &Bound<'_, PyAny>, sampling_rate: f64) -> Result<Signal, PyErr> {
    // compilation_job.SignalObj
    let py = ob.py();
    let signal_type = match ob
        .getattr(intern!(py, "signal_type"))?
        .cast_into::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "integration" => cjob::SignalKind::INTEGRATION,
        "iq" => cjob::SignalKind::IQ,
        "single" => cjob::SignalKind::SINGLE,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown signal type: {ob}"
            )));
        }
    };
    let start_delay_seconds = ob.getattr(intern!(py, "start_delay"))?.extract::<f64>()?;
    let signal_delay_seconds = ob.getattr(intern!(py, "delay_signal"))?.extract::<f64>()?;
    let start_delay = length_to_samples(start_delay_seconds, sampling_rate);
    let signal_delay = length_to_samples(signal_delay_seconds, sampling_rate);
    let automute = ob.getattr(intern!(py, "automute"))?.extract::<bool>()?;

    let signal = Signal {
        uid: ob.getattr(intern!(py, "id"))?.extract::<String>()?,
        kind: signal_type,
        channels: ob.getattr(intern!(py, "channels"))?.extract::<Vec<u8>>()?,
        start_delay,
        signal_delay,
        // AWG SignalObj does not have full oscillator info, we take it from elsewhere
        oscillator: None,
        mixer_type: extract_mixer_type(&ob.getattr(intern!(py, "mixer_type"))?)?,
        automute,
    };
    Ok(signal)
}

pub fn extract_device_kind(ob: &Bound<'_, PyAny>) -> Result<cjob::DeviceKind, PyErr> {
    // device_type.DeviceType
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let kind = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "HDAWG" => cjob::DeviceKind::HDAWG,
        "SHFQA" => cjob::DeviceKind::SHFQA,
        "SHFSG" => cjob::DeviceKind::SHFSG,
        "UHFQA" => cjob::DeviceKind::UHFQA,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown device type: {ob}"
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
        .cast_into::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "IQ" => cjob::AwgKind::IQ,
        "SINGLE" => cjob::AwgKind::SINGLE,
        "DOUBLE" => cjob::AwgKind::DOUBLE,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown awg signal type: {ob}"
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

fn extract_trigger_mode(ob: &Bound<'_, PyAny>) -> Result<cjob::TriggerMode, PyErr> {
    // compilation_job.TriggerMode
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let mode = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "NONE" => cjob::TriggerMode::ZSync,
        "DIO_TRIGGER" => cjob::TriggerMode::DioTrigger,
        "DIO_WAIT" => cjob::TriggerMode::DioWait,
        "INTERNAL_TRIGGER_WAIT" => cjob::TriggerMode::InternalTriggerWait,
        "INTERNAL_READY_CHECK" => cjob::TriggerMode::InternalReadyCheck,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown trigger mode: {ob}"
            )));
        }
    };
    Ok(mode)
}

pub fn extract_awg<'py>(
    ob: &Bound<'py, PyAny>,
    ir_signals: &Bound<'py, PyAny>,
) -> Result<AwgCore, PyErr> {
    // awg_info.AWGInfo
    let py = ob.py();
    let sampling_rate = ob.getattr(intern!(py, "sampling_rate"))?.extract::<f64>()?;
    let device_kind = extract_device_kind(&ob.getattr(intern!(py, "device_type"))?)?;
    // AWG signal must be combined from `SignalIR` and `SignalObj`
    let mut osc_map = extract_oscillator_from_ir_signal(ir_signals)?;
    let signals: Result<Vec<Arc<Signal>>, PyErr> = ob
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
            Ok(Arc::new(sig))
        })
        .collect();
    let signal_kinds = HashSet::<_, RandomState>::from_iter(
        signals.as_ref().unwrap().iter().map(|s| s.kind.clone()),
    );
    assert!(
        signal_kinds.len() == 1
            || signal_kinds.len() == 2 && signal_kinds.contains(&cjob::SignalKind::INTEGRATION),
        "AWG signals must be of the same type, or of two types, where one is INTEGRATION. Found: {:?}",
        &signal_kinds
    );
    let awg_id = ob.getattr(intern!(py, "awg_id"))?.extract::<u16>()?;
    let trigger_mode = extract_trigger_mode(&ob.getattr(intern!(py, "trigger_mode"))?)?;
    let reference_clock_source = &ob
        .getattr(intern!(py, "reference_clock_source"))?
        .extract::<Option<String>>()?;
    let is_reference_clock_internal = reference_clock_source
        .as_ref()
        .map(|s| s == "internal")
        .unwrap_or(false);
    let awg = AwgCore::new(
        awg_id,
        extract_awg_kind(&ob.getattr(intern!(py, "signal_type"))?)?,
        signals?,
        sampling_rate,
        Arc::new(Device::new(
            ob.getattr(intern!(py, "device_id"))?
                .extract::<String>()?
                .into(),
            device_kind,
        )),
        extract_awg_oscs(&ob.getattr(intern!(py, "oscs"))?)?,
        Some(trigger_mode),
        is_reference_clock_internal,
    );
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
            let msg = format!("Invalid array type on sweep parameter '{uid}'. {e}");
            return Err(PyValueError::new_err(msg));
        }
    };
    let obj = Arc::new(cjob::SweepParameter {
        uid,
        values: numeric_array,
    });
    dedup.set_parameter(Arc::clone(&obj));
    Ok(obj)
}

pub fn extract_acquisition_type(ob: &Bound<'_, PyAny>) -> Result<AcquisitionType, PyErr> {
    // compilation_job.AcquisitionType
    let value = ob.getattr(intern!(ob.py(), "value"))?;
    match value.cast::<PyString>()?.to_cow()?.as_ref() {
        "integration_trigger" => Ok(AcquisitionType::INTEGRATION),
        "spectroscopy" => Ok(AcquisitionType::SPECTROSCOPY_IQ),
        "spectroscopy_psd" => Ok(AcquisitionType::SPECTROSCOPY_PSD),
        "discrimination" => Ok(AcquisitionType::DISCRIMINATION),
        "RAW" => Ok(AcquisitionType::RAW),
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown acquisition type: {value}"
        ))),
    }
}

/// Transform Python IR to code IR
///
/// While the main purpose of this function is to translate Python IR
/// structs into Rust, it also lowers the source IR into code IR as we
/// do not (yet) have Rust models for the original IR.
pub fn transform_py_ir(
    ob: &Bound<'_, PyAny>,
    awgs: &[AwgCore],
) -> Result<(ir::IrNode, HashMap<PulseParametersId, PulseParameters>), PyErr> {
    let all_signals = awgs
        .iter()
        .flat_map(|awg| awg.signals.iter())
        .collect::<Vec<_>>();
    let mut deduplicator = Deduplicator::new(all_signals);
    let root = extract_node(ob, &mut deduplicator, 0)?
        .into_iter()
        .next()
        .expect("Internal error: No nodes found");
    Ok((root, deduplicator.take_pulse_parameters()))
}

pub fn extract_feedback_register_layout(
    ob: &Bound<'_, PyDict>,
) -> Result<FeedbackRegisterLayout, PyErr> {
    let mut out = FeedbackRegisterLayout::new();
    for (k, v) in ob.iter() {
        let k = if let Ok(device) = k.getattr(intern!(k.py(), "device")) {
            FeedbackRegister::Local {
                device: device.extract::<&str>()?.into(),
            }
        } else if let Ok(source) = k.getattr(intern!(k.py(), "source")) {
            let device = source.getattr(intern!(k.py(), "device_id"))?;
            let awg_key = AwgKey::new(
                device.extract::<&str>()?.into(),
                source
                    .getattr(intern!(k.py(), "awg_id"))?
                    .extract::<u16>()?,
            );
            FeedbackRegister::Global { awg_key }
        } else {
            unreachable!(
                "Internal error: Feedback register key must be either have a 'device' or 'source' attribute"
            );
        };
        let mut register_list_out = Vec::new();
        for item in v.cast::<PyList>()?.iter() {
            let item = item.cast::<PyTuple>()?;
            if item.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "Internal error: Expected tuple of length 2, got: {}",
                    item.len()
                )));
            }
            let width = item.get_item(0)?.extract::<u8>()?;
            let signal = item.get_item(1)?.extract::<Option<String>>()?;
            register_list_out.push(SingleFeedbackRegisterLayoutItem { width, signal });
        }
        out.insert(k, register_list_out);
    }
    Ok(out)
}
