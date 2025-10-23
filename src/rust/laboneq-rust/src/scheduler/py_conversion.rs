// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python conversion layer for Scheduler data structures.
//!
//! The module converts Python DSL Experiment into Rust Scheduler IR.
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use crate::error::{Error, Result};
use anyhow::Context;
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_scheduler::experiment::ExperimentNode;
use laboneq_scheduler::experiment::types::{
    Acquire, AcquisitionType, AveragingLoop, AveragingMode, Case, Delay, ExternalParameterUid,
    HandleUid, Marker, MarkerSelector, Match, MatchTarget, NumericValue, Operation, Parameter,
    ParameterKind, ParameterUid, PlayPulse, PrngLoop, PrngSetup, PulseLength, PulseParameterValue,
    PulseRef, PulseUid, RepetitionMode, Reserve, ResetOscillatorPhase, Section, SectionAlignment,
    SectionUid, SignalUid, Sweep, Trigger, Value,
};
use laboneq_units::duration::seconds;
use num_complex::Complex64;
use numeric_array::NumericArray;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

pub struct ExperimentBuilder<'py> {
    // List of root sections
    pub sections: Vec<ExperimentNode>,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, Parameter>,
    pub pulses: HashMap<PulseUid, PulseRef>,
    pub signals: HashSet<SignalUid>,
    pub external_parameters: HashMap<ExternalParameterUid, Py<PyAny>>,
    pub dsl_types: DslTypes<'py>,
}

impl<'py> ExperimentBuilder<'py> {
    fn new(py: Python<'py>) -> Self {
        Self {
            sections: Vec::new(),
            id_store: NamedIdStore::new(),
            parameters: HashMap::new(),
            pulses: HashMap::new(),
            signals: HashSet::new(),
            external_parameters: HashMap::new(),
            dsl_types: DslTypes::new(py).unwrap(),
        }
    }

    pub(crate) fn register_uid(&mut self, uid: &str) -> NamedId {
        self.id_store.get_or_insert(uid)
    }

    fn add_root_section(&mut self, root: ExperimentNode) {
        self.sections.push(root);
    }

    /// Register a signal that is available for use.
    ///
    /// To match the current behavior, we register experiment signals
    /// even if they are not used.
    fn register_experiment_signal(&mut self, uid: SignalUid) {
        self.signals.insert(uid);
    }

    fn register_pulse(&mut self, pulse: PulseRef) -> PulseUid {
        let uid = pulse.uid;
        self.pulses.entry(uid).or_insert(pulse);
        uid
    }

    fn register_external_parameter(
        &mut self,
        key: &str,
        value: Bound<'_, PyAny>,
    ) -> Result<ExternalParameterUid> {
        let py = value.py();
        let func_create_pulse_parameters_id = py
            .import(intern!(py, "laboneq.compiler.common.pulse_parameters"))?
            .getattr(intern!(py, "create_pulse_parameters_id"))?;
        let py_dict = PyDict::new(py);
        py_dict.set_item(key, &value)?;
        let uid = func_create_pulse_parameters_id
            .call1((py_dict, py.None()))?
            .extract::<u64>()?;
        self.external_parameters
            .insert(ExternalParameterUid(uid), value.unbind());
        Ok(ExternalParameterUid(uid))
    }
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub(crate) enum DslType {
    LinearSweepParameter,
    SweepParameter,
    Parameter,
    Sweep,
    Section,
    Delay,
    Reserve,
    Acquire,
    PlayPulse,
    Match,
    Case,
    PulseFunctional,
    AcquireLoopRt,
    Call,
    PrngSetup,
    PrngLoop,
    SetNode,
    ResetOscillatorPhase,
}

pub(crate) struct DslTypes<'a> {
    type_map: HashMap<DslType, Bound<'a, PyAny>>,
}

impl<'a> DslTypes<'a> {
    fn new(py: Python<'a>) -> Result<DslTypes<'a>> {
        let linear_sweep_parameter_py = py
            .import(intern!(py, "laboneq.data.parameter"))?
            .getattr(intern!(py, "LinearSweepParameter"))?;
        let sweep_parameter_py: Bound<'_, PyAny> = py
            .import(intern!(py, "laboneq.data.parameter"))?
            .getattr(intern!(py, "SweepParameter"))?;
        let parameter_py: Bound<'_, PyAny> = py
            .import(intern!(py, "laboneq.data.parameter"))?
            .getattr(intern!(py, "Parameter"))?;
        let sweep_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Sweep"))?;
        let section_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Section"))?;
        let delay_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Delay"))?;
        let reserve_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Reserve"))?;
        let acquire_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Acquire"))?;
        let play_pulse_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "PlayPulse"))?;
        let match_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Match"))?;
        let case_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Case"))?;
        let pulse_functional_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "PulseFunctional"))?;
        let acquire_loop_rt_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "AcquireLoopRt"))?;
        let neartime_callback = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "Call"))?;
        let prng_setup_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "PrngSetup"))?;
        let prng_loop_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "PrngLoop"))?;
        let set_node_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "SetNode"))?;
        let reset_oscillator_phase_py = py
            .import(intern!(py, "laboneq.data.experiment_description"))?
            .getattr(intern!(py, "ResetOscillatorPhase"))?;
        let type_map = HashMap::from([
            (DslType::LinearSweepParameter, linear_sweep_parameter_py),
            (DslType::SweepParameter, sweep_parameter_py),
            (DslType::Parameter, parameter_py),
            (DslType::Sweep, sweep_py),
            (DslType::Match, match_py),
            (DslType::Section, section_py),
            (DslType::Delay, delay_py),
            (DslType::Reserve, reserve_py),
            (DslType::Acquire, acquire_py),
            (DslType::PlayPulse, play_pulse_py),
            (DslType::Case, case_py),
            (DslType::PulseFunctional, pulse_functional_py),
            (DslType::AcquireLoopRt, acquire_loop_rt_py),
            (DslType::Call, neartime_callback),
            (DslType::PrngSetup, prng_setup_py),
            (DslType::PrngLoop, prng_loop_py),
            (DslType::SetNode, set_node_py),
            (DslType::ResetOscillatorPhase, reset_oscillator_phase_py),
        ]);
        Ok(Self { type_map })
    }

    pub(crate) fn laboneq_type(&self, dsl_type: DslType) -> &Bound<'a, PyAny> {
        self.type_map
            .get(&dsl_type)
            .unwrap_or_else(|| panic!("DSL type not found: {dsl_type:?}"))
    }
}

/// Extract an integer.
///
/// Any real number is valid and is casted to integer.
fn extract_integer<T: TryFrom<i64>>(obj: &Bound<'_, PyAny>) -> Result<T> {
    let py = obj.py();
    // Real or enforce integral numbers only?
    let py_numbers_real = PyModule::import(py, "numbers")?.getattr(intern!(py, "Real"))?;
    if !obj.is_instance(&py_numbers_real)? {
        return Err(Error::new("Expected a real number"));
    }
    let func_int = PyModule::import(py, "builtins")?.getattr(intern!(py, "int"))?;
    let value = func_int.call1((obj,))?.extract::<i64>()?;
    value
        .try_into()
        .map_err(|_| Error::new("Failed to convert integer"))
}

pub fn extract_value(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<Option<Value>> {
    if obj.is_none() {
        return Ok(None);
    }
    let py: Python<'_> = obj.py();
    if obj.is_instance(&py.get_type::<pyo3::types::PyFloat>())? {
        let value = obj.extract::<f64>()?;
        Ok(Some(Value::Float(value)))
    } else if obj.is_instance(&py.get_type::<pyo3::types::PyInt>())? {
        let value = obj.extract::<i64>()?;
        Ok(Some(Value::Int(value)))
    } else if obj.is_instance(&py.get_type::<pyo3::types::PyBool>())? {
        let value = obj.extract::<bool>()?;
        Ok(Some(Value::Bool(value)))
    } else if obj.is_instance(&py.get_type::<pyo3::types::PyComplex>())? {
        let value = obj.getattr(intern!(py, "real"))?.extract::<f64>()?;
        let value_imag = obj.getattr(intern!(py, "imag"))?.extract::<f64>()?;
        Ok(Some(Value::Complex(Complex64::new(value, value_imag))))
    } else if obj.is_instance(&py.get_type::<PyString>())? {
        let value = obj.extract::<String>()?;
        Ok(Some(Value::String(value)))
    } else if obj.is_instance(builder.dsl_types.laboneq_type(DslType::Parameter))? {
        let param_uid = &obj.getattr(intern!(py, "uid"))?.extract::<String>()?;
        let parameter = builder.register_uid(param_uid);
        Ok(Some(Value::ParameterUid(ParameterUid(parameter))))
    } else {
        Err(Error::new(
            "Expected a numeric literal, string, or parameter",
        ))
    }
}

fn extract_section_alignment(obj: &Bound<'_, PyAny>) -> Result<SectionAlignment> {
    if obj.is_none() {
        return Ok(SectionAlignment::Left); // Default alignment
    }
    let py = obj.py();
    let out = match obj
        .getattr(intern!(py, "name"))?
        .downcast_into::<PyString>()
        .map_err(PyErr::from)?
        .to_cow()?
        .as_ref()
    {
        "LEFT" => SectionAlignment::Left,
        "RIGHT" => SectionAlignment::Right,
        _ => {
            return Err(Error::new(format!("Unknown section alignment: {obj}")));
        }
    };
    Ok(out)
}

fn extract_parameter(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<ParameterUid> {
    let py: Python<'_> = obj.py();
    let uid =
        ParameterUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
    if builder.parameters.contains_key(&uid) {
        return Ok(uid);
    }

    let linear_sweep_parameter_py = builder
        .dsl_types
        .laboneq_type(DslType::LinearSweepParameter);
    let sweep_parameter_py = builder.dsl_types.laboneq_type(DslType::SweepParameter);

    if obj.is_instance(linear_sweep_parameter_py)? {
        let start = extract_value(&obj.getattr(intern!(py, "start"))?, builder)?;
        let stop = extract_value(&obj.getattr(intern!(py, "stop"))?, builder)?;
        let count = obj.getattr(intern!(py, "count"))?.extract::<usize>()?;
        let kind = ParameterKind::Linear {
            start: start.map_or(Err(Error::new("Linear sweep start must be defined")), |v| {
                v.try_into().map_err(Error::new)
            })?,
            stop: stop.map_or(Err(Error::new("Linear sweep stop must be defined")), |v| {
                v.try_into().map_err(Error::new)
            })?,
            count,
        };
        let parameter = Parameter { uid, kind };
        builder.parameters.insert(parameter.uid, parameter);
        return Ok(uid);
    } else if obj.is_instance(sweep_parameter_py)? {
        let values = NumericArray::from_py(&obj.getattr(intern!(py, "values"))?)?;
        let kind = ParameterKind::Array { values };
        let parameter = Parameter { uid, kind };
        builder.parameters.insert(parameter.uid, parameter);
        return Ok(uid);
    }
    Err(Error::new(
        "Expected 'LinearSweepParameter' or 'SweepParameter'",
    ))
}

fn extract_sweep(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Sweep> {
    let mut parameters = vec![];
    for param in obj.getattr(intern!(obj.py(), "parameters"))?.try_iter()? {
        let param = param?;
        let parameter_uid = extract_parameter(&param, builder)?;
        parameters.push(parameter_uid);
    }
    let uid_py = obj.getattr(intern!(obj.py(), "uid"))?.extract::<String>()?;
    let alignment = extract_section_alignment(&obj.getattr(intern!(obj.py(), "alignment"))?)?;
    let reset_oscillator_phase = obj
        .getattr(intern!(obj.py(), "reset_oscillator_phase"))?
        .extract::<bool>()?;
    let obj = Sweep {
        uid: SectionUid(builder.register_uid(&uid_py)),
        parameters,
        alignment,
        reset_oscillator_phase,
    };
    Ok(obj)
}

fn extract_delay(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Delay> {
    let py = obj.py();
    let signal_name = obj.getattr(intern!(py, "signal"))?.extract::<String>()?;
    let signal_uid = builder.register_uid(&signal_name);
    let time = extract_value(&obj.getattr(intern!(py, "time"))?, builder)?;

    let precompensation_clear = obj
        .getattr(intern!(obj.py(), "precompensation_clear"))?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    let out = Delay {
        signal: SignalUid(signal_uid),
        time: time.map_or_else(
            || Err(Error::new("Delay time cannot be None")),
            |v| v.try_into().map_err(Error::new),
        )?,
        precompensation_clear,
    };
    Ok(out)
}

fn extract_reserve(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> PyResult<Reserve> {
    let py = obj.py();
    let signal_name = obj.getattr(intern!(py, "signal"))?.extract::<String>()?;
    let signal_uid = builder.register_uid(&signal_name);
    let out = Reserve {
        signal: SignalUid(signal_uid),
    };
    Ok(out)
}

fn extract_pulse_parameters(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<HashMap<Arc<String>, PulseParameterValue>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let obj = obj.downcast::<PyDict>().map_err(PyErr::from)?;
    let mut out = HashMap::new();
    for (key, value) in obj.iter() {
        let key = Arc::new(key.extract::<String>()?);
        if value.is_instance(builder.dsl_types.laboneq_type(DslType::Parameter))? {
            let param_uid = &value
                .getattr(intern!(obj.py(), "uid"))?
                .extract::<String>()?;
            out.insert(
                Arc::clone(&key),
                PulseParameterValue::Parameter(ParameterUid(builder.register_uid(param_uid))),
            );
        } else {
            out.insert(
                Arc::clone(&key),
                PulseParameterValue::ExternalParameter(
                    builder.register_external_parameter(&key, value)?,
                ),
            );
        }
    }
    Ok(out)
}

/// Convert `Pulse` and its subclasses
fn extract_pulse(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<(PulseUid, HashMap<Arc<String>, PulseParameterValue>)> {
    let py = obj.py();
    let uid = PulseUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
    let length = if obj.is_instance(builder.dsl_types.laboneq_type(DslType::PulseFunctional))? {
        PulseLength::Seconds(seconds(
            obj.getattr(intern!(py, "length"))?.extract::<f64>()?,
        ))
    } else {
        // PulseSampled
        let samples_py_arr = obj.getattr(intern!(py, "samples"))?;
        PulseLength::Samples(samples_py_arr.len()?)
    };
    let pulse = PulseRef { uid, length };
    let pulse_parameters =
        if obj.is_instance(builder.dsl_types.laboneq_type(DslType::PulseFunctional))? {
            extract_pulse_parameters(&obj.getattr(intern!(py, "pulse_parameters"))?, builder)?
        } else {
            HashMap::new()
        };
    Ok((builder.register_pulse(pulse), pulse_parameters))
}

fn extract_acquire(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Acquire> {
    let py = obj.py();
    let signal_name = obj.getattr(intern!(py, "signal"))?.extract::<String>()?;
    let handle = obj.getattr(intern!(py, "handle"))?.extract::<String>()?;

    let mut kernels = vec![];
    let kernel = obj.getattr(intern!(py, "kernel"))?;
    let mut pulse_parameters = vec![];
    if !kernel.is_none() {
        if kernel.is_instance(&py.get_type::<PyList>())? {
            for kernel in kernel.try_iter()? {
                let kernel = kernel?;
                let (pulse_uid, parameters) = extract_pulse(&kernel, builder)?;
                kernels.push(pulse_uid);
                pulse_parameters.push(parameters);
            }
        } else {
            let (pulse_uid, parameters) = extract_pulse(&kernel, builder)?;
            kernels.push(pulse_uid);
            pulse_parameters.push(parameters);
        }
    }
    let length = obj
        .getattr(intern!(py, "length"))?
        .extract::<Option<f64>>()?;
    let play_parameters = obj.getattr(intern!(py, "pulse_parameters"))?;
    let mut parameters = vec![];
    if play_parameters.is_instance(&py.get_type::<PyList>())? {
        for param in play_parameters.try_iter()? {
            let param = param?;
            let pulse_param = extract_pulse_parameters(&param, builder)?;
            parameters.push(pulse_param);
        }
    } else {
        let pulse_param = extract_pulse_parameters(&play_parameters, builder)?;
        parameters.push(pulse_param);
    }
    let out = Acquire {
        signal: SignalUid(builder.register_uid(&signal_name)),
        handle: HandleUid(builder.register_uid(&handle)),
        kernel: kernels,
        length: length.map(seconds),
        parameters,
        pulse_parameters,
    };
    Ok(out)
}

/// Convert dictionary specifying one of two markers
fn extract_marker_dict(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<Vec<Marker>> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(vec![]);
    }
    let mut markers = vec![];
    for (marker_selector_name, marker) in obj.downcast::<PyDict>().map_err(PyErr::from)?.iter() {
        let marker_selector = match marker_selector_name.extract::<&str>()? {
            "marker1" => MarkerSelector::M1,
            "marker2" => MarkerSelector::M2,
            other => {
                return Err(Error::new(format!("Unknown marker selector: {other}")));
            }
        };
        let marker = marker.downcast::<PyDict>().map_err(PyErr::from)?;
        let enable = marker
            .get_item(intern!(py, "enable"))?
            .map_or(Ok(false), |o: Bound<'_, PyAny>| -> PyResult<bool> {
                o.extract::<bool>()
            })?;
        let start = marker
            .get_item(intern!(py, "start"))?
            .map(|o| -> PyResult<f64> { o.extract::<f64>() })
            .transpose()?;
        let length = marker
            .get_item(intern!(py, "length"))?
            .map(|o| -> PyResult<f64> { o.extract::<f64>() })
            .transpose()?;
        let pulse_id = marker
            .get_item(intern!(py, "waveform"))?
            .map(|o| -> Result<PulseUid> {
                let (pulse, _) = extract_pulse(&o, builder)?;
                Ok(pulse)
            })
            .transpose()?;
        let marker = Marker {
            marker_selector,
            enable,
            start: start.map(seconds),
            length: length.map(seconds),
            pulse_id,
        };
        markers.push(marker);
    }
    Ok(markers)
}

/// Convert `PlayPulse`
fn extract_play_pulse(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<PlayPulse> {
    let py = obj.py();
    let signal = obj.getattr(intern!(py, "signal"))?.extract::<String>()?;
    let maybe_pulse = &obj.getattr(intern!(py, "pulse"))?;
    let pulse = if !maybe_pulse.is_none() {
        Some(extract_pulse(maybe_pulse, builder)?)
    } else {
        None
    };
    let amplitude: Option<NumericValue> =
        extract_value(&obj.getattr(intern!(py, "amplitude"))?, builder)?
            .map(|v| v.try_into().expect("Amplitude must be a real value"));
    let precompensation_clear = obj
        .getattr(intern!(py, "precompensation_clear"))?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    let phase = extract_value(&obj.getattr(intern!(py, "phase"))?, builder)?;

    let increment_oscillator_phase = extract_value(
        &obj.getattr(intern!(py, "increment_oscillator_phase"))?,
        builder,
    )?;
    let set_oscillator_phase =
        extract_value(&obj.getattr(intern!(py, "set_oscillator_phase"))?, builder)?;

    let length = extract_value(&obj.getattr(intern!(py, "length"))?, builder)?;
    let parameters =
        extract_pulse_parameters(&obj.getattr(intern!(py, "pulse_parameters"))?, builder)?;
    let markers = extract_marker_dict(&obj.getattr(intern!(py, "marker"))?, builder)?;
    let pulse = PlayPulse {
        signal: SignalUid(builder.register_uid(&signal)),
        pulse: pulse.as_ref().map(|x| x.0),
        amplitude: amplitude.unwrap_or(NumericValue::Float(1.0)),
        phase: phase.map(|v| v.try_into().expect("Phase must be a real value")),
        increment_oscillator_phase: increment_oscillator_phase.map(|v| {
            v.try_into()
                .expect("Increment oscillator phase must be a real value")
        }),
        set_oscillator_phase: set_oscillator_phase.map(|v| {
            v.try_into()
                .expect("Set oscillator phase must be a real value")
        }),
        length: length.map(|v| v.try_into().expect("Length must be a real value")),
        precompensation_clear,
        parameters,
        pulse_parameters: pulse.map(|x| x.1).unwrap_or_default(),
        markers,
    };
    Ok(pulse)
}

/// Convert `Case`
fn extract_case(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Case> {
    let py = obj.py();
    let uid =
        SectionUid(builder.register_uid(&obj.getattr(intern!(py, "uid"))?.extract::<String>()?));
    let state = extract_integer(&obj.getattr(intern!(py, "state"))?)?;
    let out = Case { uid, state };
    Ok(out)
}

/// Convert `Match`
fn extract_match(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Match> {
    let py = obj.py();
    let uid = SectionUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
    let local = obj
        .getattr(intern!(py, "local"))?
        .extract::<Option<bool>>()?;

    let handle = obj.getattr(intern!(py, "handle"))?;
    let user_register = obj.getattr(intern!(py, "user_register"))?;
    let sweep_parameter = obj.getattr(intern!(py, "sweep_parameter"))?;
    let prng_sample = obj.getattr(intern!(py, "prng_sample"))?;

    let match_target = if !handle.is_none() {
        MatchTarget::Handle(HandleUid(
            builder.register_uid(obj.getattr(intern!(py, "handle"))?.extract::<&str>()?),
        ))
    } else if !user_register.is_none() {
        let value = obj
            .getattr(intern!(py, "user_register"))?
            .extract::<u16>()?;
        MatchTarget::UserRegister(value)
    } else if !sweep_parameter.is_none() {
        let parameter_uid =
            builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?);
        MatchTarget::SweepParameter(ParameterUid(parameter_uid))
    } else if !prng_sample.is_none() {
        let parameter_uid =
            builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?);
        MatchTarget::PrngSample(SectionUid(parameter_uid))
    } else {
        return Err(Error::new(
            "Match must have one of handle, user_register, sweep_parameter, or prng_sample defined",
        ));
    };
    let out = Match {
        uid,
        target: match_target,
        local,
        play_after: extract_play_after(&obj.getattr(intern!(py, "play_after"))?, builder)?,
    };
    Ok(out)
}

fn extract_play_after(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<Vec<SectionUid>> {
    let py = obj.py();
    let mut play_after = vec![];
    if obj.is_none() {
        return Ok(play_after);
    }
    // List of section UIDs or Section objects
    if obj.is_instance(&py.get_type::<PyList>())? {
        for section in obj.try_iter()? {
            let section = section?;
            if section.is_instance(&py.get_type::<PyString>())? {
                play_after.push(SectionUid(builder.register_uid(section.extract::<&str>()?)))
            } else {
                play_after.push(SectionUid(
                    builder.register_uid(section.getattr(intern!(py, "uid"))?.extract::<&str>()?),
                ));
            }
        }
    } else {
        // A single section UID or Section object
        if obj.is_instance(&py.get_type::<PyString>())? {
            play_after.push(SectionUid(builder.register_uid(obj.extract::<&str>()?)))
        } else {
            play_after.push(SectionUid(
                builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?),
            ));
        }
    }
    Ok(play_after)
}

/// Convert `Section`
fn extract_section(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Section> {
    let py = obj.py();
    let uid =
        SectionUid(builder.register_uid(&obj.getattr(intern!(py, "uid"))?.extract::<String>()?));
    let alignment = extract_section_alignment(&obj.getattr(intern!(py, "alignment"))?)?;
    let on_system_grid = obj
        .getattr(intern!(py, "on_system_grid"))?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    let length = obj
        .getattr(intern!(py, "length"))?
        .extract::<Option<f64>>()?
        .map(seconds);
    let play_after = extract_play_after(&obj.getattr(intern!(py, "play_after"))?, builder)?;
    let trigger_py = obj.getattr(intern!(py, "trigger"))?;
    let trigger_map = trigger_py.downcast::<PyDict>().map_err(PyErr::from)?;
    let mut triggers = vec![];
    for (signal_uid, trigger) in trigger_map.iter() {
        let trigger_signal = signal_uid.extract::<String>()?;
        let trigger_state = trigger.get_item("state")?.extract::<u16>()?;
        let trigger = Trigger {
            signal: SignalUid(builder.register_uid(&trigger_signal)),
            state: trigger_state,
        };
        triggers.push(trigger)
    }

    let out = Section {
        uid,
        alignment,
        length,
        play_after,
        triggers,
        on_system_grid,
    };
    Ok(out)
}

fn extract_averaging_mode(obj: &Bound<'_, PyAny>) -> Result<AveragingMode> {
    if obj.is_none() {
        return Ok(AveragingMode::Cyclic); // Default averaging mode
    }
    let py = obj.py();
    let out = match obj
        .getattr(intern!(py, "name"))?
        .downcast_into::<PyString>()
        .map_err(PyErr::from)?
        .to_cow()?
        .as_ref()
    {
        "SEQUENTIAL" => AveragingMode::Sequential,
        "CYCLIC" => AveragingMode::Cyclic,
        "SINGLE_SHOT" => AveragingMode::SingleShot,
        _ => {
            return Err(Error::new(format!("Unknown averaging mode: {obj}")));
        }
    };
    Ok(out)
}

fn extract_repetition_mode(
    obj: &Bound<'_, PyAny>,
    repetition_time: Option<f64>,
) -> Result<RepetitionMode> {
    if obj.is_none() {
        return Ok(RepetitionMode::Fastest); // Default repetition mode
    }
    let py = obj.py();
    let out = match obj
        .getattr(intern!(py, "name"))?
        .downcast_into::<PyString>()
        .map_err(PyErr::from)?
        .to_cow()?
        .as_ref()
    {
        "FASTEST" => RepetitionMode::Fastest,
        "CONSTANT" => {
            if repetition_time.is_none() {
                return Err(Error::new(
                    "Repetition time must be set for CONSTANT repetition mode",
                ));
            }
            RepetitionMode::Constant {
                time: repetition_time.unwrap(),
            }
        }
        "AUTO" => RepetitionMode::Auto,
        _ => {
            return Err(Error::new(format!("Unknown repetition mode: {obj}")));
        }
    };
    Ok(out)
}

fn extract_averaging_loop(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<AveragingLoop> {
    let uid_py = obj.getattr(intern!(obj.py(), "uid"))?.extract::<String>()?;
    // Currently DSL supports `count` to be a floating point number, but it must be integral
    let count: u32 = extract_integer(&obj.getattr(intern!(obj.py(), "count"))?)?;
    let reset_oscillator_phase = obj
        .getattr(intern!(obj.py(), "reset_oscillator_phase"))?
        .extract::<bool>()?;
    let acquisition_type = AcquisitionType::Integration;
    let averaging_mode =
        extract_averaging_mode(&obj.getattr(intern!(obj.py(), "averaging_mode"))?)?;
    let repetition_time = obj
        .getattr(intern!(obj.py(), "repetition_time"))?
        .extract::<Option<f64>>()?;
    let repetition_mode = extract_repetition_mode(
        &obj.getattr(intern!(obj.py(), "repetition_mode"))?,
        repetition_time,
    )?;
    let obj = AveragingLoop {
        uid: SectionUid(builder.register_uid(&uid_py)),
        count,
        reset_oscillator_phase,
        acquisition_type,
        averaging_mode,
        repetition_mode,
        alignment: SectionAlignment::Left, // Averaging loops are always left aligned by default
    };
    Ok(obj)
}

fn extract_prng_setup(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<PrngSetup> {
    let py = obj.py();
    let uid = builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?);
    let prng_py = obj.getattr(intern!(py, "prng"))?;
    let obj = PrngSetup {
        uid: SectionUid(uid),
        range: prng_py.getattr(intern!(py, "range"))?.extract::<u32>()?,
        seed: prng_py.getattr(intern!(py, "seed"))?.extract::<u32>()?,
    };
    Ok(obj)
}

fn extract_prng_loop(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<PrngLoop> {
    let py = obj.py();
    let uid = builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?);
    let prng_sample_py = obj.getattr(intern!(py, "prng_sample"))?;
    let obj = PrngLoop {
        uid: SectionUid(uid),
        count: prng_sample_py
            .getattr(intern!(py, "count"))?
            .extract::<u32>()?,
    };
    Ok(obj)
}

fn extract_reset_oscillator_phase(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<ResetOscillatorPhase> {
    let py = obj.py();
    let signal_uid = obj
        .getattr(intern!(py, "signal"))?
        .extract::<Option<String>>()?;
    let obj = ResetOscillatorPhase {
        signal: signal_uid.map(|uid| SignalUid(builder.register_uid(&uid))),
    };
    Ok(obj)
}

fn extract_object(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Operation> {
    let variant = if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::PlayPulse))
    {
        Operation::PlayPulse(extract_play_pulse(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Section))
    {
        Operation::Section(extract_section(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Delay))
    {
        Operation::Delay(extract_delay(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Reserve))
    {
        Operation::Reserve(extract_reserve(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Acquire))
    {
        Operation::Acquire(extract_acquire(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Match))
    {
        Operation::Match(extract_match(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Case))
    {
        Operation::Case(extract_case(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::AcquireLoopRt))
    {
        Operation::AveragingLoop(extract_averaging_loop(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Sweep))
    {
        Operation::Sweep(extract_sweep(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::Call))
    {
        Operation::NearTimeCallback
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::PrngSetup))
    {
        Operation::PrngSetup(extract_prng_setup(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::PrngLoop))
    {
        Operation::PrngLoop(extract_prng_loop(obj, builder)?)
    } else if obj
        .get_type()
        .is(builder.dsl_types.laboneq_type(DslType::SetNode))
    {
        Operation::SetNode
    } else if obj.get_type().is(builder
        .dsl_types
        .laboneq_type(DslType::ResetOscillatorPhase))
    {
        Operation::ResetOscillatorPhase(extract_reset_oscillator_phase(obj, builder)?)
    } else {
        return Err(Error::new(format!("Unknown experiment object type: {obj}")));
    };
    Ok(variant)
}

/// Recursively traverse the experiment tree and create the nodes.
fn traverse_experiment(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<ExperimentNode> {
    let variant = extract_object(obj, builder)
        .with_context(|| format!("Error while handling object: '{obj}'"))?;
    let mut node = ExperimentNode::new(variant);
    if let Some(children) = obj.getattr_opt(intern!(obj.py(), "children"))? {
        node.children.reserve(children.len()?);
        for child in children.try_iter()? {
            let child_value = traverse_experiment(&child?, builder)?;
            node.children.push(child_value.into());
        }
    }
    Ok(node)
}

/// Convert `ExperimentSignal`
fn extract_experiment_signal(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<SignalUid> {
    let py = obj.py();
    let uid = obj.getattr(intern!(py, "uid"))?.extract::<String>()?;
    let signal = SignalUid(builder.register_uid(&uid));
    Ok(signal)
}

/// Convert `Experiment` into Rust Scheduler IR.
pub fn build_experiment<'py>(experiment: &Bound<'py, PyAny>) -> Result<ExperimentBuilder<'py>> {
    let mut builder = ExperimentBuilder::new(experiment.py());
    let signals = experiment.getattr(intern!(experiment.py(), "signals"))?;
    for signal in signals.try_iter()? {
        let signal_uid = extract_experiment_signal(&signal?, &mut builder)?;
        builder.register_experiment_signal(signal_uid);
    }
    for section in experiment
        .getattr(intern!(experiment.py(), "sections"))?
        .try_iter()?
    {
        let root_section = traverse_experiment(&section?, &mut builder)?;
        let mut root = ExperimentNode::new(Operation::Root);
        root.children.push(root_section.into());
        builder.add_root_section(root);
    }
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::ffi::c_str;
    use std::env;

    #[macro_export]
    macro_rules! include_py_file {
        ($path:literal) => {
            c_str!(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $path)))
        };
    }

    #[test]
    /// This test will test that the experiment building works as expected.
    ///
    /// The experiment is defined in a Python file (`src/scheduler/test_dsl_experiment`), which is then executed
    /// and the experiment is then tested.
    fn test_schedule_experiment() {
        let py_testfile = include_py_file!("/src/scheduler/test_dsl_experiment.py");
        Python::attach(|py| {
            // Load the testfile and import test function
            let module =
                PyModule::from_code(py, py_testfile, c_str!("testfile.py"), c_str!("")).unwrap();
            let run_experiment = module.getattr("run_experiment").unwrap();
            let experiment = run_experiment.call0().unwrap();

            // Test Experiment building
            let builder = build_experiment(&experiment).unwrap();
            let id_store = builder.id_store;
            // Test sweep parameter collection
            let parameter = builder.parameters.iter().next().unwrap().1;
            assert_eq!(id_store.resolve(parameter.uid).unwrap(), "sweep_param123");
            // Test experiment signals
            let experiment_signals: Vec<&str> = builder
                .signals
                .iter()
                .map(|s| id_store.resolve(*s).unwrap())
                .collect();
            assert_eq!(experiment_signals[0], "q0/drive");
            // Test external pulse parameters
            // Sweep parameter is not external, therefore not stored
            assert_eq!(builder.external_parameters.len(), 1);
            let external_pulse_parameter = builder.external_parameters.values().next().unwrap();
            assert!(
                external_pulse_parameter
                    .bind(py)
                    .eq((0.5).into_pyobject(py).unwrap())
                    .unwrap(),
            );
        });
    }
}
