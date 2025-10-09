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
use laboneq_scheduler::IrNode;
use laboneq_scheduler::ir::{
    Acquire, AcquireLoopRt, AcquisitionType, AveragingMode, Case, Delay, ExecutionType,
    ExternalParameterUid, HandleUid, IrVariant, Marker, MarkerSelector, Match, MatchTarget,
    NumericValue, Parameter, ParameterKind, ParameterUid, PlayPulse, PulseLength,
    PulseParameterValue, PulseRef, PulseUid, RepetitionMode, Reserve, Section, SectionAlignment,
    SectionUid, SignalUid, Sweep, Trigger, Value,
};
use laboneq_units::duration::seconds;
use num_complex::Complex64;
use numeric_array::NumericArray;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

pub struct ExperimentBuilder {
    // List of root sections
    pub sections: Vec<IrNode>,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, Parameter>,
    pub pulses: HashMap<PulseUid, PulseRef>,
    pub signals: HashSet<SignalUid>,
    pub external_parameters: HashMap<ExternalParameterUid, Py<PyAny>>,
}

impl ExperimentBuilder {
    fn new() -> Self {
        Self {
            sections: Vec::new(),
            id_store: NamedIdStore::new(),
            parameters: HashMap::new(),
            pulses: HashMap::new(),
            signals: HashSet::new(),
            external_parameters: HashMap::new(),
        }
    }

    fn register_uid(&mut self, uid: &str) -> NamedId {
        self.id_store.get_or_insert(uid)
    }

    fn add_root_section(&mut self, root: IrNode) {
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
enum DslType {
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
}

struct DslTypes<'a> {
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
        ]);
        Ok(Self { type_map })
    }

    fn laboneq_type(&self, dsl_type: DslType) -> &Bound<'a, PyAny> {
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

fn extract_value(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
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
    } else if obj.is_instance(laboneq_types.laboneq_type(DslType::Parameter))? {
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

fn extract_execution_type(obj: &Bound<'_, PyAny>) -> Result<ExecutionType> {
    // Not actually read by the scheduler
    if obj.is_none() {
        return Ok(ExecutionType::RealTime); // Default execution type, feasible?
    }
    let out = match obj
        .getattr(intern!(obj.py(), "name"))?
        .downcast_into::<PyString>()
        .map_err(PyErr::from)?
        .to_cow()?
        .as_ref()
    {
        "NEAR_TIME" => ExecutionType::NearTime,
        "REAL_TIME" => ExecutionType::RealTime,
        _ => {
            return Err(Error::new(format!("Unknown execution type: {obj}")));
        }
    };
    Ok(out)
}

fn extract_parameter(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
) -> Result<Parameter> {
    let py: Python<'_> = obj.py();
    let uid_name = obj.getattr(intern!(py, "uid"))?.extract::<String>()?;
    let uid = builder.register_uid(&uid_name);
    let linear_sweep_parameter_py = laboneq_types.laboneq_type(DslType::LinearSweepParameter);
    let sweep_parameter_py = laboneq_types.laboneq_type(DslType::SweepParameter);

    if obj.is_instance(linear_sweep_parameter_py)? {
        let start = extract_value(&obj.getattr(intern!(py, "start"))?, builder, laboneq_types)?;
        let stop = extract_value(&obj.getattr(intern!(py, "stop"))?, builder, laboneq_types)?;
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
        let parameter = Parameter {
            uid: ParameterUid(uid),
            kind,
        };
        return Ok(parameter);
    } else if obj.is_instance(sweep_parameter_py)? {
        let values = NumericArray::from_py(&obj.getattr(intern!(py, "values"))?)?;
        let kind = ParameterKind::Array { values };
        let parameter = Parameter {
            uid: ParameterUid(uid),
            kind,
        };
        return Ok(parameter);
    }
    Err(Error::new(
        "Expected 'LinearSweepParameter' or 'SweepParameter'",
    ))
}

fn extract_sweep(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
) -> Result<Sweep> {
    let mut parameters = vec![];
    for param in obj.getattr(intern!(obj.py(), "parameters"))?.try_iter()? {
        let param = param?;
        let parameter = extract_parameter(&param, builder, laboneq_types)?;
        parameters.push(parameter.uid);
        builder.parameters.insert(parameter.uid, parameter);
    }
    let uid_py = obj.getattr(intern!(obj.py(), "uid"))?.extract::<String>()?;
    let alignment = extract_section_alignment(&obj.getattr(intern!(obj.py(), "alignment"))?)?;
    let reset_oscillator_phase = obj
        .getattr(intern!(obj.py(), "reset_oscillator_phase"))?
        .extract::<bool>()?;
    let execution_type =
        extract_execution_type(&obj.getattr(intern!(obj.py(), "execution_type"))?)?;
    let obj = Sweep {
        uid: SectionUid(builder.register_uid(&uid_py)),
        parameters,
        alignment,
        reset_oscillator_phase,
        execution_type,
    };
    Ok(obj)
}

fn extract_delay(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
) -> Result<Delay> {
    let py = obj.py();
    let signal_name = obj.getattr(intern!(py, "signal"))?.extract::<String>()?;
    let signal_uid = builder.register_uid(&signal_name);
    let time = extract_value(&obj.getattr(intern!(py, "time"))?, builder, laboneq_types)?;

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
    laboneq_types: &DslTypes,
) -> Result<HashMap<Arc<String>, PulseParameterValue>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let obj = obj.downcast::<PyDict>().map_err(PyErr::from)?;
    let mut out = HashMap::new();
    for (key, value) in obj.iter() {
        let key = Arc::new(key.extract::<String>()?);
        if value.is_instance(laboneq_types.laboneq_type(DslType::Parameter))? {
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
    laboneq_types: &DslTypes,
) -> Result<(PulseUid, HashMap<Arc<String>, PulseParameterValue>)> {
    let py = obj.py();
    let name = obj.getattr(intern!(py, "uid"))?.extract::<String>()?;
    let uid = PulseUid(builder.register_uid(&name));
    let length = if obj.is_instance(laboneq_types.laboneq_type(DslType::PulseFunctional))? {
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
        if obj.is_instance(laboneq_types.laboneq_type(DslType::PulseFunctional))? {
            extract_pulse_parameters(
                &obj.getattr(intern!(py, "pulse_parameters"))?,
                builder,
                laboneq_types,
            )?
        } else {
            HashMap::new()
        };
    Ok((builder.register_pulse(pulse), pulse_parameters))
}

fn extract_acquire(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
) -> Result<Acquire> {
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
                let (pulse_uid, parameters) = extract_pulse(&kernel, builder, laboneq_types)?;
                kernels.push(pulse_uid);
                pulse_parameters.push(parameters);
            }
        } else {
            let (pulse_uid, parameters) = extract_pulse(&kernel, builder, laboneq_types)?;
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
            let pulse_param = extract_pulse_parameters(&param, builder, laboneq_types)?;
            parameters.push(pulse_param);
        }
    } else {
        let pulse_param = extract_pulse_parameters(&play_parameters, builder, laboneq_types)?;
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
    dsl_types: &DslTypes,
) -> Result<Vec<Marker>> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(vec![]);
    }
    let mut markers = vec![];
    for (marker_selector_name, marker) in obj.downcast::<PyDict>().map_err(PyErr::from)?.iter() {
        let marker_selector = match marker_selector_name
            .downcast::<PyString>()
            .map_err(PyErr::from)?
            .to_cow()?
            .as_ref()
        {
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
            .get_item(intern!(py, "pulse_uid"))?
            .map(|o| -> Result<PulseUid> {
                let (pulse, _) = extract_pulse(&o, builder, dsl_types)?;
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
    dsl_types: &DslTypes,
) -> Result<PlayPulse> {
    let py = obj.py();
    let signal = obj.getattr(intern!(py, "signal"))?.extract::<String>()?;
    let maybe_pulse = &obj.getattr(intern!(py, "pulse"))?;
    let pulse = if !maybe_pulse.is_none() {
        Some(extract_pulse(maybe_pulse, builder, dsl_types)?)
    } else {
        None
    };
    let amplitude: Option<NumericValue> =
        extract_value(&obj.getattr(intern!(py, "amplitude"))?, builder, dsl_types)?
            .map(|v| v.try_into().expect("Amplitude must be a real value"));
    let precompensation_clear = obj
        .getattr(intern!(py, "precompensation_clear"))?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    let phase = extract_value(&obj.getattr(intern!(py, "phase"))?, builder, dsl_types)?;

    let increment_oscillator_phase = extract_value(
        &obj.getattr(intern!(py, "increment_oscillator_phase"))?,
        builder,
        dsl_types,
    )?;
    let set_oscillator_phase = extract_value(
        &obj.getattr(intern!(py, "set_oscillator_phase"))?,
        builder,
        dsl_types,
    )?;

    let length = extract_value(&obj.getattr(intern!(py, "length"))?, builder, dsl_types)?;
    let parameters = extract_pulse_parameters(
        &obj.getattr(intern!(py, "pulse_parameters"))?,
        builder,
        dsl_types,
    )?;
    let markers = extract_marker_dict(&obj.getattr(intern!(py, "marker"))?, builder, dsl_types)?;
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
    let uid =
        SectionUid(builder.register_uid(&obj.getattr(intern!(py, "uid"))?.extract::<String>()?));
    let local = obj
        .getattr(intern!(py, "local"))?
        .extract::<Option<bool>>()?;

    let handle = obj.getattr(intern!(py, "handle"))?;
    let user_register = obj.getattr(intern!(py, "user_register"))?;
    let sweep_parameter = obj.getattr(intern!(py, "sweep_parameter"))?;
    let prng_sample = obj.getattr(intern!(py, "prng_sample"))?;

    let match_target = if !handle.is_none() {
        let value = obj.getattr(intern!(py, "handle"))?.extract::<String>()?;
        MatchTarget::Handle(HandleUid(builder.register_uid(&value)))
    } else if !user_register.is_none() {
        let value = obj
            .getattr(intern!(py, "user_register"))?
            .extract::<u16>()?;
        MatchTarget::UserRegister(value)
    } else if !sweep_parameter.is_none() {
        let parameter_uid =
            builder.register_uid(&obj.getattr(intern!(py, "uid"))?.extract::<String>()?);
        MatchTarget::SweepParameter(ParameterUid(parameter_uid))
    } else if !prng_sample.is_none() {
        let parameter_uid =
            builder.register_uid(&obj.getattr(intern!(py, "uid"))?.extract::<String>()?);
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
    };
    Ok(out)
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
    let mut play_after = vec![];
    let play_after_py = obj.getattr(intern!(py, "play_after"))?;
    if !play_after_py.is_none() {
        // List of section UIDs or Section objects
        if obj.is_instance(&py.get_type::<PyList>())? {
            for section in obj.getattr(intern!(py, "play_after"))?.try_iter()? {
                let section = section?;
                if section.is_instance(&py.get_type::<PyString>())? {
                    let target_section = section.extract::<String>()?;
                    play_after.push(SectionUid(builder.register_uid(&target_section)))
                } else {
                    play_after.push(SectionUid(builder.register_uid(
                        &section.getattr(intern!(py, "uid"))?.extract::<String>()?,
                    )));
                }
            }
        } else {
            // A single section UID or Section object
            if play_after_py.is_instance(&py.get_type::<PyString>())? {
                let target_section = play_after_py.extract::<String>()?;
                play_after.push(SectionUid(builder.register_uid(&target_section)))
            } else {
                play_after.push(SectionUid(
                    builder.register_uid(
                        &play_after_py
                            .getattr(intern!(py, "uid"))?
                            .extract::<String>()?,
                    ),
                ));
            }
        }
    }
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

fn extract_acquire_loop_rt(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<AcquireLoopRt> {
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
    let obj = AcquireLoopRt {
        uid: SectionUid(builder.register_uid(&uid_py)),
        count,
        reset_oscillator_phase,
        acquisition_type,
        averaging_mode,
        repetition_mode,
    };
    Ok(obj)
}

fn extract_object(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
) -> Result<IrVariant> {
    let variant = if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::PlayPulse))
    {
        IrVariant::PlayPulse(extract_play_pulse(obj, builder, laboneq_types)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::Section))
    {
        IrVariant::Section(extract_section(obj, builder)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::Delay))
    {
        IrVariant::Delay(extract_delay(obj, builder, laboneq_types)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::Reserve))
    {
        IrVariant::Reserve(extract_reserve(obj, builder)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::Acquire))
    {
        IrVariant::Acquire(extract_acquire(obj, builder, laboneq_types)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::Match))
    {
        IrVariant::Match(extract_match(obj, builder)?)
    } else if obj.get_type().is(laboneq_types.laboneq_type(DslType::Case)) {
        IrVariant::Case(extract_case(obj, builder)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::AcquireLoopRt))
    {
        IrVariant::AcquireLoopRt(extract_acquire_loop_rt(obj, builder)?)
    } else if obj
        .get_type()
        .is(laboneq_types.laboneq_type(DslType::Sweep))
    {
        IrVariant::Sweep(extract_sweep(obj, builder, laboneq_types)?)
    } else {
        IrVariant::NotYetImplemented
    };
    Ok(variant)
}

/// Recursively traverse the experiment tree and create the nodes.
fn traverse_experiment(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
    laboneq_types: &DslTypes,
) -> Result<IrNode> {
    let variant = extract_object(obj, builder, laboneq_types)
        .with_context(|| format!("Error while handling object: '{obj}'"))?;
    let mut node = IrNode::new(variant);
    if let Some(children) = obj.getattr_opt(intern!(obj.py(), "children"))? {
        for child in children.try_iter()? {
            let child_value = traverse_experiment(&child?, builder, laboneq_types)?;
            node.add_child(child_value.into(), ());
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
pub fn build_experiment(experiment: &Bound<'_, PyAny>) -> Result<ExperimentBuilder> {
    let laboneq_types = DslTypes::new(experiment.py())?;
    let mut builder = ExperimentBuilder::new();
    let signals = experiment.getattr(intern!(experiment.py(), "signals"))?;
    for signal in signals.try_iter()? {
        let signal_uid = extract_experiment_signal(&signal?, &mut builder)?;
        builder.register_experiment_signal(signal_uid);
    }
    for section in experiment
        .getattr(intern!(experiment.py(), "sections"))?
        .try_iter()?
    {
        let root_section = traverse_experiment(&section?, &mut builder, &laboneq_types)?;
        let mut root = IrNode::new(IrVariant::Root);
        root.add_child(root_section.into(), ());
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
