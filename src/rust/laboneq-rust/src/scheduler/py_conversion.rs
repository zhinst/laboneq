// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python conversion layer for Scheduler data structures.
//!
//! The module converts Python DSL Experiment into Rust Scheduler IR.
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;

use crate::error::{Error, Result};
use crate::scheduler::pulse::PulseDef;
use crate::scheduler::pulse::PulseFunction;
use crate::scheduler::pulse::PulseFunctional;
use crate::scheduler::pulse::PulseKind;
use crate::scheduler::pulse::PulseSampled;
use crate::scheduler::py_object_interner::PyObjectInterner;
use anyhow::Context;
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::operation::{
    Acquire, AveragingLoop, Case, Chunking, Delay, Match, Operation, PlayPulse, PrngLoop,
    PrngSetup, PulseParameterValue, Reserve, ResetOscillatorPhase, Section, Sweep,
};
use laboneq_dsl::types::AcquisitionType;
use laboneq_dsl::types::AveragingMode;
use laboneq_dsl::types::ComplexOrFloat;
use laboneq_dsl::types::ExternalParameterUid;
use laboneq_dsl::types::HandleUid;
use laboneq_dsl::types::Marker;
use laboneq_dsl::types::MarkerSelector;
use laboneq_dsl::types::MatchTarget;
use laboneq_dsl::types::NumericLiteral;
use laboneq_dsl::types::ParameterUid;
use laboneq_dsl::types::PulseParameterUid;
use laboneq_dsl::types::PulseUid;
use laboneq_dsl::types::RepetitionMode;
use laboneq_dsl::types::SectionAlignment;
use laboneq_dsl::types::SectionUid;
use laboneq_dsl::types::SignalUid;
use laboneq_dsl::types::SweepParameter;
use laboneq_dsl::types::Trigger;
use laboneq_dsl::types::ValueOrParameter;
use laboneq_units::duration::seconds;
use num_complex::Complex64;
use numeric_array::NumericArray;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use pyo3::types::{PyDict, PyList, PyString};

pub(super) struct ExperimentBuilder<'py> {
    // List of root sections
    pub root: ExperimentNode,
    pub id_store: NamedIdStore,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub pulses: HashMap<PulseUid, PulseDef>,
    /// Signals that are defined in the experiment and can be referenced.
    /// Signals used in operations must be part of this set.
    pub available_signals: HashSet<SignalUid>,
    pub py_object_store: PyObjectInterner<ExternalParameterUid>,
    // Parameters that drive other parameters
    pub driving_parameters: HashMap<ParameterUid, HashSet<ParameterUid>>,
    pub dsl_types: DslTypes<'py>,
}

impl<'py> ExperimentBuilder<'py> {
    pub(super) fn new(py: Python<'py>) -> Self {
        Self {
            root: ExperimentNode::new(Operation::Root),
            id_store: NamedIdStore::new(),
            parameters: HashMap::new(),
            pulses: HashMap::new(),
            available_signals: HashSet::new(),
            py_object_store: PyObjectInterner::new(),
            dsl_types: DslTypes::new(py).unwrap(),
            driving_parameters: HashMap::new(),
        }
    }

    pub(crate) fn register_uid(&mut self, uid: &str) -> NamedId {
        self.id_store.get_or_insert(uid)
    }

    fn add_section(&mut self, root: ExperimentNode) {
        self.root.children.push(root.into());
    }

    /// Register a signal that is available for use.
    ///
    /// To match the current behavior, we register experiment signals
    /// even if they are not used.
    fn register_experiment_signal(&mut self, uid: SignalUid) {
        self.available_signals.insert(uid);
    }

    /// Register a signal.
    ///
    /// Returns an error if the signal is not listed as experiment signal.
    fn register_signal(&mut self, uid: &str) -> Result<SignalUid> {
        let signal_uid = self.register_uid(uid).into();
        if !self.available_signals.contains(&signal_uid) {
            let msg = format!(
                "Signal '{}' is not available in the experiment definition. Available signals are: '{}'.",
                uid,
                self.available_signals
                    .iter()
                    .map(|s| self.id_store.resolve(*s).unwrap())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return Err(Error::new(msg));
        }
        Ok(signal_uid)
    }

    fn register_pulse(&mut self, pulse: PulseDef) -> PulseUid {
        let uid = pulse.uid;
        self.pulses.entry(uid).or_insert(pulse);
        uid
    }

    fn register_external_parameter(
        &mut self,
        value: Bound<'_, PyAny>,
    ) -> Result<ExternalParameterUid> {
        Ok(ExternalParameterUid(
            self.py_object_store.get_or_intern(&value)?.0,
        ))
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

fn extract_numeric_value(obj: &Bound<'_, PyAny>) -> Result<NumericLiteral> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(NumericLiteral::Int(v));
    }
    // Extracting f64 on NumPy complex will drop the imaginary part, so check for complex first
    if is_complex(obj) {
        let c: Complex64 = obj.extract()?;
        return Ok(NumericLiteral::Complex(c));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(NumericLiteral::Float(v));
    }
    Err(Error::new(
        "Expected a numeric literal (int, float, or complex)",
    ))
}

fn is_complex(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyComplex>()
}

pub(super) fn extract_value_or_parameter<T: TryFrom<NumericLiteral>>(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<Option<ValueOrParameter<T>>>
where
    T::Error: Display,
{
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(value) = extract_numeric_value(obj) {
        let value = ValueOrParameter::<T>::Value(value.try_into().map_err(Error::new)?);
        return Ok(Some(value));
    }
    if obj.is_instance(builder.dsl_types.laboneq_type(DslType::Parameter))? {
        let parameter_uid = extract_parameter(obj, builder)?;
        return Ok(Some(ValueOrParameter::Parameter(parameter_uid)));
    }
    Err(Error::new(
        "Expected a numeric literal, string, bool or a parameter",
    ))
}

fn extract_section_alignment(obj: &Bound<'_, PyAny>) -> Result<SectionAlignment> {
    if obj.is_none() {
        return Ok(SectionAlignment::Left); // Default alignment
    }
    let py = obj.py();
    let out = match obj
        .getattr(intern!(py, "name"))?
        .cast_into::<PyString>()
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

/// Convert `Parameter` and its subclasses.
///
/// The 2 supported subclasses are `LinearSweepParameter` and `SweepParameter`.
/// When a parameter is registered, it is added to the builder's parameter map.
/// If the parameter already exists, its UID is returned directly.
///
/// This function also handles registering driving parameters for `SweepParameter`s and
/// updates the builder's `driving_parameters` map accordingly.
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
        let start = extract_numeric_value(&obj.getattr(intern!(py, "start"))?)?;
        let stop = extract_numeric_value(&obj.getattr(intern!(py, "stop"))?)?;
        let count = obj.getattr(intern!(py, "count"))?.extract::<usize>()?;
        let values = match start {
            NumericLiteral::Float(start) => {
                let stop: f64 = stop
                    .try_into()
                    .map_err(Error::new)
                    .with_context(|| "Linear sweep 'start' and 'stop' must be of same type")?;
                NumericArray::linspace(start, stop, count)
            }
            NumericLiteral::Int(start) => {
                let stop: f64 = stop
                    .try_into()
                    .map_err(Error::new)
                    .with_context(|| "Linear sweep 'start' and 'stop' must be of same type")?;
                NumericArray::linspace(start as f64, stop, count)
            }
            NumericLiteral::Complex(start) => {
                let stop: Complex64 = stop
                    .try_into()
                    .map_err(Error::new)
                    .with_context(|| "Linear sweep 'start' and 'stop' must be of same type")?;
                NumericArray::linspace_complex(start, stop, count)
            }
        };
        let parameter = SweepParameter::new(uid, values);
        builder.parameters.insert(parameter.uid, parameter);
        return Ok(uid);
    } else if obj.is_instance(sweep_parameter_py)? {
        let values = NumericArray::from_py(&obj.getattr(intern!(py, "values"))?)?;
        let parameter = SweepParameter {
            uid,
            values: values.into(),
        };
        builder.parameters.insert(parameter.uid, parameter);
        register_driving_parameters(uid, obj, builder)?;
        return Ok(uid);
    }
    Err(Error::new(format!(
        "Expected 'LinearSweepParameter' or 'SweepParameter', found: {obj}"
    )))
}

/// Traverse a `SweepParameter`'s `driven_by` recursively and register all driving parameters.
///
/// This allows to resolve derived parameters when registering sweeps, as they may not exist
/// in the sweep definition by default, but only in the field they are assigned to.
fn register_driving_parameters(
    uid: ParameterUid,
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<()> {
    let sweep_parameter_py = builder.dsl_types.laboneq_type(DslType::SweepParameter);
    if !obj.is_instance(sweep_parameter_py)? {
        return Ok(());
    }
    let py = obj.py();
    for driving_param in obj.getattr(intern!(py, "driven_by"))?.try_iter()? {
        let driving_param = driving_param?;
        let p_uid = ParameterUid(
            builder.register_uid(
                driving_param
                    .getattr(intern!(py, "uid"))?
                    .extract::<&str>()?,
            ),
        );
        builder
            .driving_parameters
            .entry(p_uid)
            .or_default()
            .insert(uid);
        register_driving_parameters(uid, &driving_param, builder)?;
    }
    Ok(())
}

/// Recursively collect all derived parameters driven by the given parameter.
fn collect_derived_parameters(
    parameter_uid: ParameterUid,
    driving_parameters: &HashMap<ParameterUid, HashSet<ParameterUid>>,
    collected: &mut HashSet<ParameterUid>,
) {
    if let Some(drivers) = driving_parameters.get(&parameter_uid) {
        for child in drivers {
            collected.insert(*child);
            collect_derived_parameters(*child, driving_parameters, collected);
        }
    }
}

fn extract_sweep(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Sweep> {
    let mut parameters = HashSet::new();
    for param in obj.getattr(intern!(obj.py(), "parameters"))?.try_iter()? {
        let param = param?;
        let parameter_uid = extract_parameter(&param, builder)?;
        parameters.insert(parameter_uid);
        // Resolve derived parameters that this parameter drives. The derived parameters may not
        // be explicitly listed in the sweep's parameter list, but need to be included.
        collect_derived_parameters(parameter_uid, &builder.driving_parameters, &mut parameters);
    }
    let uid_py_binding = obj.getattr(intern!(obj.py(), "uid"))?;
    let uid_py = uid_py_binding.extract::<&str>()?;
    if parameters.is_empty() {
        return Err(Error::new(format!(
            "Sweep '{uid_py}' must have at least one sweep parameter"
        )));
    }
    let alignment = extract_section_alignment(&obj.getattr(intern!(obj.py(), "alignment"))?)?;
    let reset_oscillator_phase = obj
        .getattr(intern!(obj.py(), "reset_oscillator_phase"))?
        .extract::<bool>()?;
    let count = builder
        .parameters
        .get(parameters.iter().next().unwrap())
        .expect("Internal error: Missing sweep parameter")
        .len();
    let chunk_count = extract_chunk_count(&obj.getattr(intern!(obj.py(), "chunk_count"))?)?;
    let auto_chunking = obj
        .getattr(intern!(obj.py(), "auto_chunking"))?
        .extract::<bool>()?;
    let chunking = if auto_chunking {
        Some(Chunking::Auto)
    } else if chunk_count > 1 {
        // Chunk count 1 is the same no chunking at all.
        Some(Chunking::Count { count: chunk_count })
    } else {
        None
    };
    let obj = Sweep {
        uid: SectionUid(builder.register_uid(uid_py)),
        parameters: parameters.into_iter().collect(),
        alignment,
        reset_oscillator_phase,
        count: count as u32,
        chunking,
    };
    Ok(obj)
}

fn extract_chunk_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    obj.extract::<usize>().map_err(|e| {
        if let Ok(v) = obj.extract::<i64>()
            && v < 1
        {
            let msg = format!("Chunk count must be >= 1, but {} was provided.", v);
            return Error::new(&msg).into();
        }
        e
    })
}

fn extract_delay(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Delay> {
    let py = obj.py();
    let signal = builder.register_signal(obj.getattr(intern!(py, "signal"))?.extract::<&str>()?)?;
    let time = extract_value_or_parameter::<f64>(&obj.getattr(intern!(py, "time"))?, builder)?
        .ok_or_else(|| Error::new("Delay time must be specified"))?;
    let precompensation_clear = obj
        .getattr(intern!(obj.py(), "precompensation_clear"))?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    let out = Delay {
        signal,
        time: match time {
            ValueOrParameter::Value(v) => ValueOrParameter::Value(v.into()),
            ValueOrParameter::Parameter(p) => ValueOrParameter::Parameter(p),
            _ => unreachable!(),
        },
        precompensation_clear,
    };
    Ok(out)
}

fn extract_reserve(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> PyResult<Reserve> {
    let py = obj.py();
    let signal = builder.register_signal(obj.getattr(intern!(py, "signal"))?.extract::<&str>()?)?;
    let out = Reserve { signal };
    Ok(out)
}

fn extract_pulse_parameters(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<HashMap<PulseParameterUid, PulseParameterValue>> {
    if obj.is_none() {
        return Ok(HashMap::new());
    }
    let obj = obj.cast::<PyDict>().map_err(PyErr::from)?;
    let mut out = HashMap::new();
    for (key, value) in obj.iter() {
        let key_str = key.extract::<&str>()?;
        let key = builder.id_store.get_or_insert(key_str);
        if let Ok(Some(value_or_param)) =
            extract_value_or_parameter::<NumericLiteral>(&value, builder)
        {
            out.insert(
                key.into(),
                PulseParameterValue::ValueOrParameter(value_or_param),
            );
        } else {
            out.insert(
                key.into(),
                PulseParameterValue::ExternalParameter(builder.register_external_parameter(value)?),
            );
        }
    }
    Ok(out)
}

/// Convert `Pulse` and its subclasses
fn extract_pulse(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<(PulseUid, HashMap<PulseParameterUid, PulseParameterValue>)> {
    let py = obj.py();
    let uid = PulseUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
    let pulse_parameters =
        if obj.is_instance(builder.dsl_types.laboneq_type(DslType::PulseFunctional))? {
            extract_pulse_parameters(&obj.getattr(intern!(py, "pulse_parameters"))?, builder)?
        } else {
            HashMap::new()
        };
    if builder.pulses.contains_key(&uid) {
        return Ok((uid, pulse_parameters));
    }
    let pulse = if obj.is_instance(builder.dsl_types.laboneq_type(DslType::PulseFunctional))? {
        PulseDef {
            uid,
            kind: PulseKind::Functional(PulseFunctional {
                length: seconds(obj.getattr(intern!(py, "length"))?.extract::<f64>()?),
                function: PulseFunction::Custom {
                    function: obj.getattr(intern!(py, "function"))?.extract::<String>()?,
                },
            }),
            amplitude: extract_amplitude(&obj.getattr(intern!(py, "amplitude"))?)?
                .unwrap_or(1.0.into()),
            can_compress: obj
                .getattr(intern!(py, "can_compress"))?
                .extract::<bool>()?,
        }
    } else {
        let samples_py_arr = obj.getattr(intern!(py, "samples"))?;
        let length = samples_py_arr.len()?;
        PulseDef {
            uid,
            kind: PulseKind::Sampled(PulseSampled {
                samples: samples_py_arr.into(),
                length,
            }),
            amplitude: 1.0.into(),
            can_compress: obj
                .getattr(intern!(py, "can_compress"))?
                .extract::<bool>()?,
        }
    };
    Ok((builder.register_pulse(pulse), pulse_parameters))
}

fn extract_amplitude(value: &Bound<'_, PyAny>) -> Result<Option<NumericLiteral>> {
    if value.is_none() {
        return Ok(None);
    }
    let amplitude = extract_numeric_value(value)?;
    Ok(Some(amplitude))
}

fn extract_acquire(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Acquire> {
    let py = obj.py();
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
    } else if !play_parameters.is_none() {
        let pulse_param = extract_pulse_parameters(&play_parameters, builder)?;
        parameters.push(pulse_param);
    } else {
        // Pad kernels to match the kernel count
        parameters.resize_with(kernels.len(), HashMap::new);
    }
    let out = Acquire {
        signal: builder.register_signal(obj.getattr(intern!(py, "signal"))?.extract::<&str>()?)?,
        handle: HandleUid(
            builder.register_uid(obj.getattr(intern!(py, "handle"))?.extract::<&str>()?),
        ),
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
    for (marker_selector_name, marker) in obj.cast::<PyDict>().map_err(PyErr::from)?.iter() {
        let marker_selector = match marker_selector_name.extract::<&str>()? {
            "marker1" => MarkerSelector::M1,
            "marker2" => MarkerSelector::M2,
            other => {
                return Err(Error::new(format!("Unknown marker selector: {other}")));
            }
        };
        let marker = marker.cast::<PyDict>().map_err(PyErr::from)?;
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
    let maybe_pulse = &obj.getattr(intern!(py, "pulse"))?;
    let pulse = if !maybe_pulse.is_none() {
        Some(extract_pulse(maybe_pulse, builder)?)
    } else {
        None
    };
    let py_amplitude = &obj.getattr(intern!(py, "amplitude"))?;
    let amplitude = extract_value_or_parameter::<ComplexOrFloat>(py_amplitude, builder)
        .with_context(|| {
            Error::new(format!(
                "Play operation amplitude must be a float or complex value: {}",
                py_amplitude
            ))
        })?;
    let phase = extract_value_or_parameter::<f64>(&obj.getattr(intern!(py, "phase"))?, builder)?;

    let increment_oscillator_phase = extract_value_or_parameter::<f64>(
        &obj.getattr(intern!(py, "increment_oscillator_phase"))?,
        builder,
    )?;
    let set_oscillator_phase = extract_value_or_parameter::<f64>(
        &obj.getattr(intern!(py, "set_oscillator_phase"))?,
        builder,
    )?;

    let length = extract_value_or_parameter::<f64>(&obj.getattr(intern!(py, "length"))?, builder)?;
    let parameters =
        extract_pulse_parameters(&obj.getattr(intern!(py, "pulse_parameters"))?, builder)?;
    let markers = extract_marker_dict(&obj.getattr(intern!(py, "marker"))?, builder)?;
    let pulse = PlayPulse {
        signal: builder.register_signal(obj.getattr(intern!(py, "signal"))?.extract::<&str>()?)?,
        pulse: pulse.as_ref().map(|x| x.0),
        amplitude: amplitude.unwrap_or(ValueOrParameter::Value(ComplexOrFloat::Float(1.0))),
        phase,
        increment_oscillator_phase,
        set_oscillator_phase,
        length: length.map(|length| match length {
            ValueOrParameter::Value(v) => ValueOrParameter::Value(v.into()),
            ValueOrParameter::Parameter(p) => ValueOrParameter::Parameter(p),
            _ => unreachable!(),
        }),
        parameters,
        pulse_parameters: pulse.map(|x| x.1).unwrap_or_default(),
        markers,
    };
    Ok(pulse)
}

/// Convert `Case`
fn extract_case(obj: &Bound<'_, PyAny>, builder: &mut ExperimentBuilder) -> Result<Case> {
    let py = obj.py();
    let uid = SectionUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
    let state = extract_numeric_value(&obj.getattr(intern!(py, "state"))?)
        .with_context(|| Error::new("Match case state must be numeric"))?;
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
        let parameter_uid = extract_parameter(&sweep_parameter, builder)?;
        MatchTarget::SweepParameter(parameter_uid)
    } else if !prng_sample.is_none() {
        let parameter_uid =
            builder.register_uid(prng_sample.getattr(intern!(py, "uid"))?.extract::<&str>()?);
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
    let uid = SectionUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
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
    let trigger_map = trigger_py.cast::<PyDict>().map_err(PyErr::from)?;
    let mut triggers = vec![];
    for (signal_uid, trigger) in trigger_map.iter() {
        let trigger_signal = signal_uid.extract::<&str>()?;
        let trigger_state = trigger.get_item("state")?.extract::<u8>()?;
        let trigger = Trigger {
            signal: builder.register_signal(trigger_signal)?,
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
        .cast_into::<PyString>()
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

/// Convert `RepetitionMode` enum.
///
/// Defaults to `FASTEST` if not specified.
fn extract_repetition_mode(
    obj: &Bound<'_, PyAny>,
    repetition_time: Option<f64>,
) -> Result<RepetitionMode> {
    if obj.is_none() {
        return Ok(RepetitionMode::Fastest); // Default repetition mode
    }
    let out = match obj.getattr(intern!(obj.py(), "name"))?.extract::<&str>()? {
        "FASTEST" => RepetitionMode::Fastest,
        "CONSTANT" => {
            if repetition_time.is_none() {
                return Err(Error::new(
                    "Repetition time must be set for CONSTANT repetition mode",
                ));
            }
            RepetitionMode::Constant {
                time: seconds(repetition_time.unwrap()),
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
    // Currently DSL supports `count` to be a floating point number, but it must be integral
    let count_py = obj.getattr(intern!(obj.py(), "count"))?;
    let count = if let Ok(count) = count_py.extract::<u32>() {
        count
    } else {
        let count = count_py.extract::<f64>()?;
        if count.fract() != 0.0 {
            return Err(Error::new("Sweep 'count' must be a positive integer"));
        }
        count as u32
    };
    let reset_oscillator_phase = obj
        .getattr(intern!(obj.py(), "reset_oscillator_phase"))?
        .extract::<bool>()?;
    let acquisition_type =
        extract_acquisition_type(&obj.getattr(intern!(obj.py(), "acquisition_type"))?)?;
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
        uid: builder
            .register_uid(obj.getattr(intern!(obj.py(), "uid"))?.extract::<&str>()?)
            .into(),
        count,
        reset_oscillator_phase,
        acquisition_type: acquisition_type.unwrap_or(AcquisitionType::Integration),
        averaging_mode,
        repetition_mode,
        alignment: SectionAlignment::Left, // Averaging loops are always left aligned by default
    };
    Ok(obj)
}

/// Convert `AcquisitionType` enum
pub(super) fn extract_acquisition_type(obj: &Bound<'_, PyAny>) -> Result<Option<AcquisitionType>> {
    if obj.is_none() {
        return Ok(None);
    }
    let enum_name = obj.getattr(intern!(obj.py(), "name"))?;
    let acq_type = match enum_name.extract::<&str>()? {
        "INTEGRATION" => Ok(AcquisitionType::Integration),
        "SPECTROSCOPY" => Ok(AcquisitionType::Spectroscopy),
        "SPECTROSCOPY_IQ" => Ok(AcquisitionType::Spectroscopy),
        "SPECTROSCOPY_PSD" => Ok(AcquisitionType::SpectroscopyPsd),
        "DISCRIMINATION" => Ok(AcquisitionType::Discrimination),
        "RAW" => Ok(AcquisitionType::Raw),
        _ => Err(Error::new(format!("Unknown acquisition type: {enum_name}"))),
    }?;
    Ok(Some(acq_type))
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
    let sample_uid = builder.register_uid(
        prng_sample_py
            .getattr(intern!(py, "uid"))?
            .extract::<&str>()?,
    );
    let count = prng_sample_py
        .getattr(intern!(py, "count"))?
        .extract::<u32>()?;
    let obj = PrngLoop {
        uid: uid.into(),
        count,
        sample_uid: sample_uid.into(),
    };
    Ok(obj)
}

fn extract_reset_oscillator_phase(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<ResetOscillatorPhase> {
    let py = obj.py();
    let py_signal = obj.getattr(intern!(py, "signal"))?;
    let signal_uid = py_signal.extract::<Option<&str>>()?;
    let signals = signal_uid
        .map(|uid| builder.register_signal(uid))
        .transpose()?;
    let obj = ResetOscillatorPhase {
        signals: signals.into_iter().collect(),
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
    // Post-order traversal to build children first to collect e.g. sweep parameters
    // before building the parent node.
    let mut children_nodes = vec![];
    if let Some(children) = obj.getattr_opt(intern!(obj.py(), "children"))? {
        children_nodes.reserve(children.len()?);
        for child in children.try_iter()? {
            let child_value = traverse_experiment(&child?, builder)?;
            children_nodes.push(child_value.into());
        }
    }
    let variant = extract_object(obj, builder)
        .with_context(|| format!("Error while handling object: '{obj}'"))?;
    let mut node = ExperimentNode::new(variant);
    node.children = children_nodes;
    Ok(node)
}

/// Convert `ExperimentSignal`
fn extract_experiment_signal(
    obj: &Bound<'_, PyAny>,
    builder: &mut ExperimentBuilder,
) -> Result<SignalUid> {
    let py = obj.py();
    let signal =
        SignalUid(builder.register_uid(obj.getattr(intern!(py, "uid"))?.extract::<&str>()?));
    Ok(signal)
}

/// Convert `Experiment` into Rust Scheduler IR.
pub(super) fn build_experiment<'py>(
    experiment: &Bound<'py, PyAny>,
    builder: &mut ExperimentBuilder<'py>,
) -> Result<()> {
    let signals = experiment.getattr(intern!(experiment.py(), "signals"))?;
    for signal in signals.try_iter()? {
        let signal_uid = extract_experiment_signal(&signal?, builder)?;
        builder.register_experiment_signal(signal_uid);
    }
    for section in experiment
        .getattr(intern!(experiment.py(), "sections"))?
        .try_iter()?
    {
        let root_section = traverse_experiment(&section?, builder)?;
        builder.add_section(root_section);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::ffi::c_str;

    #[test]
    fn test_numeric_literal_python_rust_conversion() {
        Python::attach(|py| {
            let code = c_str!(
                r#"
import numpy as np

python_int = 42
python_float = 3.16
python_complex = 1 + 2j
numpy_int = np.int64(42)
numpy_float = np.float64(3.16)
numpy_complex = np.complex128(1 + 2j)
foobar_string = "foobar"
"#
            );
            let test_module: Py<PyAny> = PyModule::from_code(py, code, c_str!(""), c_str!(""))
                .unwrap()
                .into();
            let globals = test_module.bind(py);

            let python_int = globals.getattr("python_int").unwrap();
            let value = extract_numeric_value(&python_int).unwrap();
            assert_eq!(value, NumericLiteral::Int(42));

            let python_float = globals.getattr("python_float").unwrap();
            let value = extract_numeric_value(&python_float).unwrap();
            assert_eq!(value, NumericLiteral::Float(3.16));

            let python_complex = globals.getattr("python_complex").unwrap();
            let value = extract_numeric_value(&python_complex).unwrap();
            assert_eq!(value, NumericLiteral::Complex(Complex64::new(1.0, 2.0)));

            let numpy_int = globals.getattr("numpy_int").unwrap();
            let value = extract_numeric_value(&numpy_int).unwrap();
            assert_eq!(value, NumericLiteral::Int(42));

            let numpy_float = globals.getattr("numpy_float").unwrap();
            let value = extract_numeric_value(&numpy_float).unwrap();
            assert_eq!(value, NumericLiteral::Float(3.16));

            let numpy_complex = globals.getattr("numpy_complex").unwrap();
            let value = extract_numeric_value(&numpy_complex).unwrap();
            assert_eq!(value, NumericLiteral::Complex(Complex64::new(1.0, 2.0)));

            let foobar_string = globals.getattr("foobar_string").unwrap();
            let result = extract_numeric_value(&foobar_string);
            assert!(result.is_err());
        });
    }
}
