// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::scheduler::py_export::complex_or_float_to_py;
use crate::scheduler::py_pulse_defs::pulse_def_to_py;
use crate::scheduler::{experiment::Experiment, py_pulse_parameters::pulse_parameters_to_py_dict};
use crate::{error::Result, scheduler::py_object_interner::PyObjectInterner};
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_dsl::types::{
    AcquisitionType, AveragingMode, ComplexOrFloat, ExternalParameterUid, Marker, MarkerSelector,
    ParameterUid, PulseUid, SectionUid, SignalUid, SweepParameter, ValueOrParameter,
};
use laboneq_scheduler::ScheduledNode;
use laboneq_scheduler::ir::{
    Acquire, Case, ChangeOscillatorPhase, InitialLocalOscillatorFrequency,
    InitialOscillatorFrequency, InitialVoltageOffset, IrKind, Loop, LoopKind, Match, MatchTarget,
    PlayPulse, PpcStep, PrngSetup, Section, SetOscillatorFrequency, Trigger,
};
use pyo3::{
    intern,
    prelude::*,
    types::{PyDict, PyList, PySet, PyString, PyTuple},
};
use std::collections::HashMap;

/// Type representing `laboneq.compiler.ir.IntervalIR`
pub(crate) type IntervalSchedule = Py<PyAny>;

/// Converts a [`ScheduledNode`] into a [`PyScheduleCompat`], a container for Python schedule objects.
///
/// This is a temporary solution to facilitate the transition from Python-based scheduling to Rust-based scheduling.
/// It allows us to return Python schedule objects from Rust functions, enabling gradual migration of scheduling logic
/// to Rust without breaking existing functionality.
pub(super) fn generate_py_schedules<'py>(
    py: Python<'py>,
    scheduled_node: &ScheduledNode,
    experiment: &Experiment,
    acquisition_type: &AcquisitionType,
) -> PyResult<IntervalSchedule> {
    let mut context: Context<'_, 'py> = Context {
        id_store: &experiment.id_store,
        py_objects: &experiment.py_object_store,
        acquisition_type_py: acquisition_type_to_py(py, acquisition_type)?,
        py_string_store: HashMap::new(),
        py_sweep_parameters: HashMap::new(),
        py_pulse_defs: HashMap::new(),
    };
    for param in experiment.parameters.values() {
        context.create_py_sweep_parameter(py, param)?;
    }
    for pulse_def in experiment.pulses.values() {
        let py_pulse_def = pulse_def_to_py(py, &experiment.id_store, pulse_def)?;
        context
            .py_pulse_defs
            .insert(pulse_def.uid, py_pulse_def.bind(py).clone());
    }
    let root = generate_py_schedules_impl(py, scheduled_node, &mut context, None)?.unwrap();
    Ok(root.into())
}

struct Context<'a, 'py> {
    id_store: &'a NamedIdStore,
    py_objects: &'a PyObjectInterner<ExternalParameterUid>,
    acquisition_type_py: Bound<'py, PyAny>,
    // Cache for Python string objects corresponding to NamedId keys
    py_string_store: HashMap<NamedId, Bound<'py, PyString>>,
    py_sweep_parameters: HashMap<ParameterUid, Bound<'py, PyAny>>,
    py_pulse_defs: HashMap<PulseUid, Bound<'py, PyAny>>,
}

impl<'py> Context<'_, 'py> {
    // Get or create a Python string object for the given UID.
    fn py_string(
        &mut self,
        py: Python<'py>,
        uid: impl Into<NamedId> + Copy,
    ) -> PyResult<Bound<'py, PyString>> {
        if let Some(py_str) = self.py_string_store.get(&uid.into()) {
            Ok(py_str.clone())
        } else {
            let signal_name = self.id_store.resolve(uid.into()).unwrap().to_string();
            let py_str = PyString::new(py, &signal_name);
            self.py_string_store.insert(uid.into(), py_str.clone());
            Ok(py_str)
        }
    }

    fn create_py_sweep_parameter(
        &mut self,
        py: Python<'py>,
        parameter: &SweepParameter,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(py_param) = self.py_sweep_parameters.get(&parameter.uid) {
            Ok(py_param.clone())
        } else {
            let py_param = sweep_parameter_to_py(py, parameter, self)?;
            self.py_sweep_parameters
                .insert(parameter.uid, py_param.clone());
            Ok(py_param)
        }
    }

    fn get_py_sweep_parameter(&self, parameter_uid: ParameterUid) -> Bound<'py, PyAny> {
        self.py_sweep_parameters
            .get(&parameter_uid)
            .unwrap()
            .clone()
    }

    fn get_py_pulse_def(&self, pulse_def_uid: PulseUid) -> Bound<'py, PyAny> {
        self.py_pulse_defs.get(&pulse_def_uid).unwrap().clone()
    }
}

fn generate_py_schedules_impl<'ctx, 'py>(
    py: Python<'py>,
    scheduled_node: &'ctx ScheduledNode,
    context: &mut Context<'ctx, 'py>,
    parent_section_uid: Option<&SectionUid>,
) -> Result<Option<Bound<'py, PyAny>>> {
    match &scheduled_node.kind {
        IrKind::InitialOscillatorFrequency(obj) => Ok(Some(handle_initial_oscillator_frequency(
            py,
            context,
            scheduled_node,
            obj,
        )?)),
        IrKind::InitialLocalOscillatorFrequency(obj) => Ok(Some(
            handle_initial_local_oscillator_frequency(py, context, scheduled_node, obj)?,
        )),
        IrKind::InitialVoltageOffset(obj) => Ok(Some(handle_initial_voltage_offset(
            py,
            context,
            scheduled_node,
            obj,
        )?)),
        IrKind::Loop(obj) => {
            let schedule = handle_loop_schedule(py, context, scheduled_node, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::LoopIteration => {
            let schedule =
                handle_loop_iteration(py, context, scheduled_node, *parent_section_uid.unwrap())?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, parent_section_uid)?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::LoopIterationPreamble => {
            let schedule = handle_loop_iteration_preamble(py, context, scheduled_node)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, parent_section_uid)?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::SetOscillatorFrequency(obj) => {
            let schedule = handle_set_oscillator_frequency(py, context, scheduled_node, obj)?;
            Ok(Some(schedule))
        }
        IrKind::ResetOscillatorPhase { signals } => {
            let schedule = handle_phase_reset(py, context, scheduled_node, signals)?;
            Ok(Some(schedule))
        }
        IrKind::PpcStep(obj) => {
            let schedule = handle_ppc_steps(py, context, scheduled_node, obj)?;
            Ok(Some(schedule))
        }
        IrKind::Section(obj) => {
            let schedule = handle_section(py, context, scheduled_node, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::Acquire(obj) => {
            let schedule = handle_acquisition(py, context, scheduled_node, obj)?;
            Ok(Some(schedule))
        }
        IrKind::Delay { signal } => {
            let schedule = handle_delay(py, context, scheduled_node, *signal)?;
            Ok(Some(schedule))
        }
        IrKind::ClearPrecompensation { signal } => {
            let schedule = handle_reset_precompensation(py, context, scheduled_node, *signal)?;
            Ok(Some(schedule))
        }
        IrKind::PlayPulse(obj) => {
            let schedule = handle_play_pulse(py, context, scheduled_node, obj)?;
            Ok(Some(schedule))
        }
        IrKind::ChangeOscillatorPhase(obj) => {
            let schedule = handle_change_oscillator_phase(py, context, scheduled_node, obj)?;
            Ok(Some(schedule))
        }
        IrKind::Match(obj) => {
            let schedule = handle_match(py, context, scheduled_node, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::Case(obj) => {
            let schedule = handle_case(py, context, scheduled_node, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::Root => {
            let schedule = handle_root_schedule(py, context, scheduled_node)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, None)?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                    schedule
                        .getattr(intern!(py, "children_start"))?
                        .call_method1(intern!(py, "append"), (child.offset().value(),))?;
                }
            }
            Ok(Some(schedule))
        }
    }
}

fn handle_root_schedule<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "RootScheduleIR"))?;
    let signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "acquisition_type"), &ctx.acquisition_type_py)?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_loop_iteration_preamble<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "LoopIterationPreambleIR"))?;
    let signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_loop_iteration<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    parent_section: SectionUid,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "LoopIterationIR"))?;
    let signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "section"), ctx.py_string(py, parent_section)?)?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_initial_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &InitialOscillatorFrequency,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "InitialOscillatorFrequencyIR"))?;
    let mut signals = Vec::new();
    let out: Vec<(Bound<'_, PyString>, f64)> = obj
        .values
        .iter()
        .map(|(signal, freq)| {
            let sig_py = ctx.py_string(py, *signal).unwrap();
            signals.push(sig_py.clone());
            (sig_py, TryInto::<f64>::try_into(*freq).unwrap())
        })
        .collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "values"), out)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_initial_local_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &InitialLocalOscillatorFrequency,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir.oscillator_ir"))?;
    let py_obj = m.getattr(intern!(py, "InitialLocalOscillatorFrequencyIR"))?;
    let value: f64 = obj.value.try_into().unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "value"), value)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_initial_voltage_offset<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &InitialVoltageOffset,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir.voltage_offset"))?;
    let py_obj = m.getattr(intern!(py, "InitialOffsetVoltageIR"))?;

    let value: f64 = obj.value.try_into().unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "value"), value)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_set_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &SetOscillatorFrequency,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "SetOscillatorFrequencyIR"))?;

    let mut signals = Vec::new();
    let out: Vec<(Bound<'_, PyString>, f64)> = obj
        .values
        .iter()
        .map(|(signal, freq)| {
            let sig_py = ctx.py_string(py, *signal).unwrap();
            signals.push(sig_py.clone());
            (sig_py, TryInto::<f64>::try_into(*freq).unwrap())
        })
        .collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "values"), out)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_phase_reset<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    signals: &[SignalUid],
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "PhaseResetIR"))?;

    let signals = signals
        .iter()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_ppc_steps<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &PpcStep,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "PPCStepIR"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(
        intern!(py, "trigger_duration"),
        obj.trigger_duration.value(),
    )?;
    kwargs.set_item(
        intern!(py, "ppc_device"),
        ctx.py_string(py, obj.device).unwrap(),
    )?;
    kwargs.set_item(intern!(py, "ppc_channel"), obj.channel)?;

    if let Some(pump_power) = obj.pump_power {
        let f: f64 = pump_power.try_into().unwrap();
        kwargs.set_item(intern!(py, "pump_power"), f)?;
    }
    if let Some(pump_freq) = obj.pump_frequency {
        let f: f64 = pump_freq.try_into().unwrap();
        kwargs.set_item(intern!(py, "pump_frequency"), f)?;
    }
    if let Some(probe_power) = obj.probe_power {
        let f: f64 = probe_power.try_into().unwrap();
        kwargs.set_item(intern!(py, "probe_power"), f)?;
    }
    if let Some(probe_frequency) = obj.probe_frequency {
        let f: f64 = probe_frequency.try_into().unwrap();
        kwargs.set_item(intern!(py, "probe_frequency"), f)?;
    }
    if let Some(cancellation_phase) = obj.cancellation_phase {
        let f: f64 = cancellation_phase.try_into().unwrap();
        kwargs.set_item(intern!(py, "cancellation_phase"), f)?;
    }
    if let Some(cancellation_attenuation) = obj.cancellation_attenuation {
        let f: f64 = cancellation_attenuation.try_into().unwrap();
        kwargs.set_item(intern!(py, "cancellation_attenuation"), f)?;
    }

    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_section<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &Section,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_schedule = m.getattr(intern!(py, "SectionIR"))?;

    let py_signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    let uid = ctx.py_string(py, obj.uid)?;
    let py_triggers = obj
        .triggers
        .iter()
        .map(|trigger| trigger_to_py(py, trigger.clone(), ctx))
        .collect::<Result<Vec<Bound<'_, PyTuple>>>>()?;
    kwargs.set_item(intern!(py, "section"), uid)?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "trigger_output"), PySet::new(py, py_triggers)?)?;
    kwargs.set_item(
        intern!(py, "prng_setup"),
        obj.prng_setup
            .as_ref()
            .map(|o| create_py_prng_info(py, o))
            .transpose()?,
    )?;
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn create_py_prng_info<'py>(py: Python<'py>, prng: &PrngSetup) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.data.compilation_job"))?;
    let py_prng_info = m.getattr(intern!(py, "PRNGInfo"))?;

    let prng_init_kwargs = PyDict::new(py);
    prng_init_kwargs.set_item(intern!(py, "range"), prng.range as i64)?;
    prng_init_kwargs.set_item(intern!(py, "seed"), prng.seed as i64)?;
    Ok(py_prng_info.call((), Some(&prng_init_kwargs))?)
}

fn trigger_to_py<'py>(
    py: Python<'py>,
    trigger: Trigger,
    ctx: &mut Context<'_, 'py>,
) -> Result<Bound<'py, PyTuple>> {
    let signal = ctx.py_string(py, trigger.signal).unwrap();
    let state = trigger.state as i64;
    let out = (signal, state);
    let tuple = out.into_pyobject(py)?;
    Ok(tuple)
}

fn handle_acquisition<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &Acquire,
) -> Result<Bound<'py, PyAny>> {
    if obj.kernels.len() > 1 {
        return handle_acquire_group_schedule(py, ctx, node, obj);
    }
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "PulseIR"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(
        intern!(py, "integration_length"),
        obj.integration_length.value(),
    )?;
    if let Some(pulse_def) = obj.kernels.first() {
        let pulse = ctx.get_py_pulse_def(*pulse_def);
        kwargs.set_item(intern!(py, "pulse"), pulse)?;
    } else {
        kwargs.set_item(intern!(py, "pulse"), py.None())?;
    };
    kwargs.set_item(
        intern!(py, "handle"),
        ctx.py_string(py, obj.handle).unwrap(),
    )?;
    kwargs.set_item(intern!(py, "is_acquire"), true)?;
    kwargs.set_item(intern!(py, "phase"), 0.0)?;
    kwargs.set_item(intern!(py, "amplitude"), 1.0)?;
    kwargs.set_item(intern!(py, "markers"), PyList::empty(py))?;
    let pulse_params_py = obj.parameters.first().map(|parameters| {
        pulse_parameters_to_py_dict(py, parameters, ctx.id_store, ctx.py_objects)
    });
    kwargs.set_item(
        intern!(py, "play_pulse_params"),
        pulse_params_py.transpose()?,
    )?;

    let pulse_pulse_params_py = obj.pulse_parameters.first().map(|parameters| {
        pulse_parameters_to_py_dict(py, parameters, ctx.id_store, ctx.py_objects)
    });
    kwargs.set_item(
        intern!(py, "pulse_pulse_params"),
        pulse_pulse_params_py.transpose()?,
    )?;
    kwargs.set_item(intern!(py, "acquisition_type"), &ctx.acquisition_type_py)?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_acquire_group_schedule<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &Acquire,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "AcquireGroupIR"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(
        intern!(py, "handle"),
        ctx.py_string(py, obj.handle).unwrap(),
    )?;
    let pulses = obj
        .kernels
        .iter()
        .map(|pulse_def| ctx.get_py_pulse_def(*pulse_def))
        .collect::<Vec<_>>();
    kwargs.set_item(intern!(py, "pulses"), &pulses)?;
    kwargs.set_item(intern!(py, "amplitudes"), [1.0].repeat(pulses.len()))?;
    let pulse_params_py = obj.parameters.iter().map(|parameters| {
        pulse_parameters_to_py_dict(py, parameters, ctx.id_store, ctx.py_objects)
    });
    kwargs.set_item(
        intern!(py, "play_pulse_params"),
        pulse_params_py.collect::<Result<Vec<_>>>()?,
    )?;

    let pulse_pulse_params_py = obj.pulse_parameters.iter().map(|parameters| {
        pulse_parameters_to_py_dict(py, parameters, ctx.id_store, ctx.py_objects)
    });
    kwargs.set_item(
        intern!(py, "pulse_pulse_params"),
        pulse_pulse_params_py.collect::<Result<Vec<_>>>()?,
    )?;
    kwargs.set_item(intern!(py, "acquisition_type"), &ctx.acquisition_type_py)?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_delay<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    signal: SignalUid,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_pulse_schedule = m.getattr(intern!(py, "PulseIR"))?;
    let signal_string = ctx.py_string(py, signal).unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, [&signal_string])?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "integration_length"), py.None())?;
    kwargs.set_item(intern!(py, "pulse"), py.None())?;
    kwargs.set_item(intern!(py, "is_acquire"), false)?;
    kwargs.set_item(intern!(py, "phase"), 0.0)?;
    kwargs.set_item(intern!(py, "amplitude"), 1.0)?;
    let py_schedule = py_pulse_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_reset_precompensation<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    signal: SignalUid,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_clear_precompensation = m.getattr(intern!(py, "PrecompClearIR"))?;

    let signal_string = ctx.py_string(py, signal).unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, [&signal_string])?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule = py_clear_precompensation.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_play_pulse<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &PlayPulse,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "PulseIR"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "integration_length"), py.None())?;
    kwargs.set_item(intern!(py, "pulse"), ctx.get_py_pulse_def(obj.pulse))?;
    kwargs.set_item(intern!(py, "handle"), py.None())?;
    kwargs.set_item(intern!(py, "is_acquire"), false)?;
    if let Some(phase) = &obj.phase {
        let (phase, _) = value_or_parameter_to_py_f64(py, phase, ctx)?;
        kwargs.set_item(intern!(py, "phase"), phase)?;
    } else {
        kwargs.set_item(intern!(py, "phase"), 0.0)?;
    }
    if let Some(phase) = &obj.increment_oscillator_phase {
        let (phase, phase_param) = value_or_parameter_to_py_f64(py, phase, ctx)?;
        kwargs.set_item(intern!(py, "increment_oscillator_phase"), phase)?;
        kwargs.set_item(intern!(py, "incr_phase_param_name"), phase_param)?;
    }
    if let Some(phase) = &obj.set_oscillator_phase {
        let (phase, _) = value_or_parameter_to_py_f64(py, phase, ctx)?;
        kwargs.set_item(intern!(py, "set_oscillator_phase"), phase)?;
    }
    let (amplitude, amp_param) =
        complex_or_float_or_parameter_to_py_complex64(py, &obj.amplitude, ctx)?;
    kwargs.set_item(intern!(py, "amplitude"), amplitude)?;
    kwargs.set_item(intern!(py, "amp_param_name"), amp_param)?;
    kwargs.set_item(
        intern!(py, "markers"),
        obj.markers
            .iter()
            .map(|marker| marker_to_py(py, marker, ctx))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let pulse_params_py =
        pulse_parameters_to_py_dict(py, &obj.parameters, ctx.id_store, ctx.py_objects)?;
    kwargs.set_item(intern!(py, "play_pulse_params"), pulse_params_py)?;

    let pulse_pulse_params_py =
        pulse_parameters_to_py_dict(py, &obj.pulse_parameters, ctx.id_store, ctx.py_objects)?;
    kwargs.set_item(intern!(py, "pulse_pulse_params"), pulse_pulse_params_py)?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_change_oscillator_phase<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &ChangeOscillatorPhase,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_obj = m.getattr(intern!(py, "PulseIR"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "integration_length"), py.None())?;
    kwargs.set_item(intern!(py, "pulse"), py.None())?;
    kwargs.set_item(intern!(py, "handle"), py.None())?;
    kwargs.set_item(intern!(py, "is_acquire"), false)?;
    kwargs.set_item(intern!(py, "phase"), py.None())?;
    if let Some(phase) = &obj.increment {
        let (phase, phase_param) = value_or_parameter_to_py_f64(py, phase, ctx)?;
        kwargs.set_item(intern!(py, "increment_oscillator_phase"), phase)?;
        kwargs.set_item(intern!(py, "incr_phase_param_name"), phase_param)?;
    }
    if let Some(phase) = &obj.set {
        let (phase, _) = value_or_parameter_to_py_f64(py, phase, ctx)?;
        kwargs.set_item(intern!(py, "set_oscillator_phase"), phase)?;
    }
    kwargs.set_item(intern!(py, "amplitude"), py.None())?;
    kwargs.set_item(intern!(py, "amp_param_name"), py.None())?;
    kwargs.set_item(intern!(py, "markers"), py.None())?;
    kwargs.set_item(intern!(py, "play_pulse_params"), py.None())?;
    kwargs.set_item(intern!(py, "pulse_pulse_params"), py.None())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn value_or_parameter_to_py_f64<'py>(
    py: Python<'py>,
    value: &ValueOrParameter<f64>,
    ctx: &mut Context<'_, 'py>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match value {
        ValueOrParameter::Value(value) => Ok((value.into_pyobject(py)?.into(), py.None())),
        ValueOrParameter::ResolvedParameter { value, uid } => {
            let py_param = ctx.py_string(py, *uid).unwrap();
            Ok((
                value.into_pyobject(py)?.into(),
                py_param.into_pyobject(py)?.into(),
            ))
        }
        _ => panic!("Expected value or resolved parameter"),
    }
}

fn complex_or_float_or_parameter_to_py_complex64<'py>(
    py: Python<'py>,
    value: &ValueOrParameter<ComplexOrFloat>,
    ctx: &mut Context<'_, 'py>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match value {
        ValueOrParameter::Value(value) => Ok((complex_or_float_to_py(py, value)?, py.None())),
        ValueOrParameter::ResolvedParameter { value, uid } => {
            let py_param = ctx.py_string(py, *uid).unwrap();
            Ok((
                complex_or_float_to_py(py, value)?,
                py_param.into_pyobject(py)?.into(),
            ))
        }
        _ => panic!("Expected value or resolved parameter"),
    }
}

fn marker_to_py<'py>(
    py: Python<'py>,
    marker: &Marker,
    ctx: &mut Context<'_, 'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.data.compilation_job"))?;
    let marker_py = m.getattr(intern!(py, "Marker"))?;
    let d = PyDict::new(py);
    let marker_selector = match marker.marker_selector {
        MarkerSelector::M1 => intern!(py, "marker1"),
        MarkerSelector::M2 => intern!(py, "marker2"),
    };
    d.set_item(intern!(py, "marker_selector"), marker_selector)?;
    d.set_item(intern!(py, "enable"), marker.enable)?;
    if let Some(start) = marker.start {
        d.set_item(intern!(py, "start"), start.value())?;
    } else {
        d.set_item(intern!(py, "start"), py.None())?;
    }
    if let Some(length) = marker.length {
        d.set_item(intern!(py, "length"), length.value())?;
    } else {
        d.set_item(intern!(py, "length"), py.None())?;
    }
    if let Some(pulse_uid) = marker.pulse_id {
        let pulse = ctx.py_string(py, pulse_uid).unwrap();
        d.set_item(intern!(py, "pulse_id"), pulse)?;
    } else {
        d.set_item(intern!(py, "pulse_id"), py.None())?;
    }
    marker_py.call((), Some(&d))
}

fn handle_loop_schedule<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &Loop,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_schedule = m.getattr(intern!(py, "LoopIR"))?;

    let py_signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    let uid = ctx.py_string(py, obj.uid)?;
    kwargs.set_item(intern!(py, "section"), uid)?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "iterations"), obj.iterations)?;
    kwargs.set_item(intern!(py, "compressed"), obj.compressed())?;
    kwargs.set_item(intern!(py, "averaging_mode"), py.None())?;
    match &obj.kind {
        LoopKind::Averaging { mode } => {
            kwargs.set_item(
                intern!(py, "averaging_mode"),
                averaging_mode_to_py(py, mode).unwrap(),
            )?;
        }
        LoopKind::Sweeping { parameters } => {
            let sweep_parameters = parameters
                .iter()
                .map(|param_uid| ctx.get_py_sweep_parameter(*param_uid))
                .collect::<Vec<_>>();
            kwargs.set_item(intern!(py, "sweep_parameters"), sweep_parameters)?;
        }
        LoopKind::Prng { sample_uid } => {
            let uid_py = ctx.py_string(py, *sample_uid)?;
            kwargs.set_item(intern!(py, "prng_sample"), uid_py)?;
        }
    }
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_match<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &Match,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_schedule = m.getattr(intern!(py, "MatchIR"))?;

    let py_signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    let uid = ctx.py_string(py, obj.uid)?;
    kwargs.set_item(intern!(py, "section"), uid)?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    kwargs.set_item(intern!(py, "trigger_output"), PySet::empty(py)?)?;
    match obj.target {
        MatchTarget::Handle(handle) => {
            let handle_py = ctx.py_string(py, handle).unwrap();
            kwargs.set_item(intern!(py, "handle"), handle_py)?;
            kwargs.set_item(intern!(py, "user_register"), py.None())?;
            kwargs.set_item(intern!(py, "prng_sample"), py.None())?;
        }
        MatchTarget::UserRegister(user_reg) => {
            kwargs.set_item(intern!(py, "user_register"), user_reg as i64)?;
            kwargs.set_item(intern!(py, "handle"), py.None())?;
            kwargs.set_item(intern!(py, "prng_sample"), py.None())?;
        }
        MatchTarget::PrngSample(sample_uid) => {
            kwargs.set_item(
                intern!(py, "prng_sample"),
                ctx.py_string(py, sample_uid).unwrap(),
            )?;
            kwargs.set_item(intern!(py, "handle"), py.None())?;
            kwargs.set_item(intern!(py, "user_register"), py.None())?;
        }
        MatchTarget::SweepParameter(_) => {
            panic!("Expected MatchTarget::SweepParameter to be resolved");
        }
    }
    kwargs.set_item(intern!(py, "local"), obj.local)?;
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn averaging_mode_to_py<'py>(
    py: Python<'py>,
    averaging_mode: &AveragingMode,
) -> PyResult<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.core.types.enums.averaging_mode"))?;
    let py_obj = m.getattr(intern!(py, "AveragingMode"))?;
    match averaging_mode {
        AveragingMode::Cyclic => py_obj.getattr(intern!(py, "CYCLIC")),
        AveragingMode::Sequential => py_obj.getattr(intern!(py, "SEQUENTIAL")),
        AveragingMode::SingleShot => py_obj.getattr(intern!(py, "SINGLE_SHOT")),
    }
}

fn handle_case<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    node: &ScheduledNode,
    obj: &Case,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.ir"))?;
    let py_schedule = m.getattr(intern!(py, "CaseIR"))?;

    let py_signals = node
        .signals()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "section"), ctx.py_string(py, obj.uid)?)?;
    kwargs.set_item(intern!(py, "state"), obj.state)?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(intern!(py, "length"), node.length().value())?;
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn sweep_parameter_to_py<'py>(
    py: Python<'py>,
    parameter: &SweepParameter,
    ctx: &mut Context<'_, 'py>,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.data.compilation_job"))?;
    let py_obj = m.getattr(intern!(py, "ParameterInfo"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "uid"), ctx.py_string(py, parameter.uid)?)?;
    kwargs.set_item(intern!(py, "values"), parameter.values.to_py(py)?)?;
    let py_parameter = py_obj.call((), Some(&kwargs))?;
    Ok(py_parameter)
}

fn acquisition_type_to_py<'py>(
    py: Python<'py>,
    acquisition_type: &AcquisitionType,
) -> PyResult<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.core.types.enums.acquisition_type"))?;
    let py_obj = m.getattr(intern!(py, "AcquisitionType"))?;
    match acquisition_type {
        AcquisitionType::Discrimination => py_obj.getattr(intern!(py, "DISCRIMINATION")),
        AcquisitionType::Integration => py_obj.getattr(intern!(py, "INTEGRATION")),
        AcquisitionType::Spectroscopy => py_obj.getattr(intern!(py, "SPECTROSCOPY")),
        AcquisitionType::SpectroscopyIq => py_obj.getattr(intern!(py, "SPECTROSCOPY_IQ")),
        AcquisitionType::SpectroscopyPsd => py_obj.getattr(intern!(py, "SPECTROSCOPY_PSD")),
        AcquisitionType::Raw => py_obj.getattr(intern!(py, "RAW")),
    }
}
