// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::scheduler::py_export::complex_or_float_to_py;
use crate::scheduler::{experiment::Experiment, py_pulse_parameters::pulse_parameters_to_py_dict};
use crate::{error::Result, scheduler::py_object_interner::PyObjectInterner};
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_scheduler::experiment::types::{AveragingMode, ComplexOrFloat, SectionAlignment};
use laboneq_scheduler::experiment::types::{
    ExternalParameterUid, Marker, MarkerSelector, SectionUid, SignalUid, ValueOrParameter,
};
use laboneq_scheduler::ir::{
    Acquire, Case, ChangeOscillatorPhase, InitialLocalOscillatorFrequency,
    InitialOscillatorFrequency, InitialVoltageOffset, IrKind, Loop, LoopKind, Match, MatchTarget,
    PlayPulse, PpcStep, PrngSetup, Section, SetOscillatorFrequency, Trigger,
};
use laboneq_scheduler::{RepetitionMode, ScheduleInfo, ScheduledNode};
use pyo3::IntoPyObjectExt;
use pyo3::{
    intern,
    prelude::*,
    types::{PyDict, PyList, PySet, PyString, PyTuple},
};
use std::collections::HashMap;

/// Type representing `laboneq.compiler.scheduler.interval_schedule.IntervalSchedule`
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
) -> PyResult<IntervalSchedule> {
    let mut context: Context<'_, 'py> = Context {
        id_store: &experiment.id_store,
        py_objects: &experiment.py_object_store,
        py_string_store: HashMap::new(),
    };
    let root = generate_py_schedules_impl(py, scheduled_node, &mut context, None)?.unwrap();
    Ok(root.into())
}

struct Context<'a, 'py> {
    id_store: &'a NamedIdStore,
    py_objects: &'a PyObjectInterner<ExternalParameterUid>,
    // Cache for Python string objects corresponding to NamedId keys
    py_string_store: HashMap<NamedId, Bound<'py, PyString>>,
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
            &scheduled_node.schedule,
            obj,
        )?)),
        IrKind::InitialLocalOscillatorFrequency(obj) => Ok(Some(
            handle_initial_local_oscillator_frequency(py, context, &scheduled_node.schedule, obj)?,
        )),
        IrKind::InitialVoltageOffset(obj) => Ok(Some(handle_initial_voltage_offset(
            py,
            context,
            &scheduled_node.schedule,
            obj,
        )?)),
        IrKind::Loop(obj) => {
            let py_schedule = handle_loop_schedule(py, context, &scheduled_node.schedule, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    py_schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(py_schedule))
        }
        IrKind::LoopIteration => {
            let schedule = handle_loop_iteration(
                py,
                context,
                &scheduled_node.schedule,
                *parent_section_uid.unwrap(),
            )?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, parent_section_uid)?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::LoopIterationPreamble => {
            let schedule = handle_loop_iteration_preamble(py, context, &scheduled_node.schedule)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, parent_section_uid)?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::SetOscillatorFrequency(obj) => {
            let schedule =
                handle_set_oscillator_frequency(py, context, &scheduled_node.schedule, obj)?;
            Ok(Some(schedule))
        }
        IrKind::ResetOscillatorPhase { signals } => {
            let schedule = handle_phase_reset(py, context, &scheduled_node.schedule, signals)?;
            Ok(Some(schedule))
        }
        IrKind::PpcStep(obj) => {
            let schedule = handle_ppc_steps(py, context, &scheduled_node.schedule, obj)?;
            Ok(Some(schedule))
        }
        IrKind::Section(obj) => {
            let section = handle_section(py, context, &scheduled_node.schedule, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    section
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(section))
        }
        IrKind::Acquire(obj) => {
            let schedule = handle_acquisition(py, context, &scheduled_node.schedule, obj)?;
            Ok(Some(schedule))
        }
        IrKind::Delay { signal } => {
            let schedule = handle_delay(py, context, &scheduled_node.schedule, *signal)?;
            Ok(Some(schedule))
        }
        IrKind::ClearPrecompensation { signal } => {
            let schedule =
                handle_reset_precompensation(py, context, &scheduled_node.schedule, *signal)?;
            Ok(Some(schedule))
        }
        IrKind::PlayPulse(obj) => {
            let schedule = handle_play_pulse(py, context, &scheduled_node.schedule, obj)?;
            Ok(Some(schedule))
        }
        IrKind::ChangeOscillatorPhase(obj) => {
            let schedule =
                handle_change_oscillator_phase(py, context, &scheduled_node.schedule, obj)?;
            Ok(Some(schedule))
        }
        IrKind::Match(obj) => {
            let section = handle_match(py, context, &scheduled_node.schedule, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    section
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(section))
        }
        IrKind::Case(obj) => {
            let schedule = handle_case(py, context, &scheduled_node.schedule, obj)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid))?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(schedule))
        }
        IrKind::Root => {
            let schedule = handle_root_schedule(py, context, &scheduled_node.schedule)?;
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) =
                    generate_py_schedules_impl(py, &child.node, context, None)?
                {
                    schedule
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            Ok(Some(schedule))
        }
    }
}

fn handle_root_schedule<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.root_schedule"))?;
    let py_obj = m.getattr(intern!(py, "RootSchedule"))?;
    let signals = schedule
        .signals
        .iter()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_loop_iteration_preamble<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.loop_iteration_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "LoopIterationPreambleSchedule"))?;
    let signals = schedule
        .signals
        .iter()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "length"), py.None())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_loop_iteration<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    parent_section: SectionUid,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.loop_iteration_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "LoopIterationSchedule"))?;
    let signals = schedule
        .signals
        .iter()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "length"), py.None())?;
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    kwargs.set_item(
        intern!(py, "right_aligned"),
        schedule.alignment_mode == SectionAlignment::Right,
    )?;
    kwargs.set_item(intern!(py, "section"), ctx.py_string(py, parent_section)?)?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_initial_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &InitialOscillatorFrequency,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.oscillator_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "InitialOscillatorFrequencySchedule"))?;
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
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "values"), out)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_initial_local_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &InitialLocalOscillatorFrequency,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.oscillator_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "InitialLocalOscillatorFrequencySchedule"))?;
    let value: f64 = obj.value.try_into().unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "value"), value)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_initial_voltage_offset<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &InitialVoltageOffset,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.voltage_offset"))?;
    let py_obj = m.getattr(intern!(py, "InitialOffsetVoltageSchedule"))?;

    let value: f64 = obj.value.try_into().unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "value"), value)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_set_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &SetOscillatorFrequency,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.oscillator_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "OscillatorFrequencyStepSchedule"))?;

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
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(intern!(py, "values"), out)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_phase_reset<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    signals: &[SignalUid],
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.phase_reset_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "PhaseResetSchedule"))?;

    let signals = signals
        .iter()
        .map(|s| ctx.py_string(py, *s).unwrap())
        .collect::<Vec<_>>();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, signals)?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_ppc_steps<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &PpcStep,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.ppc_step_schedule"))?;
    let py_obj = m.getattr(intern!(py, "PPCStepSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
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
    schedule: &ScheduleInfo,
    obj: &Section,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.section_schedule"))?;
    let py_schedule = m.getattr(intern!(py, "SectionSchedule"))?;

    let py_signals = schedule
        .signals
        .iter()
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
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(
        intern!(py, "right_aligned"),
        schedule.alignment_mode == SectionAlignment::Right,
    )?;
    kwargs.set_item(intern!(py, "trigger_output"), PySet::new(py, py_triggers)?)?;
    kwargs.set_item(
        intern!(py, "prng_setup"),
        obj.prng_setup
            .as_ref()
            .map(|o| create_py_prng_info(py, o))
            .transpose()?,
    )?;
    kwargs.set_item(
        intern!(py, "play_after"),
        schedule
            .play_after
            .iter()
            .map(|section_uid| ctx.py_string(py, *section_uid).unwrap())
            .collect::<Vec<_>>(),
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
    schedule: &ScheduleInfo,
    obj: &Acquire,
) -> Result<Bound<'py, PyAny>> {
    if obj.kernels.len() > 1 {
        return handle_acquire_group_schedule(py, ctx, schedule, obj);
    }
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_obj = m.getattr(intern!(py, "PulseSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(
        intern!(py, "integration_length"),
        obj.integration_length.value(),
    )?;
    if let Some(pulse_def) = obj.kernels.first() {
        let pulse = ctx.py_string(py, *pulse_def).unwrap();
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

    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_acquire_group_schedule<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &Acquire,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(
        py,
        "laboneq.compiler.scheduler.acquire_group_schedule"
    ))?;
    let py_obj = m.getattr(intern!(py, "AcquireGroupSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(
        intern!(py, "handle"),
        ctx.py_string(py, obj.handle).unwrap(),
    )?;
    let pulses = obj
        .kernels
        .iter()
        .map(|pulse_def| ctx.py_string(py, *pulse_def).unwrap())
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

    let py_schedule = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_delay<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    signal: SignalUid,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_pulse_schedule = m.getattr(intern!(py, "PulseSchedule"))?;
    let signal_string = ctx.py_string(py, signal).unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, [&signal_string])?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
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
    schedule: &ScheduleInfo,
    signal: SignalUid,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_clear_precompensation = m.getattr(intern!(py, "PrecompClearSchedule"))?;

    let signal_string = ctx.py_string(py, signal).unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, [&signal_string])?)?;
    kwargs.set_item(intern!(py, "signal"), &signal_string)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    let py_schedule = py_clear_precompensation.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn handle_play_pulse<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &PlayPulse,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_obj = m.getattr(intern!(py, "PulseSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(intern!(py, "integration_length"), py.None())?;
    kwargs.set_item(intern!(py, "pulse"), ctx.py_string(py, obj.pulse)?)?;
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
    schedule: &ScheduleInfo,
    obj: &ChangeOscillatorPhase,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_obj = m.getattr(intern!(py, "PulseSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
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
    schedule: &ScheduleInfo,
    obj: &Loop,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.loop_schedule"))?;
    let py_schedule = m.getattr(intern!(py, "LoopSchedule"))?;

    let py_signals = schedule
        .signals
        .iter()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    let uid = ctx.py_string(py, obj.uid)?;
    kwargs.set_item(intern!(py, "section"), uid)?;
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(
        intern!(py, "right_aligned"),
        schedule.alignment_mode == SectionAlignment::Right,
    )?;
    kwargs.set_item(intern!(py, "iterations"), obj.iterations)?;
    if let Some(repetition_mode) = &schedule.repetition_mode {
        let (mode, time) = repetition_mode_to_py(py, repetition_mode)?;
        kwargs.set_item(intern!(py, "repetition_mode"), mode)?;
        kwargs.set_item(intern!(py, "repetition_time"), time)?;
    } else {
        kwargs.set_item(intern!(py, "repetition_mode"), py.None())?;
        kwargs.set_item(intern!(py, "repetition_time"), py.None())?;
    }
    kwargs.set_item(intern!(py, "compressed"), obj.compressed())?;
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
                .map(|param_uid| ctx.py_string(py, *param_uid).unwrap())
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
    schedule: &ScheduleInfo,
    obj: &Match,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.match_schedule"))?;
    let py_schedule = m.getattr(intern!(py, "MatchSchedule"))?;

    let py_signals = schedule
        .signals
        .iter()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    let uid = ctx.py_string(py, obj.uid)?;
    kwargs.set_item(intern!(py, "section"), uid)?;
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(intern!(py, "right_aligned"), false)?;
    kwargs.set_item(intern!(py, "trigger_output"), PySet::empty(py)?)?;
    kwargs.set_item(
        intern!(py, "play_after"),
        schedule
            .play_after
            .iter()
            .map(|section_uid| ctx.py_string(py, *section_uid).unwrap())
            .collect::<Vec<_>>(),
    )?;
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
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}

fn repetition_mode_to_py<'py>(
    py: Python<'py>,
    repetition_mode: &RepetitionMode,
) -> PyResult<(Bound<'py, PyAny>, Py<PyAny>)> {
    let m = py.import(intern!(py, "laboneq.core.types.enums.repetition_mode"))?;
    let py_obj = m.getattr(intern!(py, "RepetitionMode"))?;
    match repetition_mode {
        RepetitionMode::Fastest => {
            let obj = py_obj.getattr(intern!(py, "FASTEST"))?;
            Ok((obj, py.None()))
        }
        RepetitionMode::Constant { time } => {
            let obj = py_obj.getattr(intern!(py, "CONSTANT"))?;
            Ok((obj, time.value().into_py_any(py)?))
        }
        RepetitionMode::Auto => {
            let obj = py_obj.getattr(intern!(py, "AUTO"))?;
            Ok((obj, py.None()))
        }
    }
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
    schedule: &ScheduleInfo,
    obj: &Case,
) -> Result<Bound<'py, PyAny>> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.case_schedule"))?;
    let py_schedule = m.getattr(intern!(py, "CaseSchedule"))?;

    let py_signals = schedule
        .signals
        .iter()
        .map(|s| ctx.py_string(py, *s).map_err(Into::into))
        .collect::<Result<Vec<Bound<'_, PyString>>>>()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "section"), ctx.py_string(py, obj.uid)?)?;
    kwargs.set_item(intern!(py, "state"), obj.state)?;
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "compressed_loop_grid"),
        schedule.compressed_loop_grid.value(),
    )?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(
        intern!(py, "length"),
        schedule.try_length().map(|v| v.value()),
    )?;
    kwargs.set_item(
        intern!(py, "right_aligned"),
        schedule.alignment_mode == SectionAlignment::Right,
    )?;
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
}
