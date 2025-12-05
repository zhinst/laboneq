// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{
    intern,
    prelude::*,
    types::{PyDict, PyList, PySet, PyString, PyTuple},
};
use std::collections::HashMap;

use crate::scheduler::{experiment::Experiment, py_pulse_parameters::pulse_parameters_to_py_dict};
use crate::{error::Result, scheduler::py_object_interner::PyObjectInterner};
use laboneq_common::named_id::{NamedId, NamedIdStore};
use laboneq_scheduler::ir::{
    Acquire, InitialLocalOscillatorFrequency, InitialOscillatorFrequency, InitialVoltageOffset,
    IrKind, PpcStep, Section, SetOscillatorFrequency,
};
use laboneq_scheduler::{ScheduleInfo, ScheduledNode};
use laboneq_scheduler::{
    experiment::types::{ExternalParameterUid, SectionUid, SignalUid},
    ir::Trigger,
};

/// Type representing `laboneq.compiler.scheduler.interval_schedule.IntervalSchedule`
type IntervalSchedule = Py<PyAny>;

/// An schedule structure for multiple dimensions: Loop Visits -> Loop Iteration -> [IntervalSchedules]
/// i.e. loop is visited 10 times, if its parent has 10 iterations.
type MultiDimIntervalSchedule = Vec<Vec<Vec<IntervalSchedule>>>;

/// A compatibility structure to hold Python schedule objects.
#[pyclass(name = "Schedules", frozen)]
#[derive(Default)]
pub(super) struct PyScheduleCompat {
    #[pyo3(get)]
    pub initial_oscillator_frequency: Vec<IntervalSchedule>,
    #[pyo3(get)]
    pub initial_local_oscillator_frequency: Vec<IntervalSchedule>,
    #[pyo3(get)]
    pub initial_voltage_offset: Vec<IntervalSchedule>,
    /// Oscillator frequency steps per loop UID.
    #[pyo3(get)]
    pub oscillator_frequency_steps: HashMap<String, Vec<IntervalSchedule>>,
    /// Phase resets per loop UID
    #[pyo3(get)]
    pub phase_resets: HashMap<String, Vec<IntervalSchedule>>,
    /// PPC Steps per loop UID.
    #[pyo3(get)]
    pub ppc_steps: HashMap<String, MultiDimIntervalSchedule>,
    /// Section per section UID and how many times visited (i.e. in case of loops).
    #[pyo3(get)]
    pub sections: HashMap<String, Vec<IntervalSchedule>>,
    /// Acquire schedules per section UID and in the order of appearance in depth-first traversal.
    #[pyo3(get)]
    pub acquire_schedules: HashMap<String, Vec<IntervalSchedule>>,
    /// Section delays per section UID and in the order of appearance in depth-first traversal.
    #[pyo3(get)]
    pub section_delays: HashMap<String, Vec<IntervalSchedule>>,
}

/// Converts a [`ScheduledNode`] into a [`PyScheduleCompat`], a container for Python schedule objects.
///
/// This is a temporary solution to facilitate the transition from Python-based scheduling to Rust-based scheduling.
/// It allows us to return Python schedule objects from Rust functions, enabling gradual migration of scheduling logic
/// to Rust without breaking existing functionality.
pub(super) fn generate_py_schedules<'py>(
    py: Python<'py>,
    scheduled_node: &ScheduledNode,
    experiment: &Experiment,
) -> PyResult<PyScheduleCompat> {
    let mut context: Context<'_, 'py> = Context {
        id_store: &experiment.id_store,
        py_objects: &experiment.py_object_store,
        py_string_store: HashMap::new(),
        loop_visit_count: HashMap::new(),
        section_visit_count: HashMap::new(),
        initial_oscillator_frequency: Vec::new(),
        initial_local_oscillator_frequency: Vec::new(),
        initial_voltage_offset: Vec::new(),
        oscillator_frequency_steps: HashMap::new(),
        phase_resets: HashMap::new(),
        ppc_steps: HashMap::new(),
        sections: HashMap::new(),
        acquire_schedules: HashMap::new(),
        section_delays: HashMap::new(),
    };
    generate_py_schedules_impl(py, scheduled_node, &mut context, None, None)?;
    Ok(PyScheduleCompat {
        initial_oscillator_frequency: context.initial_oscillator_frequency,
        initial_local_oscillator_frequency: context.initial_local_oscillator_frequency,
        initial_voltage_offset: context.initial_voltage_offset,
        oscillator_frequency_steps: context.oscillator_frequency_steps,
        phase_resets: context.phase_resets,
        ppc_steps: context.ppc_steps,
        sections: context.sections,
        acquire_schedules: context.acquire_schedules,
        section_delays: context.section_delays,
    })
}

struct Context<'a, 'py> {
    id_store: &'a NamedIdStore,
    py_objects: &'a PyObjectInterner<ExternalParameterUid>,
    // Cache for Python string objects corresponding to NamedId keys
    py_string_store: HashMap<NamedId, Bound<'py, PyString>>,
    // Visit counters
    loop_visit_count: HashMap<SectionUid, usize>, // Track loop visit counts in case for nested loops
    section_visit_count: HashMap<SectionUid, usize>, // Track section visit counts
    /// Collected Python schedule objects.
    initial_oscillator_frequency: Vec<IntervalSchedule>,
    initial_local_oscillator_frequency: Vec<IntervalSchedule>,
    initial_voltage_offset: Vec<IntervalSchedule>,
    oscillator_frequency_steps: HashMap<String, Vec<IntervalSchedule>>,
    phase_resets: HashMap<String, Vec<IntervalSchedule>>,
    ppc_steps: HashMap<String, MultiDimIntervalSchedule>,
    sections: HashMap<String, Vec<IntervalSchedule>>,
    acquire_schedules: HashMap<String, Vec<IntervalSchedule>>,
    section_delays: HashMap<String, Vec<IntervalSchedule>>,
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
    parent_iteration: Option<usize>,
) -> Result<Option<IntervalSchedule>> {
    match &scheduled_node.kind {
        IrKind::InitialOscillatorFrequency(obj) => {
            handle_initial_oscillator_frequency(py, context, &scheduled_node.schedule, obj)?;
            Ok(None)
        }
        IrKind::InitialLocalOscillatorFrequency(obj) => {
            handle_initial_local_oscillator_frequency(py, context, &scheduled_node.schedule, obj)?;
            Ok(None)
        }
        IrKind::InitialVoltageOffset(obj) => {
            handle_initial_voltage_offset(py, context, &scheduled_node.schedule, obj)?;
            Ok(None)
        }
        IrKind::Loop(obj) => {
            // We track loop visits separately to iteration counts (for preamble sections)
            context.loop_visit_count.insert(
                obj.uid,
                context.loop_visit_count.get(&obj.uid).unwrap_or(&0) + 1,
            );
            for (i, child) in scheduled_node.children.iter().enumerate() {
                // Loop iteration carries the UID of the parent loop.
                // Therefore each visit of the loop increments the visit count of the parent loop.
                context.section_visit_count.insert(
                    obj.uid,
                    context.section_visit_count.get(&obj.uid).unwrap_or(&0) + 1,
                );
                generate_py_schedules_impl(py, &child.node, context, Some(&obj.uid), Some(i))?;
            }
            Ok(None)
        }
        IrKind::SetOscillatorFrequency(obj) => {
            handle_set_oscillator_frequency(
                py,
                context,
                *parent_section_uid.unwrap(),
                &scheduled_node.schedule,
                obj,
            )?;
            Ok(None)
        }
        IrKind::ResetOscillatorPhase { signals } => {
            handle_phase_reset(
                py,
                context,
                *parent_section_uid.unwrap(),
                &scheduled_node.schedule,
                signals,
            )?;
            Ok(None)
        }
        IrKind::PpcStep(obj) => {
            handle_ppc_steps(
                py,
                context,
                *parent_section_uid.unwrap(),
                &scheduled_node.schedule,
                obj,
                parent_iteration.unwrap(),
            )?;
            Ok(None)
        }
        IrKind::Section(obj) => {
            let section = handle_section(py, context, &scheduled_node.schedule, obj)?;
            context.section_visit_count.insert(
                obj.uid,
                context.section_visit_count.get(&obj.uid).unwrap_or(&0) + 1,
            );
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) = generate_py_schedules_impl(
                    py,
                    &child.node,
                    context,
                    Some(&obj.uid),
                    parent_iteration,
                )? {
                    section
                        .getattr(intern!(py, "children"))?
                        .call_method1(intern!(py, "append"), (child_schedule,))?;
                }
            }
            context
                .sections
                .entry(context.id_store.resolve(obj.uid).unwrap().to_string())
                .or_default()
                .push(section.clone().into());
            Ok(Some(section.into()))
        }
        IrKind::Acquire(obj) => {
            handle_acquisition(
                py,
                context,
                &scheduled_node.schedule,
                obj,
                *parent_section_uid.unwrap(),
            )?;
            Ok(None)
        }
        IrKind::Delay { signal } => {
            // Delay node can only have precompensation children
            let has_precompensation = !scheduled_node.children.is_empty();
            handle_delay(
                py,
                context,
                &scheduled_node.schedule,
                *signal,
                *parent_section_uid.unwrap(),
                has_precompensation,
            )?;
            Ok(None)
        }
        _ => {
            let parent_section_uid = scheduled_node
                .kind
                .section_info()
                .map(|info| Some(info.uid))
                .unwrap_or(parent_section_uid);
            for child in scheduled_node.children.iter() {
                generate_py_schedules_impl(
                    py,
                    &child.node,
                    context,
                    parent_section_uid,
                    parent_iteration,
                )?;
            }
            Ok(None)
        }
    }
}

fn handle_initial_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &InitialOscillatorFrequency,
) -> Result<()> {
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
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.initial_oscillator_frequency.push(py_schedule.into());
    Ok(())
}

fn handle_initial_local_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &InitialLocalOscillatorFrequency,
) -> Result<()> {
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
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.initial_local_oscillator_frequency
        .push(py_schedule.into());
    Ok(())
}

fn handle_initial_voltage_offset<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &InitialVoltageOffset,
) -> Result<()> {
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
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.initial_voltage_offset.push(py_schedule.into());
    Ok(())
}

fn handle_set_oscillator_frequency<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    parent_section_uid: SectionUid,
    schedule: &ScheduleInfo,
    obj: &SetOscillatorFrequency,
) -> Result<()> {
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
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.oscillator_frequency_steps
        .entry(
            ctx.id_store
                .resolve(parent_section_uid)
                .unwrap()
                .to_string(),
        )
        .or_default()
        .push(py_schedule.into());
    Ok(())
}

fn handle_phase_reset<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    parent_section_uid: SectionUid,
    schedule: &ScheduleInfo,
    signals: &[SignalUid],
) -> Result<()> {
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
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.phase_resets
        .entry(
            ctx.id_store
                .resolve(parent_section_uid)
                .unwrap()
                .to_string(),
        )
        .or_default()
        .push(py_schedule.into());
    Ok(())
}

fn handle_ppc_steps<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    parent_section_uid: SectionUid,
    schedule: &ScheduleInfo,
    obj: &PpcStep,
    iteration: usize,
) -> Result<()> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.ppc_step_schedule"))?;
    let py_obj = m.getattr(intern!(py, "PPCStepSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
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
    let loop_global_iterations = ctx
        .ppc_steps
        .entry(
            ctx.id_store
                .resolve(parent_section_uid)
                .unwrap()
                .to_string(),
        )
        .or_default();
    let loop_visit_count = ctx.loop_visit_count.get(&parent_section_uid).unwrap();
    if loop_global_iterations.len() < *loop_visit_count {
        loop_global_iterations.push(Vec::new());
    }
    // Loop visits, Loop Iterations, PPC Steps
    let loop_local_iterations = &mut loop_global_iterations[loop_visit_count - 1];
    if loop_local_iterations.len() <= iteration {
        loop_local_iterations.push(Vec::new());
    }
    loop_local_iterations[iteration].push(py_schedule.into());
    Ok(())
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
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, py_signals)?)?;
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    kwargs.set_item(intern!(py, "right_aligned"), false)?; // TODO
    kwargs.set_item(intern!(py, "trigger_output"), PySet::new(py, py_triggers)?)?;
    let py_schedule: Bound<'_, PyAny> = py_schedule.call((), Some(&kwargs))?;
    Ok(py_schedule)
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
    parent_section_uid: SectionUid,
) -> Result<()> {
    if obj.kernels.len() > 1 {
        handle_acquire_group_schedule(py, ctx, schedule, obj, parent_section_uid)?;
        return Ok(());
    }
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_obj = m.getattr(intern!(py, "PulseSchedule"))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(
        intern!(py, "signals"),
        PySet::new(py, [ctx.py_string(py, obj.signal).unwrap()])?,
    )?;
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
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

    let parent_section = ctx
        .acquire_schedules
        .entry(
            ctx.id_store
                .resolve(parent_section_uid)
                .expect("Acquire must have a valid parent section")
                .to_string(),
        )
        .or_default();
    parent_section.push(py_schedule.into());
    Ok(())
}

fn handle_acquire_group_schedule<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    obj: &Acquire,
    parent_section_uid: SectionUid,
) -> Result<()> {
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
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
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

    let parent_section = ctx
        .acquire_schedules
        .entry(
            ctx.id_store
                .resolve(parent_section_uid)
                .expect("Acquire must have a valid parent section")
                .to_string(),
        )
        .or_default();
    parent_section.push(py_schedule.into());
    Ok(())
}

fn handle_delay<'py>(
    py: Python<'py>,
    ctx: &mut Context<'_, 'py>,
    schedule: &ScheduleInfo,
    signal: SignalUid,
    parent_section_uid: SectionUid,
    has_precompensation: bool,
) -> Result<()> {
    let m = py.import(intern!(py, "laboneq.compiler.scheduler.pulse_schedule"))?;
    let py_pulse_schedule = m.getattr(intern!(py, "PulseSchedule"))?;
    let signal_string = ctx.py_string(py, signal).unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "grid"), schedule.grid.value())?;
    kwargs.set_item(intern!(py, "signals"), PySet::new(py, [&signal_string])?)?;
    kwargs.set_item(intern!(py, "length"), schedule.length.value())?;
    kwargs.set_item(intern!(py, "integration_length"), py.None())?;
    kwargs.set_item(intern!(py, "pulse"), py.None())?;
    kwargs.set_item(intern!(py, "is_acquire"), false)?;
    kwargs.set_item(intern!(py, "phase"), 0.0)?;
    kwargs.set_item(intern!(py, "amplitude"), 1.0)?;
    let py_schedule = py_pulse_schedule.call((), Some(&kwargs))?;

    let parent_section = ctx
        .section_delays
        .entry(
            ctx.id_store
                .resolve(parent_section_uid)
                .expect("Delay must have a valid parent section")
                .to_string(),
        )
        .or_default();
    parent_section.push(py_schedule.into());

    if has_precompensation {
        let py_clear_precompensation = m.getattr(intern!(py, "PrecompClearSchedule"))?;
        let kwargs_clear = PyDict::new(py);
        kwargs_clear.set_item(intern!(py, "grid"), schedule.grid.value())?;
        kwargs_clear.set_item(intern!(py, "signal"), signal_string)?;
        kwargs_clear.set_item(intern!(py, "length"), schedule.length.value())?;
        let py_clear_schedule = py_clear_precompensation.call((), Some(&kwargs_clear))?;
        parent_section.push(py_clear_schedule.into());
    }
    Ok(())
}
