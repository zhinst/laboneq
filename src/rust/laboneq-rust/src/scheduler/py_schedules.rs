// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{
    intern,
    prelude::*,
    types::{PyDict, PySet},
};
use std::collections::{HashMap, HashSet};

use crate::error::Result;
use laboneq_common::named_id::NamedIdStore;
use laboneq_scheduler::experiment::types::{SectionUid, SignalUid};
use laboneq_scheduler::ir::{
    InitialLocalOscillatorFrequency, InitialOscillatorFrequency, InitialVoltageOffset, IrKind,
    PpcStep, Section, SetOscillatorFrequency,
};
use laboneq_scheduler::{ScheduleInfo, ScheduledNode};

/// Type representing `laboneq.compiler.scheduler.interval_schedule.IntervalSchedule`
type IntervalSchedule = Py<PyAny>;

/// An schedule structure for multiple dimensions: Loop Visits -> Loop Iteration -> [IntervalSchedules]
/// i.e. loop is visited 10 times, if its parent has 10 iterations.
type MultiDimIntervalSchedule = Vec<Vec<Vec<IntervalSchedule>>>;

/// A compatibility structure to hold Python schedule objects.
#[pyclass(name = "Schedules", frozen)]
#[derive(Default)]
pub struct PyScheduleCompat {
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
}

/// Converts a [`ScheduledNode`] into a [`PyScheduleCompat`], a container for Python schedule objects.
///
/// This is a temporary solution to facilitate the transition from Python-based scheduling to Rust-based scheduling.
/// It allows us to return Python schedule objects from Rust functions, enabling gradual migration of scheduling logic
/// to Rust without breaking existing functionality.
pub fn generate_py_schedules(
    py: Python,
    scheduled_node: &ScheduledNode,
    id_store: &NamedIdStore,
) -> PyResult<PyScheduleCompat> {
    let mut context = Context {
        id_store,
        loop_visit_count: HashMap::new(),
        initial_oscillator_frequency: Vec::new(),
        initial_local_oscillator_frequency: Vec::new(),
        initial_voltage_offset: Vec::new(),
        oscillator_frequency_steps: HashMap::new(),
        phase_resets: HashMap::new(),
        ppc_steps: HashMap::new(),
        sections: HashMap::new(),
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
    })
}

struct Context<'a> {
    id_store: &'a NamedIdStore,
    loop_visit_count: HashMap<SectionUid, usize>, // Track loop visit counts in case for nested loops
    /// Collected Python schedule objects.
    initial_oscillator_frequency: Vec<IntervalSchedule>,
    initial_local_oscillator_frequency: Vec<IntervalSchedule>,
    initial_voltage_offset: Vec<IntervalSchedule>,
    oscillator_frequency_steps: HashMap<String, Vec<IntervalSchedule>>,
    phase_resets: HashMap<String, Vec<IntervalSchedule>>,
    ppc_steps: HashMap<String, MultiDimIntervalSchedule>,
    sections: HashMap<String, Vec<IntervalSchedule>>,
}

fn generate_py_schedules_impl<'ctx>(
    py: Python,
    scheduled_node: &'ctx ScheduledNode,
    context: &mut Context<'ctx>,
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
            context.loop_visit_count.insert(
                obj.uid,
                context.loop_visit_count.get(&obj.uid).unwrap_or(&0) + 1,
            );
            for (i, child) in scheduled_node.children.iter().enumerate() {
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
            let parent_section_uid = scheduled_node
                .kind
                .section_info()
                .map(|info| Some(info.uid))
                .unwrap_or(parent_section_uid);
            for child in scheduled_node.children.iter() {
                if let Some(child_schedule) = generate_py_schedules_impl(
                    py,
                    &child.node,
                    context,
                    parent_section_uid,
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

fn handle_initial_oscillator_frequency(
    py: Python,
    ctx: &mut Context,
    schedule: &ScheduleInfo,
    obj: &InitialOscillatorFrequency,
) -> Result<()> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("oscillator_schedule"))
        .and_then(|m| m.getattr("InitialOscillatorFrequencySchedule"))?;
    let mut signals = HashSet::new();
    let out: Vec<(String, f64)> = obj
        .values
        .iter()
        .map(|(signal, freq)| {
            let k = ctx.id_store.resolve(*signal).unwrap().to_string();
            signals.insert(k.clone());
            (k, (*freq).try_into().unwrap())
        })
        .collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item("signals", PySet::new(py, signals)?)?;
    kwargs.set_item("values", out)?;
    kwargs.set_item("length", schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.initial_oscillator_frequency.push(py_schedule.into());
    Ok(())
}

fn handle_initial_local_oscillator_frequency(
    py: Python,
    ctx: &mut Context,
    schedule: &ScheduleInfo,
    obj: &InitialLocalOscillatorFrequency,
) -> Result<()> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("oscillator_schedule"))
        .and_then(|m| m.getattr("InitialLocalOscillatorFrequencySchedule"))?;
    let value: f64 = obj.value.try_into().unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item(
        "signals",
        HashSet::from([ctx.id_store.resolve(obj.signal).unwrap()]),
    )?;
    kwargs.set_item("value", value)?;
    kwargs.set_item("length", schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.initial_local_oscillator_frequency
        .push(py_schedule.into());
    Ok(())
}

fn handle_initial_voltage_offset(
    py: Python,
    ctx: &mut Context,
    schedule: &ScheduleInfo,
    obj: &InitialVoltageOffset,
) -> Result<()> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("voltage_offset"))
        .and_then(|m| m.getattr("InitialOffsetVoltageSchedule"))?;
    let value: f64 = obj.value.try_into().unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item(
        "signals",
        HashSet::from([ctx.id_store.resolve(obj.signal).unwrap()]),
    )?;
    kwargs.set_item("value", value)?;
    kwargs.set_item("length", schedule.length.value())?;
    let py_schedule = py_obj.call((), Some(&kwargs))?;
    ctx.initial_voltage_offset.push(py_schedule.into());
    Ok(())
}

fn handle_set_oscillator_frequency(
    py: Python,
    ctx: &mut Context,
    parent_section_uid: SectionUid,
    schedule: &ScheduleInfo,
    obj: &SetOscillatorFrequency,
) -> Result<()> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("oscillator_schedule"))
        .and_then(|m| m.getattr("OscillatorFrequencyStepSchedule"))?;
    let mut signals = HashSet::new();
    let out: Vec<(String, f64)> = obj
        .values
        .iter()
        .map(|(k, v)| {
            let k = ctx.id_store.resolve(*k).unwrap().to_string();
            signals.insert(k.clone());
            (k, (*v).try_into().unwrap())
        })
        .collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item("signals", PySet::new(py, signals)?)?;
    kwargs.set_item("values", out)?;
    kwargs.set_item("length", schedule.length.value())?;
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

fn handle_phase_reset(
    py: Python,
    ctx: &mut Context,
    parent_section_uid: SectionUid,
    schedule: &ScheduleInfo,
    signals: &[SignalUid],
) -> Result<()> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("phase_reset_schedule"))
        .and_then(|m| m.getattr("PhaseResetSchedule"))?;
    let signals: HashSet<String> = signals
        .iter()
        .map(|s| ctx.id_store.resolve(*s).unwrap().to_string())
        .collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item("signals", PySet::new(py, signals)?)?;
    kwargs.set_item("length", schedule.length.value())?;
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

fn handle_ppc_steps(
    py: Python,
    ctx: &mut Context,
    parent_section_uid: SectionUid,
    schedule: &ScheduleInfo,
    obj: &PpcStep,
    iteration: usize,
) -> Result<()> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("ppc_step_schedule"))
        .and_then(|m| m.getattr("PPCStepSchedule"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item(
        "signals",
        PySet::new(py, [ctx.id_store.resolve(obj.signal).unwrap().to_string()])?,
    )?;
    kwargs.set_item("length", schedule.length.value())?;
    kwargs.set_item("trigger_duration", obj.trigger_duration.value())?;
    kwargs.set_item(
        "ppc_device",
        ctx.id_store.resolve(obj.device).unwrap().to_string(),
    )?;
    kwargs.set_item("ppc_channel", obj.channel)?;

    if let Some(pump_power) = obj.pump_power {
        let f: f64 = pump_power.try_into().unwrap();
        kwargs.set_item("pump_power", f)?;
    }
    if let Some(pump_freq) = obj.pump_frequency {
        let f: f64 = pump_freq.try_into().unwrap();
        kwargs.set_item("pump_frequency", f)?;
    }
    if let Some(probe_power) = obj.probe_power {
        let f: f64 = probe_power.try_into().unwrap();
        kwargs.set_item("probe_power", f)?;
    }
    if let Some(probe_frequency) = obj.probe_frequency {
        let f: f64 = probe_frequency.try_into().unwrap();
        kwargs.set_item("probe_frequency", f)?;
    }
    if let Some(cancellation_phase) = obj.cancellation_phase {
        let f: f64 = cancellation_phase.try_into().unwrap();
        kwargs.set_item("cancellation_phase", f)?;
    }
    if let Some(cancellation_attenuation) = obj.cancellation_attenuation {
        let f: f64 = cancellation_attenuation.try_into().unwrap();
        kwargs.set_item("cancellation_attenuation", f)?;
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

fn handle_section<'a>(
    py: Python<'a>,
    ctx: &mut Context,
    schedule: &ScheduleInfo,
    obj: &Section,
) -> Result<Bound<'a, PyAny>> {
    let m = py.import("laboneq")?;
    let py_obj = m
        .getattr("compiler")
        .and_then(|m| m.getattr("scheduler"))
        .and_then(|m| m.getattr("section_schedule"))
        .and_then(|m| m.getattr("SectionSchedule"))?;
    let signals: HashSet<String> = schedule
        .signals
        .iter()
        .map(|s| ctx.id_store.resolve(*s).unwrap().to_string())
        .collect();

    let kwargs = PyDict::new(py);
    let uid = ctx.id_store.resolve(obj.uid).unwrap().to_string();
    kwargs.set_item("section", uid.clone())?;
    kwargs.set_item("grid", schedule.grid.value())?;
    kwargs.set_item("signals", signals)?;
    kwargs.set_item("length", schedule.length.value())?;
    kwargs.set_item("right_aligned", false)?; // TODO
    let py_schedule: Bound<'_, PyAny> = py_obj.call((), Some(&kwargs))?;
    Ok(py_schedule)
}
