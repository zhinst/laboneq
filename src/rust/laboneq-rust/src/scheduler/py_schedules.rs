// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::{
    prelude::*,
    types::{PyDict, PySet},
};
use std::collections::HashSet;

use crate::error::Result;
use laboneq_common::named_id::NamedIdStore;
use laboneq_scheduler::{ScheduledNode, ir::IrKind};

/// A compatibility structure to hold Python schedule objects.
#[pyclass(name = "Schedules", frozen)]
#[derive(Default)]
pub struct PyScheduleCompat {
    #[pyo3(get)]
    pub initial_oscillator_frequency: Vec<Py<PyAny>>,
    #[pyo3(get)]
    pub initial_local_oscillator_frequency: Vec<Py<PyAny>>,
    #[pyo3(get)]
    pub initial_voltage_offset: Vec<Py<PyAny>>,
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
        initial_oscillator_frequency: Vec::new(),
        initial_local_oscillator_frequency: Vec::new(),
        initial_voltage_offset: Vec::new(),
        id_store,
    };
    generate_py_schedules_impl(py, scheduled_node, &mut context)?;
    Ok(PyScheduleCompat {
        initial_oscillator_frequency: context.initial_oscillator_frequency,
        initial_local_oscillator_frequency: context.initial_local_oscillator_frequency,
        initial_voltage_offset: context.initial_voltage_offset,
    })
}

struct Context<'a> {
    initial_oscillator_frequency: Vec<Py<PyAny>>,
    initial_local_oscillator_frequency: Vec<Py<PyAny>>,
    initial_voltage_offset: Vec<Py<PyAny>>,
    id_store: &'a NamedIdStore,
}

fn generate_py_schedules_impl(
    py: Python,
    scheduled_node: &ScheduledNode,
    context: &mut Context<'_>,
) -> Result<()> {
    for child in scheduled_node.children.iter() {
        match &child.node.kind {
            IrKind::InitialOscillatorFrequency(obj) => {
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
                    .map(|(k, v)| {
                        let k = context.id_store.resolve(*k).unwrap().to_string();
                        signals.insert(k.clone());
                        (k, (*v).try_into().unwrap())
                    })
                    .collect();
                let kwargs = PyDict::new(py);
                kwargs.set_item("grid", child.node.schedule.grid)?;
                kwargs.set_item("signals", PySet::new(py, signals)?)?;
                kwargs.set_item("values", out)?;
                kwargs.set_item("length", child.node.schedule.length)?;
                let py_schedule = py_obj.call((), Some(&kwargs))?;
                context
                    .initial_oscillator_frequency
                    .push(py_schedule.into());
            }
            IrKind::InitialLocalOscillatorFrequency(obj) => {
                let m = py.import("laboneq")?;
                let py_obj = m
                    .getattr("compiler")
                    .and_then(|m| m.getattr("scheduler"))
                    .and_then(|m| m.getattr("oscillator_schedule"))
                    .and_then(|m| m.getattr("InitialLocalOscillatorFrequencySchedule"))?;
                let value: f64 = obj.value.try_into().unwrap();
                let kwargs = PyDict::new(py);
                kwargs.set_item("grid", child.node.schedule.grid)?;
                kwargs.set_item(
                    "signals",
                    HashSet::from([context.id_store.resolve(obj.signal).unwrap()]),
                )?;
                kwargs.set_item("value", value)?;
                kwargs.set_item("length", child.node.schedule.length)?;
                let py_schedule = py_obj.call((), Some(&kwargs))?;

                context
                    .initial_local_oscillator_frequency
                    .push(py_schedule.into());
            }
            IrKind::InitialVoltageOffset(obj) => {
                let m = py.import("laboneq")?;
                let py_obj = m
                    .getattr("compiler")
                    .and_then(|m| m.getattr("scheduler"))
                    .and_then(|m| m.getattr("voltage_offset"))
                    .and_then(|m| m.getattr("InitialOffsetVoltageSchedule"))?;
                let value: f64 = obj.value.try_into().unwrap();
                let kwargs = PyDict::new(py);
                kwargs.set_item("grid", child.node.schedule.grid)?;
                kwargs.set_item(
                    "signals",
                    HashSet::from([context.id_store.resolve(obj.signal).unwrap()]),
                )?;
                kwargs.set_item("value", value)?;
                kwargs.set_item("length", child.node.schedule.length)?;
                let py_schedule = py_obj.call((), Some(&kwargs))?;

                context
                    .initial_local_oscillator_frequency
                    .push(py_schedule.into());
            }
            _ => {
                generate_py_schedules_impl(py, &child.node, context)?;
            }
        }
    }
    Ok(())
}
