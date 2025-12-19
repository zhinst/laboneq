// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `laboneq-scheduler`.
use std::collections::HashMap;
use std::collections::HashSet;

use crate::error::{Error, Result, create_error_message};
use crate::scheduler::experiment::DeviceSetup;
use crate::scheduler::experiment::Experiment;
use crate::scheduler::experiment_context::ExperimentContext;
use crate::scheduler::experiment_context::experiment_context_from_experiment;
use crate::scheduler::experiment_processor::process_experiment;
use crate::scheduler::experiment_validation::validate_experiment;
use crate::scheduler::parameter_store::create_parameter_store;
use crate::scheduler::py_conversion::ExperimentBuilder;
use crate::scheduler::py_device::DevicePy;
use crate::scheduler::py_device::py_device_to_device;
use crate::scheduler::py_schedules::PyScheduleCompat;
use crate::scheduler::py_schedules::generate_py_schedules;
use crate::scheduler::py_signal::AmplifierPumpPy;
use crate::scheduler::py_signal::SweepParameterPy;
use crate::scheduler::py_signal::{OscillatorPy, py_signal_to_signal};
use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_scheduler::{ChunkingInfo, ExperimentContext as SchedulerContext, schedule_experiment};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[cfg(test)]
mod test_py;

mod py_conversion;
use py_conversion::build_experiment;
mod py_signal;
use py_signal::SignalPy;
pub(crate) mod experiment;
mod experiment_context;
mod experiment_processor;
mod experiment_validation;
mod parameter_store;
mod pulse;
mod py_device;
mod py_export;
mod py_object_interner;
mod py_pulse_defs;
mod py_pulse_parameters;
mod py_schedules;
mod signal_view;

use signal_view::signal_views;

#[pyclass(name = "Experiment", frozen)]
struct ExperimentPy {
    pub inner: Experiment,
    pub device_setup: DeviceSetup,
    pub context: ExperimentContext,
}

#[pymethods]
impl ExperimentPy {
    /// Generate `PulseDef` Python objects for all pulses defined in the experiment.
    #[getter]
    fn pulse_defs(&self, py: Python) -> Vec<Py<PyAny>> {
        self.inner
            .pulses
            .values()
            .map(|v| py_pulse_defs::pulse_def_to_py(py, &self.inner.id_store, v).unwrap())
            .collect()
    }
}

fn build_device_setup(
    builder: &mut ExperimentBuilder,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
) -> Result<DeviceSetup> {
    let devices = devices
        .into_iter()
        .map(|d| {
            let device = py_device_to_device(&d.borrow(), &mut builder.id_store)?;
            Ok((device.uid, device))
        })
        .collect::<Result<_>>()?;
    let signals = signals
        .into_iter()
        .map(|s| {
            let signal = py_signal_to_signal(s.py(), &s.borrow(), builder)?;
            Ok((signal.uid, signal))
        })
        .collect::<Result<_>>()?;
    DeviceSetup::new(signals, devices)
}

pub(crate) fn experiment_py_to_experiment(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
) -> Result<(Experiment, DeviceSetup)> {
    let mut builder = ExperimentBuilder::new(experiment.py());
    let device_setup = build_device_setup(&mut builder, signals, devices)?;
    build_experiment(experiment, &mut builder)?;
    Ok((
        Experiment {
            root: builder.root,
            id_store: builder.id_store,
            parameters: builder.parameters,
            pulses: builder.pulses,
            py_object_store: builder.py_object_store,
        },
        device_setup,
    ))
}

#[pyfunction(name = "build_experiment")]
fn build_experiment_py(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
    devices: Vec<Bound<'_, DevicePy>>,
) -> Result<ExperimentPy> {
    let (mut experiment, device_setup) = experiment_py_to_experiment(experiment, signals, devices)
        .map_err(|e| {
            // NOTE: The error message here does not resolve IDs, as the experiment's ID store is not yet built.
            // and `experiment_py_to_experiment` has access to the actual UIDs before interning.
            let msg = create_error_message(e);
            Error::new(msg)
        })?;
    let context = experiment_context_from_experiment(&experiment);
    process_experiment(&mut experiment, &device_setup, &context).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    validate_experiment(&experiment, &device_setup).map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    Ok(ExperimentPy {
        inner: experiment,
        device_setup,
        context,
    })
}

#[pyclass(name = "ScheduleResult", frozen)]
struct ScheduleResult {
    #[pyo3(get)]
    system_grid: i64,
    /// Parameters used in the experiment
    #[pyo3(get)]
    used_parameters: HashSet<String>,
    #[pyo3(get)]
    schedules: Py<PyScheduleCompat>,
}

#[pyfunction(name = "schedule_experiment")]
fn schedule_experiment_py(
    py: Python,
    experiment: &ExperimentPy,
    parameters: HashMap<String, f64>,
    chunking_info: Option<(usize, usize)>, // Current chunk index, total chunk count
) -> Result<ScheduleResult> {
    // The NT / RT boundaries should have been resolved at this point.
    // This means only 1 NT root.
    let context = &experiment.context;
    let device_setup = &experiment.device_setup;
    let experiment = &experiment.inner;
    let mut parameter_store = create_parameter_store(parameters, &experiment.id_store);
    let result = schedule_experiment(
        &experiment.root,
        SchedulerContext {
            id_store: &experiment.id_store,
            parameters: experiment.parameters.clone(),
            signals: &signal_views(device_setup),
            handle_to_signal: &context.handle_to_signal,
        },
        &parameter_store,
        chunking_info.map(|(index, count)| ChunkingInfo { index, count }),
    )
    .map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    let py_schedules = generate_py_schedules(py, &result.root.unwrap(), experiment)?;
    let out = ScheduleResult {
        system_grid: result.system_grid.value(),
        used_parameters: parameter_store
            .empty_queries()
            .iter()
            .map(|p| experiment.id_store.resolve(p.0).unwrap().to_string())
            .collect(),
        schedules: Py::new(py, py_schedules)?,
    };
    Ok(out)
}

pub(crate) fn create_py_module<'py>(py: Python<'py>, name: &str) -> Result<Bound<'py, PyModule>> {
    let m = PyModule::new(py, name)?;
    m.add_function(wrap_pyfunction!(schedule_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(build_experiment_py, &m)?)?;
    // Intermediate migration objects, shall be removed later
    m.add_class::<SignalPy>()?;
    m.add_class::<DevicePy>()?;
    m.add_class::<OscillatorPy>()?;
    m.add_class::<SweepParameterPy>()?;
    m.add_class::<AmplifierPumpPy>()?;
    Ok(m)
}
