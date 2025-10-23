// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `laboneq-scheduler`.
use std::collections::HashMap;
use std::collections::HashSet;

use crate::error::{Error, Result, create_error_message};
use crate::scheduler::experiment::Experiment;
use crate::scheduler::experiment_processor::process_experiment;
use crate::scheduler::parameter_store::create_parameter_store;
use crate::scheduler::py_schedules::PyScheduleCompat;
use crate::scheduler::py_schedules::generate_py_schedules;
use crate::scheduler::signal::{OscillatorPy, py_signal_to_signal};
use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_scheduler::ChunkingInfo;
use laboneq_scheduler::experiment::types::RepetitionMode;
use laboneq_scheduler::{Experiment as SchedulerExperiment, TinySample, schedule_experiment};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod py_conversion;
use py_conversion::build_experiment;
mod signal;
use signal::SignalPy;
pub mod experiment;
mod experiment_processor;
mod parameter_store;
mod py_schedules;

#[pyclass(name = "Experiment", frozen)]
struct ExperimentPy {
    pub inner: Experiment,
}

#[pyfunction(name = "build_experiment")]
fn build_experiment_py(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
) -> Result<ExperimentPy> {
    let mut builder = build_experiment(experiment)?;
    let signals = signals
        .into_iter()
        .map(|s| {
            let signal = py_signal_to_signal(s.py(), &s.borrow(), &mut builder)?;
            Ok((signal.uid, signal))
        })
        .collect::<Result<_>>()?;
    let mut experiment = Experiment {
        sections: builder.sections,
        id_store: builder.id_store,
        parameters: builder.parameters,
        pulses: builder.pulses,
        signals,
    };
    process_experiment(&mut experiment)?;
    Ok(ExperimentPy { inner: experiment })
}

type AwgKeyPy = i64;

#[pyclass(name = "ScheduleResult", frozen)]
struct ScheduleResult {
    #[pyo3(get)]
    max_acquisition_time_per_awg: HashMap<AwgKeyPy, f64>,
    #[pyo3(get)]
    repetition_info: Option<RepetitionInfoPy>,
    #[pyo3(get)]
    system_grid: TinySample,
    /// Parameters used in the experiment
    #[pyo3(get)]
    used_parameters: HashSet<String>,
    #[pyo3(get)]
    schedules: Py<PyScheduleCompat>,
}

#[pyclass(name = "RepetitionInfo", frozen)]
#[derive(Debug, Clone)]
struct RepetitionInfoPy {
    #[pyo3(get)]
    mode: String,
    #[pyo3(get)]
    time: Option<f64>,
    #[pyo3(get)]
    loop_uid: String,
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
    let experiment = &experiment.inner;
    let mut parameter_store = create_parameter_store(parameters, &experiment.id_store);
    let result = schedule_experiment(
        SchedulerExperiment {
            sections: experiment.sections.iter().collect(),
            id_store: &experiment.id_store,
            parameters: &experiment.parameters,
            pulses: &experiment.pulses,
        },
        &experiment.signals,
        &parameter_store,
        chunking_info.map(|(index, count)| ChunkingInfo { index, count }),
    )
    .map_err(|e| {
        let msg = create_error_message(e);
        Error::new(resolve_ids(&msg, &experiment.id_store))
    })?;
    let py_schedules = generate_py_schedules(py, &result.root.unwrap(), &experiment.id_store)?;
    let out = ScheduleResult {
        max_acquisition_time_per_awg: result
            .max_acquisition_time
            .into_iter()
            .map(|(k, v)| (k.0 as AwgKeyPy, v.into()))
            .collect(),
        repetition_info: result.repetition_info.map(|info| {
            let (mode, time) = match info.mode {
                RepetitionMode::Fastest => ("fastest", None),
                RepetitionMode::Constant { time } => ("constant", Some(time)),
                RepetitionMode::Auto => ("auto", None),
            };
            RepetitionInfoPy {
                mode: mode.into(),
                time,
                loop_uid: experiment
                    .id_store
                    .resolve(info.loop_uid)
                    .map(|s| s.to_string())
                    .unwrap(),
            }
        }),
        system_grid: result.system_grid,
        used_parameters: parameter_store
            .empty_queries()
            .iter()
            .map(|p| experiment.id_store.resolve(p.0).unwrap().to_string())
            .collect(),
        schedules: Py::new(py, py_schedules)?,
    };
    Ok(out)
}

pub fn create_py_module<'a>(
    parent: &Bound<'a, PyModule>,
    name: &str,
) -> Result<Bound<'a, PyModule>> {
    let m = PyModule::new(parent.py(), name)?;
    m.add_function(wrap_pyfunction!(schedule_experiment_py, &m)?)?;
    m.add_function(wrap_pyfunction!(build_experiment_py, &m)?)?;
    // Intermediate migration objects, shall be removed later
    m.add_class::<SignalPy>()?;
    m.add_class::<OscillatorPy>()?;
    Ok(m)
}
