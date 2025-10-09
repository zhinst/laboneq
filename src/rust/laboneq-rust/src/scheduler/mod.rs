// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `laboneq-scheduler`.
use std::collections::HashMap;

use crate::error::Result;
use laboneq_common::named_id::NamedIdStore;
use laboneq_scheduler::ir::{Parameter, ParameterUid, PulseRef, PulseUid, SignalUid};
use laboneq_scheduler::{Experiment as SchedulerExperiment, IrNode, schedule_experiment};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod py_conversion;
use py_conversion::build_experiment;
mod signal;
use signal::{Signal, SignalPy};

struct Experiment {
    sections: Vec<IrNode>,
    id_store: NamedIdStore,
    parameters: HashMap<ParameterUid, Parameter>,
    pulses: HashMap<PulseUid, PulseRef>,
    signals: HashMap<SignalUid, Signal>,
}

#[pyclass(name = "Experiment", frozen)]
struct ExperimentPy {
    pub inner: Experiment,
}

#[pyfunction(name = "build_experiment")]
fn build_experiment_py(
    experiment: &Bound<'_, PyAny>,
    signals: Vec<Bound<'_, SignalPy>>,
) -> Result<ExperimentPy> {
    let builder = build_experiment(experiment)?;
    let mut id_store = builder.id_store;
    let signals = signals
        .into_iter()
        .map(|s| {
            let signal = s.borrow().to_signal(&mut id_store);
            (signal.uid, signal)
        })
        .collect();

    let experiment = Experiment {
        sections: builder.sections,
        id_store,
        parameters: builder.parameters,
        pulses: builder.pulses,
        signals,
    };
    Ok(ExperimentPy { inner: experiment })
}

type AwgKeyPy = i64;

#[pyclass(name = "ScheduleResult", frozen)]
#[derive(Default)]
struct ScheduleResult {
    #[pyo3(get)]
    max_acquisition_time_per_awg: HashMap<AwgKeyPy, f64>,
}

#[pyfunction(name = "schedule_experiment")]
fn schedule_experiment_py(experiment: &ExperimentPy) -> Result<ScheduleResult> {
    // The NT / RT boundaries should have been resolved at this point.
    // This means only 1 NT root.
    let experiment = &experiment.inner;
    let result = schedule_experiment(
        SchedulerExperiment {
            sections: experiment.sections.iter().collect(),
            id_store: &experiment.id_store,
            parameters: &experiment.parameters,
            pulses: &experiment.pulses,
        },
        &experiment.signals,
    )?;
    let out = ScheduleResult {
        max_acquisition_time_per_awg: result
            .max_acquisition_time
            .into_iter()
            .map(|(k, v)| (k.0 as AwgKeyPy, v.into()))
            .collect(),
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
    m.add_class::<SignalPy>()?;
    Ok(m)
}
