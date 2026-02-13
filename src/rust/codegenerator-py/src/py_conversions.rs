// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Translations from Python IR into code generator IR
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::RandomState;
use std::sync::Arc;

use codegenerator::ir::compilation_job as cjob;
use codegenerator::ir::compilation_job::Device;
use codegenerator::ir::compilation_job::{AwgCore, Signal};
use codegenerator::utils::length_to_samples;
use laboneq_common::named_id::NamedIdStore;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

fn extract_awg_signal(
    ob: &Bound<'_, PyAny>,
    sampling_rate: f64,
    id_store: &NamedIdStore,
) -> Result<Signal, PyErr> {
    // compilation_job.SignalObj
    let py = ob.py();
    let signal_type = match ob
        .getattr(intern!(py, "signal_type"))?
        .cast_into::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "integration" => cjob::SignalKind::INTEGRATION,
        "iq" => cjob::SignalKind::IQ,
        "single" => cjob::SignalKind::SINGLE,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown signal type: {ob}"
            )));
        }
    };
    let start_delay_seconds = ob.getattr(intern!(py, "start_delay"))?.extract::<f64>()?;
    let signal_delay_seconds = ob.getattr(intern!(py, "delay_signal"))?.extract::<f64>()?;
    let start_delay = length_to_samples(start_delay_seconds, sampling_rate);
    let signal_delay = length_to_samples(signal_delay_seconds, sampling_rate);
    let automute = ob.getattr(intern!(py, "automute"))?.extract::<bool>()?;
    let signal_id = ob.getattr(intern!(py, "id"))?;
    let signal_uid = signal_id.extract::<&str>()?;
    let oscillator_id: Option<String> = ob.getattr(intern!(py, "oscillator_id"))?.extract()?;
    let oscillator_is_hardware: Option<bool> = ob
        .getattr(intern!(py, "oscillator_is_hardware"))?
        .extract()?;

    let oscillator = match (oscillator_id, oscillator_is_hardware) {
        (Some(uid), Some(true)) => Some(cjob::Oscillator {
            uid,
            kind: cjob::OscillatorKind::HARDWARE,
        }),
        (Some(uid), Some(false)) => Some(cjob::Oscillator {
            uid,
            kind: cjob::OscillatorKind::SOFTWARE,
        }),
        _ => None,
    };

    let signal = Signal {
        uid: id_store
            .get(signal_uid)
            .expect("Expected signal to exist")
            .into(),
        kind: signal_type,
        channels: ob.getattr(intern!(py, "channels"))?.extract::<Vec<u8>>()?,
        start_delay,
        signal_delay,
        oscillator,
        automute,
    };
    Ok(signal)
}

pub(crate) fn extract_device_kind(ob: &Bound<'_, PyAny>) -> Result<cjob::DeviceKind, PyErr> {
    // device_type.DeviceType
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let kind = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "HDAWG" => cjob::DeviceKind::HDAWG,
        "SHFQA" => cjob::DeviceKind::SHFQA,
        "SHFSG" => cjob::DeviceKind::SHFSG,
        "UHFQA" => cjob::DeviceKind::UHFQA,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown device type: {ob}"
            )));
        }
    };
    Ok(kind)
}

fn extract_awg_kind(ob: &Bound<'_, PyAny>) -> Result<cjob::AwgKind, PyErr> {
    // awg_info.AWGSignalType
    let py = ob.py();
    let out = match ob
        .getattr(intern!(py, "name"))?
        .cast_into::<PyString>()?
        .to_cow()?
        .as_ref()
    {
        "IQ" => cjob::AwgKind::IQ,
        "SINGLE" => cjob::AwgKind::SINGLE,
        "DOUBLE" => cjob::AwgKind::DOUBLE,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown awg signal type: {ob}"
            )));
        }
    };
    Ok(out)
}

fn extract_awg_oscs(ob: &Bound<'_, PyAny>) -> Result<HashMap<String, u16>, PyErr> {
    // awg_info.AWGInfo.oscs
    let out = ob.extract::<HashMap<String, u16>>()?;
    Ok(out)
}

fn extract_trigger_mode(ob: &Bound<'_, PyAny>) -> Result<cjob::TriggerMode, PyErr> {
    // compilation_job.TriggerMode
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let mode = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "NONE" => cjob::TriggerMode::ZSync,
        "DIO_TRIGGER" => cjob::TriggerMode::DioTrigger,
        "DIO_WAIT" => cjob::TriggerMode::DioWait,
        "INTERNAL_TRIGGER_WAIT" => cjob::TriggerMode::InternalTriggerWait,
        "INTERNAL_READY_CHECK" => cjob::TriggerMode::InternalReadyCheck,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown trigger mode: {ob}"
            )));
        }
    };
    Ok(mode)
}

pub(crate) fn extract_awg(
    ob: &Bound<'_, PyAny>,
    id_store: &NamedIdStore,
) -> Result<AwgCore, PyErr> {
    // awg_info.AWGInfo
    let py = ob.py();
    let sampling_rate = ob.getattr(intern!(py, "sampling_rate"))?.extract::<f64>()?;
    let device_kind = extract_device_kind(&ob.getattr(intern!(py, "device_type"))?)?;
    let signals: Result<Vec<Arc<Signal>>, PyErr> = ob
        .getattr(intern!(py, "signals"))?
        .try_iter()?
        .map(|x| {
            let sig = extract_awg_signal(&x?, sampling_rate, id_store)?;
            Ok(Arc::new(sig))
        })
        .collect();
    let signal_kinds = HashSet::<_, RandomState>::from_iter(
        signals.as_ref().unwrap().iter().map(|s| s.kind.clone()),
    );
    assert!(
        signal_kinds.len() == 1
            || signal_kinds.len() == 2 && signal_kinds.contains(&cjob::SignalKind::INTEGRATION),
        "AWG signals must be of the same type, or of two types, where one is INTEGRATION. Found: {:?}",
        &signal_kinds
    );
    let awg_id = ob.getattr(intern!(py, "awg_id"))?.extract::<u16>()?;
    let trigger_mode = extract_trigger_mode(&ob.getattr(intern!(py, "trigger_mode"))?)?;
    let reference_clock_source = &ob
        .getattr(intern!(py, "reference_clock_source"))?
        .extract::<Option<String>>()?;
    let is_reference_clock_internal = reference_clock_source
        .as_ref()
        .map(|s| s == "internal")
        .unwrap_or(false);
    let awg = AwgCore::new(
        awg_id,
        extract_awg_kind(&ob.getattr(intern!(py, "signal_type"))?)?,
        signals?,
        sampling_rate,
        Arc::new(Device::new(
            ob.getattr(intern!(py, "device_id"))?
                .extract::<String>()?
                .into(),
            device_kind,
        )),
        extract_awg_oscs(&ob.getattr(intern!(py, "oscs"))?)?,
        Some(trigger_mode),
        is_reference_clock_internal,
    );
    Ok(awg)
}
