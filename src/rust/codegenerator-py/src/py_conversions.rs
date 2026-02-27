// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Translations from Python IR into code generator IR
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use codegenerator::AwgInfo;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

use codegenerator::ir::compilation_job::Device;
use codegenerator::ir::compilation_job::DeviceKind;
use codegenerator::ir::compilation_job::TriggerMode;
use laboneq_common::named_id::NamedIdStore;

pub(crate) fn extract_device_kind(ob: &Bound<'_, PyAny>) -> Result<DeviceKind, PyErr> {
    // device_type.DeviceType
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let kind = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "HDAWG" => DeviceKind::HDAWG,
        "SHFQA" => DeviceKind::SHFQA,
        "SHFSG" => DeviceKind::SHFSG,
        "UHFQA" => DeviceKind::UHFQA,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown device type: {ob}"
            )));
        }
    };
    Ok(kind)
}

fn extract_awg_oscs(ob: &Bound<'_, PyAny>) -> Result<HashMap<String, u16>, PyErr> {
    // awg_info.AWGInfo.oscs
    let out = ob.extract::<HashMap<String, u16>>()?;
    Ok(out)
}

fn extract_trigger_mode(ob: &Bound<'_, PyAny>) -> Result<TriggerMode, PyErr> {
    // compilation_job.TriggerMode
    let py = ob.py();
    let py_name = ob.getattr(intern!(py, "name"))?;
    let mode = match py_name.cast::<PyString>()?.to_cow()?.as_ref() {
        "NONE" => TriggerMode::ZSync,
        "DIO_TRIGGER" => TriggerMode::DioTrigger,
        "DIO_WAIT" => TriggerMode::DioWait,
        "INTERNAL_TRIGGER_WAIT" => TriggerMode::InternalTriggerWait,
        "INTERNAL_READY_CHECK" => TriggerMode::InternalReadyCheck,
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
) -> Result<AwgInfo, PyErr> {
    let py = ob.py();
    let sampling_rate = ob.getattr(intern!(py, "sampling_rate"))?.extract::<f64>()?;
    let device_kind = extract_device_kind(&ob.getattr(intern!(py, "device_type"))?)?;
    let awg_id = ob.getattr(intern!(py, "awg_id"))?.extract::<u16>()?;
    let trigger_mode = extract_trigger_mode(&ob.getattr(intern!(py, "trigger_mode"))?)?;
    let reference_clock_source = &ob
        .getattr(intern!(py, "reference_clock_source"))?
        .extract::<Option<String>>()?;
    let is_reference_clock_internal = reference_clock_source
        .as_ref()
        .map(|s| s == "internal")
        .unwrap_or(false);
    let mut signal_uids = HashSet::new();
    for signal in ob.getattr(intern!(py, "signals"))?.try_iter()? {
        let signal_uid_py = signal?.getattr(intern!(py, "id"))?;
        let signal_uid = signal_uid_py.extract::<&str>()?;
        let uid = id_store.get(signal_uid).unwrap();
        signal_uids.insert(uid.into());
    }
    let awg = AwgInfo {
        uid: awg_id,
        sampling_rate,
        device: Arc::new(Device::new(
            ob.getattr(intern!(py, "device_id"))?
                .extract::<String>()?
                .into(),
            device_kind,
        )),
        osc_allocation: extract_awg_oscs(&ob.getattr(intern!(py, "oscs"))?)?,
        trigger_mode,
        is_reference_clock_internal,
        signals: signal_uids,
    };
    Ok(awg)
}
