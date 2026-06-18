// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This module provides a Python wrapper for the `device_setup_fingerprint` function, allowing it to be called from Python code.
//!
//! It defines a `TargetSetup` struct to extract the relevant information from the Python `TargetSetup` object, and a `TargetDeviceType` enum to represent the device types defined in the corresponding Python enum.

use std::collections::HashMap;

use pyo3::intern;
use pyo3::{prelude::*, types::PyString};

use laboneq_common::device_setup_fingerprint::{
    InstrumentEntry, InstrumentEntryType, device_setup_fingerprint,
};
use laboneq_common::shfqc;

/// Compute a fingerprint for a device setup.
///
/// Accepts any Python object with a `devices` attribute whose items each expose
/// `uid`, `device_type` as enum, `device_options` and `is_qc`.
#[pyfunction(name = "device_setup_fingerprint")]
pub fn device_setup_fingerprint_py(device_setup: Bound<'_, PyAny>) -> PyResult<String> {
    let device_setup: TargetSetup = device_setup.extract()?;
    let entries = to_instrument_entries(device_setup.devices)?;
    Ok(device_setup_fingerprint(entries))
}

/// Helper struct to extract the relevant information from the Python `TargetSetup` object.
#[derive(FromPyObject)]
struct TargetSetup<'py> {
    devices: Vec<TargetDevice<'py>>,
}

#[derive(FromPyObject)]
struct TargetDevice<'py> {
    uid: Bound<'py, PyString>,
    device_type: TargetDeviceType,
    device_options: Option<Bound<'py, PyString>>,
    is_qc: bool,
}

/// Enum representing the types defined in `TargetDeviceType` Python enum.
enum TargetDeviceType {
    Uhfqa,
    Hdawg,
    Shfsg,
    Shfqa,
    Shfppc,
    Pqsc,
    Qhub,
    Zqcs,
    Unmanaged,
}

impl FromPyObject<'_, '_> for TargetDeviceType {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        match obj.getattr(intern!(py, "name"))?.extract()? {
            "UHFQA" => Ok(TargetDeviceType::Uhfqa),
            "HDAWG" => Ok(TargetDeviceType::Hdawg),
            "SHFSG" => Ok(TargetDeviceType::Shfsg),
            "SHFQA" => Ok(TargetDeviceType::Shfqa),
            "SHFPPC" => Ok(TargetDeviceType::Shfppc),
            "PQSC" => Ok(TargetDeviceType::Pqsc),
            "QHUB" => Ok(TargetDeviceType::Qhub),
            "NONQC" => Ok(TargetDeviceType::Unmanaged), // Special case for non-QC devices
            "ZQCS" => Ok(TargetDeviceType::Zqcs),
            dev => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown device type: {dev}"
            ))),
        }
    }
}

impl std::fmt::Display for TargetDeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TargetDeviceType::Uhfqa => "UHFQA",
            TargetDeviceType::Hdawg => "HDAWG",
            TargetDeviceType::Shfsg => "SHFSG",
            TargetDeviceType::Shfqa => "SHFQA",
            TargetDeviceType::Shfppc => "SHFPPC",
            TargetDeviceType::Pqsc => "PQSC",
            TargetDeviceType::Qhub => "QHUB",
            TargetDeviceType::Unmanaged => "UNMANAGED",
            TargetDeviceType::Zqcs => "ZQCS",
        };
        write!(f, "{s}")
    }
}

fn to_instrument_entries(devices: Vec<TargetDevice>) -> PyResult<Vec<InstrumentEntry>> {
    let mut entries: HashMap<String, InstrumentEntry> = HashMap::with_capacity(devices.len());

    for device in devices {
        if matches!(device.device_type, TargetDeviceType::Unmanaged) {
            continue; // Skip non-QC devices
        }
        let uid = device.uid.to_str()?;
        let options = match &device.device_options {
            Some(o) => parse_options(o.to_str()?),
            None => vec![],
        };

        // SHFQC is a special case: TargetSetup splits SHFQC into SHFQA and SHFSG, yet it does not have SHFQA if the original SHFQC does not have any logical signals defined.
        if device.is_qc {
            match device.device_type {
                TargetDeviceType::Shfsg => {
                    let base_uid = shfqc::to_base_uid(uid);
                    entries.entry(base_uid.clone()).or_insert_with(|| {
                        InstrumentEntry::new(
                            base_uid,
                            InstrumentEntryType::Shfqc { has_qa: false },
                            options.clone(),
                        )
                    });
                }
                TargetDeviceType::Shfqa => {
                    entries.insert(
                        uid.to_string(),
                        InstrumentEntry::new(
                            uid.to_string(),
                            InstrumentEntryType::Shfqc { has_qa: true },
                            options.clone(),
                        ),
                    );
                }
                _ => {}
            }
        } else {
            entries.insert(
                uid.to_string(),
                InstrumentEntry::new(
                    uid.to_string(),
                    InstrumentEntryType::NonShfqc(device.device_type.to_string()),
                    options,
                ),
            );
        }
    }
    Ok(entries.into_values().collect())
}

fn parse_options(raw: &str) -> Vec<String> {
    raw.split('/').map(|s| s.to_string()).collect()
}
