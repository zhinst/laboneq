// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::DeviceKind;
use pyo3::prelude::*;

use crate::error::{Error, Result};
use crate::scheduler::experiment::Device;

#[pyclass(name = "Device", frozen)]
pub struct DevicePy {
    pub uid: String,
    pub physical_device_uid: u16,
    pub kind: DeviceKind,
    pub is_shfqc: bool,
}

#[pymethods]
impl DevicePy {
    #[new]
    pub fn new(uid: String, physical_device_uid: u16, kind: &str, is_shfqc: bool) -> Self {
        Self {
            uid,
            physical_device_uid,
            kind: extract_device_kind(kind).unwrap(),
            is_shfqc,
        }
    }
}

fn extract_device_kind(device: &str) -> Result<DeviceKind> {
    let kind = match device {
        "HDAWG" => DeviceKind::Hdawg,
        "SHFQA" => DeviceKind::Shfqa,
        "SHFSG" => DeviceKind::Shfsg,
        "UHFQA" => DeviceKind::Uhfqa,
        "PRETTYPRINTERDEVICE" => DeviceKind::PrettyPrinterDevice,
        _ => {
            return Err(Error::new(format!("Unknown device type: {device}")));
        }
    };
    Ok(kind)
}

pub(super) fn py_device_to_device(
    device: &DevicePy,
    id_store: &mut NamedIdStore,
) -> Result<Device> {
    let device = Device {
        uid: id_store.get_or_insert(&device.uid).into(),
        physical_device_uid: device.physical_device_uid.into(),
        kind: device.kind,
        is_shfqc: device.is_shfqc,
    };
    Ok(device)
}
