// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::DeviceKind;
use laboneq_ir::{device::builder::DeviceBuilder, system::Device};
use pyo3::prelude::*;

use crate::error::{Error, Result};

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
    let device = DeviceBuilder::new(
        id_store.get_or_insert(&device.uid).into(),
        device.physical_device_uid.into(),
        device.kind,
    )
    .shfqc(device.is_shfqc)
    .build();
    Ok(device)
}
