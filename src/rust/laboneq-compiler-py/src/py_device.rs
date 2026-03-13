// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

#[pyclass(name = "Device", frozen)]
pub struct DevicePy {
    pub uid: String,
    pub physical_device_uid: u16,
    pub kind: String,
    pub is_shfqc: bool,
    pub options: Vec<String>,
    pub reference_clock: Option<String>,
}

#[pymethods]
impl DevicePy {
    #[new]
    #[pyo3(signature = (uid, physical_device_uid, kind, is_shfqc=false, options=None, reference_clock=None))]
    pub fn new(
        uid: &str,
        physical_device_uid: u16,
        kind: &str,
        is_shfqc: bool,
        options: Option<Vec<String>>,
        reference_clock: Option<String>,
    ) -> Self {
        Self {
            uid: uid.to_string(),
            physical_device_uid,
            kind: kind.to_string(),
            is_shfqc,
            options: options.unwrap_or_default(),
            reference_clock,
        }
    }
}
