// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

use laboneq_common::types::AwgKey;

#[pyclass(name = "AwgInfo", frozen)]
pub struct AwgInfoPy {
    pub uid: AwgKey,
    pub number: Vec<u16>,
}

#[pymethods]
impl AwgInfoPy {
    #[new]
    pub fn new(uid: i64, number: Vec<u16>) -> Self {
        Self {
            uid: AwgKey(uid as u64),
            number,
        }
    }
}
