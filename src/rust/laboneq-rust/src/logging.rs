// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

use laboneq_log::init_logging;

#[pyfunction(name = "init_logging")]
pub fn init_logging_py(log_level: i64) {
    // A level between Python info and debug, a custom `laboneq` logging level
    const DIAGNOSTICS_LEVEL: i64 = 15;
    let log_diagnostics = log_level <= DIAGNOSTICS_LEVEL;
    init_logging(log_diagnostics);
}
