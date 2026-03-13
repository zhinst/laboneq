// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Logging utilities for the Python bindings.
//!
//! This module provides a function to initialize logging that bridges between Python and Rust,
//! allowing Rust logs to be captured and displayed according to the log level and loggers set in Python.

use std::sync::LazyLock;

use log::LevelFilter;

use pyo3::prelude::*;
use pyo3_log::Logger;
use pyo3_log::ResetHandle;

use laboneq_log::init_logging;

static PY_RESET_HANDLE: LazyLock<ResetHandle> = LazyLock::new(|| {
    let py_logger = Box::new(Logger::default());
    let handle = py_logger.reset_handle();
    log::set_boxed_logger(Box::new(py_logger)).unwrap();
    handle
});

/// A bridge between Python and Rust logging.
///
/// This function will setup logging that respects the log level and loggers set in Python.
///
/// Therefore this function should be called at the start of each compilation to ensure that Python log level
/// changes made between compilations are picked up.
///
/// Arguments:
/// - `log_level`: The Python log level as an integer. Follows the standard Python logging levels.
pub fn init_logging_py(log_level: i64) -> PyResult<()> {
    // Diagnostics is a custom `laboneq`-specific logging level between info and debug.
    // Therefore it needs to be handled separately here, and cannot be directly mapped to a Python log level.
    const DIAGNOSTICS_LEVEL: i64 = 15;
    let log_diagnostics = log_level <= DIAGNOSTICS_LEVEL;
    init_logging(log_diagnostics);
    PY_RESET_HANDLE.reset();

    let level_filter = match log_level {
        0 => LevelFilter::Warn,
        1..=10 => LevelFilter::Debug,
        11..=20 => LevelFilter::Info,
        21..=30 => LevelFilter::Warn,
        31..=40 => LevelFilter::Error,
        _ => LevelFilter::Error,
    };
    log::set_max_level(level_filter);
    Ok(())
}
