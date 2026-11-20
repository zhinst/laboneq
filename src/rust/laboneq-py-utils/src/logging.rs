// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Logging utilities for the Python bindings.
//!
//! This module provides a function to initialize logging that bridges between Python and Rust,
//! allowing Rust logs to be captured and displayed according to the log level and loggers set in Python.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fmt;
use std::sync::Once;
use std::sync::RwLock;

use log::{Level, LevelFilter, Log, Metadata, Record};

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyTuple;

use laboneq_log::init_logging;

static PYTHON_LOGGER: PythonLogger = PythonLogger::new();
static INSTALL_PYTHON_LOGGER: Once = Once::new();

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
    INSTALL_PYTHON_LOGGER.call_once(|| {
        log::set_logger(&PYTHON_LOGGER).expect("no other Rust logger is installed");
    });
    PYTHON_LOGGER.reset();

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

/// Forwards Rust log records to the Python logger named after their target,
/// with `::` replaced by `.`.
///
/// Caches the level each Python logger accepts, so that a record the logger
/// would drop does not take the GIL. For a target it has seen, a change to
/// the Python logging configuration applies after the next reset.
struct PythonLogger {
    levels: RwLock<BTreeMap<String, LevelFilter>>,
}

impl PythonLogger {
    const fn new() -> Self {
        Self {
            levels: RwLock::new(BTreeMap::new()),
        }
    }

    fn reset(&self) {
        self.levels.write().unwrap().clear();
    }

    fn level_filter(&self, target: &str) -> LevelFilter {
        if let Some(filter) = self.levels.read().unwrap().get(target) {
            return *filter;
        }
        // Taking the GIL while holding the lock would deadlock against a
        // thread that holds the GIL and waits for the lock.
        let filter = Python::try_attach(|py| {
            python_level_filter(py, target)
                .map_err(|err| report(py, err))
                .ok()
        });
        // A failed lookup stays out of the cache, so the next record retries it.
        let Some(filter) = filter.flatten() else {
            return LevelFilter::Off;
        };
        self.levels
            .write()
            .unwrap()
            .insert(target.to_owned(), filter);
        filter
    }
}

impl Log for PythonLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level_filter(metadata.target())
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        Python::try_attach(|py| {
            if let Err(err) = forward(py, record) {
                report(py, err);
            }
        });
    }

    fn flush(&self) {}
}

/// Returns the most verbose level the Python logger accepts, which accounts
/// for `logging.disable` as well as the logger's effective level.
fn python_level_filter(py: Python<'_>, target: &str) -> PyResult<LevelFilter> {
    let logger = python_logger(py, &python_logger_name(target))?;
    for level in [
        Level::Trace,
        Level::Debug,
        Level::Info,
        Level::Warn,
        Level::Error,
    ] {
        if logger
            .call_method1(intern!(py, "isEnabledFor"), (python_level(level),))?
            .is_truthy()?
        {
            return Ok(level.to_level_filter());
        }
    }
    Ok(LevelFilter::Off)
}

/// Builds the Python record from the Rust source location, so that it names
/// the Rust file and line rather than the Python caller.
fn forward(py: Python<'_>, record: &Record) -> PyResult<()> {
    let name = python_logger_name(record.target());
    let logger = python_logger(py, &name)?;
    let python_record = logger.call_method1(
        intern!(py, "makeRecord"),
        (
            name,
            python_level(record.level()),
            record.file(),
            record.line().unwrap_or_default(),
            record.args().to_string(),
            PyTuple::empty(py),
            py.None(),
        ),
    )?;
    logger.call_method1(intern!(py, "handle"), (python_record,))?;
    Ok(())
}

/// Reports an error raised by Python logging, which has no Python caller to
/// propagate to. A `KeyboardInterrupt` goes back to the main thread, so that
/// Ctrl-C during logging still interrupts.
fn report(py: Python<'_>, err: PyErr) {
    let interrupted = err.is_instance_of::<PyKeyboardInterrupt>(py)
        && py
            .import("_thread")
            .and_then(|thread| thread.call_method0("interrupt_main"))
            .is_ok();
    if !interrupted {
        err.write_unraisable(py, None);
    }
}

fn python_logger<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    static GET_LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    GET_LOGGER
        .import(py, "logging", "getLogger")?
        .call1((name,))
}

fn python_logger_name(target: &str) -> String {
    target.replace("::", ".")
}

fn python_level(level: Level) -> u32 {
    match level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    }
}

/// Issues a Python-visible `FutureWarning` and mirrors it to the Rust log,
/// routed through `PyErr_WarnEx` so it respects the `warnings` module's
/// filters (`-W` flags, `simplefilter("error")`, pytest capture, ...).
///
/// `FutureWarning` is used rather than `DeprecationWarning` because it isn't
/// hidden by Python's default warning filters.
///
/// Takes [`fmt::Arguments`] (like `log::warn!`/`format!`) so callers can pass
/// either a plain static message or a format string with arguments; prefer
/// the [`deprecation_warning!`] macro over calling this directly.
pub fn deprecation_warning(py: Python<'_>, message: fmt::Arguments<'_>) -> PyResult<()> {
    log::warn!("Deprecation warning: {message}");
    let message = CString::new(message.to_string())
        .expect("deprecation warning message must not contain NUL bytes");
    PyErr::warn(
        py,
        py.get_type::<pyo3::exceptions::PyFutureWarning>().as_any(),
        &message,
        2,
    )
}

/// Emits a deprecation warning, formatting its arguments like `format!`.
///
/// ```ignore
/// deprecation_warning!(py, "`foo` is deprecated")?;
/// deprecation_warning!(py, "`{name}` is deprecated, use `{replacement}` instead")?;
/// ```
#[macro_export]
macro_rules! deprecation_warning {
    ($py:expr, $($arg:tt)*) => {
        $crate::logging::deprecation_warning($py, format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwards_records_the_python_logger_accepts() {
        Python::attach(|py| {
            let handler = capture(py, "bridge_test.forward", 20);
            let logger = PythonLogger::new();
            logger.log(
                &Record::builder()
                    .target("bridge_test::forward")
                    .level(Level::Info)
                    .file(Some("src/lib.rs"))
                    .line(Some(7))
                    .args(format_args!("{}% done", 50))
                    .build(),
            );
            logger.log(
                &Record::builder()
                    .target("bridge_test::forward")
                    .level(Level::Debug)
                    .args(format_args!("dropped"))
                    .build(),
            );

            let records = handler.getattr("buffer").unwrap();
            assert_eq!(records.len().unwrap(), 1);
            let record = records.get_item(0).unwrap();
            let attr = |name: &str| record.getattr(name).unwrap();
            assert_eq!(
                attr("name").extract::<String>().unwrap(),
                "bridge_test.forward"
            );
            assert_eq!(attr("levelno").extract::<u32>().unwrap(), 20);
            assert_eq!(attr("pathname").extract::<String>().unwrap(), "src/lib.rs");
            assert_eq!(attr("lineno").extract::<u32>().unwrap(), 7);
            let message = record.call_method0("getMessage").unwrap();
            assert_eq!(message.extract::<String>().unwrap(), "50% done");
        });
    }

    #[test]
    fn applies_python_level_changes_after_reset() {
        Python::attach(|py| {
            let handler = capture(py, "bridge_test.reset", 30);
            let logger = PythonLogger::new();
            let log_info = || {
                logger.log(
                    &Record::builder()
                        .target("bridge_test::reset")
                        .level(Level::Info)
                        .args(format_args!("info"))
                        .build(),
                );
            };
            let records = handler.getattr("buffer").unwrap();

            log_info();
            assert_eq!(records.len().unwrap(), 0);

            python_logger(py, "bridge_test.reset")
                .unwrap()
                .call_method1("setLevel", (20,))
                .unwrap();
            log_info();
            assert_eq!(records.len().unwrap(), 0);

            logger.reset();
            log_info();
            assert_eq!(records.len().unwrap(), 1);
        });
    }

    #[test]
    fn filters_at_the_most_verbose_level_python_accepts() {
        Python::attach(|py| {
            let logger = python_logger(py, "bridge_test.levels").unwrap();
            for (python_level, filter) in [
                (5, LevelFilter::Trace),
                (10, LevelFilter::Debug),
                (15, LevelFilter::Info),
                (30, LevelFilter::Warn),
                (50, LevelFilter::Off),
            ] {
                logger.call_method1("setLevel", (python_level,)).unwrap();
                assert_eq!(
                    python_level_filter(py, "bridge_test::levels").unwrap(),
                    filter,
                    "Python level {python_level}"
                );
            }
        });
    }

    /// Attaches a handler that keeps every record to the Python logger `name`.
    fn capture<'py>(py: Python<'py>, name: &str, level: u32) -> Bound<'py, PyAny> {
        let handler = py
            .import("logging.handlers")
            .unwrap()
            .call_method1("BufferingHandler", (100,))
            .unwrap();
        let logger = py
            .import("logging")
            .unwrap()
            .call_method1("getLogger", (name,))
            .unwrap();
        logger.call_method1("setLevel", (level,)).unwrap();
        logger.call_method1("addHandler", (&handler,)).unwrap();
        handler
    }
}
