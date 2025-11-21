// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::{atomic::AtomicBool, atomic::Ordering};

#[doc(hidden)]
pub use log as _log;

#[macro_export]
macro_rules! info {
    ($msg:literal, $($arg:tt)+) => {
        laboneq_log::_log::info!(target: concat!("laboneq.rust::", module_path!()), $msg, $($arg)+);
    };
    ($msg:literal) => {
        laboneq_log::_log::info!(target: concat!("laboneq.rust::", module_path!()), $msg);
    };
}

#[macro_export]
macro_rules! warn {
    ($msg:literal, $($arg:tt)+) => {
        laboneq_log::_log::warn!(target: concat!("laboneq.rust::", module_path!()), $msg, $($arg)+);
    };
    ($msg:literal) => {
        laboneq_log::_log::warn!(target: concat!("laboneq.rust::", module_path!()), $msg);
    };
}

/// Log a diagnostic message at info level if diagnostics logging is enabled.
#[macro_export]
macro_rules! diagnostic {
    ($msg:literal, $($arg:tt)+) => {
        if laboneq_log::is_diagnostics_enabled() {
             laboneq_log::_log::info!(target: concat!("laboneq.rust::", module_path!()), $msg, $($arg)+);
        }
    };
    ($msg:literal) => {
        if laboneq_log::is_diagnostics_enabled() {
            laboneq_log::_log::info!(target: concat!("laboneq.rust::", module_path!()), $msg);
        }
    };
}

static DIAGNOSTICS_ENABLED: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn is_diagnostics_enabled() -> bool {
    DIAGNOSTICS_ENABLED.load(Ordering::Acquire)
}

/// Initialize the logging.
///
/// This function is meant to be called once at the start of the program to
/// set up the logging configuration.
/// Currently the function does not set up any concrete logger.
pub fn init_logging(with_diagnostics: bool) {
    // TODO: We could implement a custom logger, or use an existing one, e.g. `env_logger` or `tracing`,
    // to have more control over the logging output format and destination.
    // However for now we skip that and just use the default logger, which is currently initialized by `pyo3-log`.
    // Here we just initialize the diagnostics flag, as its logging level is specific to `laboneq` Python,
    // and not a general Rust logging level.
    DIAGNOSTICS_ENABLED.store(with_diagnostics, Ordering::Release);
}
