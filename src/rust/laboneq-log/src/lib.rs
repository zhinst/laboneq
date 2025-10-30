// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

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
