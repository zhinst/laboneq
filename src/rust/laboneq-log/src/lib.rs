// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub use log;

#[macro_export]
macro_rules! info {
    ($msg:literal, $($arg:tt)+) => {
        log::info!(target: concat!("laboneq.rust::", module_path!()), $msg, $($arg)+);
    };
    ($msg:literal) => {
        log::info!(target: concat!("laboneq.rust::", module_path!()), $msg);
    };
}

#[macro_export]
macro_rules! warn {
    ($msg:literal, $($arg:tt)+) => {
        log::warn!(target: concat!("laboneq.rust::", module_path!()), $msg, $($arg)+);
    };
    ($msg:literal) => {
        log::warn!(target: concat!("laboneq.rust::", module_path!()), $msg);
    };
}
