// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod device_traits;
pub mod generate_code;
pub mod ir;
pub mod node;
pub(crate) mod passes;
pub mod signature;
pub mod tinysample;
pub(crate) mod utils;
pub(crate) mod virtual_signal;

pub type Samples = u64;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
