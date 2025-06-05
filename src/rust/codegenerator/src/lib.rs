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

pub use utils::string_sanitize;

pub type Samples = u64;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

impl Error {
    pub fn new(msg: &str) -> Self {
        Error::Anyhow(anyhow::anyhow!(msg.to_string()))
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
