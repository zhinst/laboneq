// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
pub(crate) mod device_traits;
pub mod ir;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
