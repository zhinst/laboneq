// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

impl Error {
    pub fn new<T>(msg: T) -> Self
    where
        T: Display,
    {
        Error::Anyhow(anyhow::anyhow!(msg.to_string()))
    }
}
