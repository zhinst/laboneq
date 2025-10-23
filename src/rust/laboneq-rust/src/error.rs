// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Error handling for the Python bindings.
//! This module provides functionality to translate Rust errors
//! into Python exceptions.

use std::fmt::Display;

use laboneq_scheduler::error::Error as SchedulerError;
use pyo3::import_exception;
use pyo3::prelude::*;
import_exception!(laboneq.core.exceptions, LabOneQException);

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

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        LabOneQException::new_err(error.to_string())
    }
}

impl From<SchedulerError> for Error {
    fn from(error: SchedulerError) -> Self {
        match error {
            SchedulerError::Anyhow(anyhow_err) => Error::Anyhow(anyhow_err),
        }
    }
}

impl From<PyErr> for Error {
    fn from(error: PyErr) -> Self {
        Error::Anyhow(error.into())
    }
}

/// Create a formatted error message.
pub fn create_error_message<T: Into<Error>>(error: T) -> String {
    let Error::Anyhow(e) = error.into();
    let mut causes = e
        .chain()
        .skip(1)
        .map(|cause| format!("{cause}"))
        .collect::<Vec<_>>();
    if causes.is_empty() {
        return format!("{e}");
    }
    // Reverse to show highest-level cause first
    causes.reverse();
    let msg = format!("Caused by:\n  {:}", causes.join("\n  "));
    format!("{e}\n{msg}")
}
