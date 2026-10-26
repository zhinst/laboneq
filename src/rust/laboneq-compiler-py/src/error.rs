// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Error handling for the Python bindings.
//! This module provides functionality to translate Rust errors
//! into Python exceptions.

use std::fmt::Display;
use std::sync::Arc;

use laboneq_common::named_id::{NamedIdStore, resolve_ids};
use laboneq_error::LabOneQError;
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

    /// Explain what the error meant for the caller, keeping the error itself
    /// as the cause.
    // Spares the backends a dependency on `anyhow`.
    // TODO: Actually use `LabOneQError` in the compiler instead of anyhow::Error
    pub fn context<T>(self, context: T) -> Self
    where
        T: Display,
    {
        let Error::Anyhow(e) = self;
        Error::Anyhow(e.context(context.to_string()))
    }
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        LabOneQException::new_err(error.to_string())
    }
}

impl From<capnp::Error> for Error {
    fn from(error: capnp::Error) -> Self {
        Error::Anyhow(error.into())
    }
}

impl From<laboneq_capnp::convert::NarrowError> for Error {
    fn from(error: laboneq_capnp::convert::NarrowError) -> Self {
        Error::Anyhow(error.into())
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

impl From<Error> for LabOneQError {
    // TODO: Actually use `LabOneQError` in the compiler instead of anyhow::Error
    fn from(error: Error) -> Self {
        let Error::Anyhow(e) = error;
        laboneq_error::laboneq_error!("{e}")
    }
}

impl From<LabOneQError> for Error {
    // TODO: Actually use `LabOneQError` in the compiler instead of anyhow::Error
    fn from(error: LabOneQError) -> Self {
        anyhow::anyhow!(error).into()
    }
}

/// A compilation failure, carrying the store that renders its interned IDs as names.
///
/// The pipeline works in interned IDs, so an error leaving it still names them by
/// number. Resolving the message eagerly would flatten [`LabOneQError`]'s variants and
/// lose what they carry: the resource usage that drives auto-chunking, and the Python
/// exception attached as `__cause__`. So the store travels with the error, and the
/// message is resolved once, when it becomes a Python exception.
pub(crate) struct CompilationError {
    error: LabOneQError,
    /// `None` when the message holds no interned IDs, because the stage that
    /// raised it already resolved its own -- as the preprocessor does.
    id_store: Option<Arc<NamedIdStore>>,
}

impl CompilationError {
    pub(crate) fn new(error: LabOneQError, id_store: Arc<NamedIdStore>) -> Self {
        CompilationError {
            error,
            id_store: Some(id_store),
        }
    }
}

impl From<LabOneQError> for CompilationError {
    fn from(error: LabOneQError) -> Self {
        CompilationError {
            error,
            id_store: None,
        }
    }
}

impl From<Error> for CompilationError {
    fn from(error: Error) -> Self {
        LabOneQError::from(error).into()
    }
}

impl From<CompilationError> for PyErr {
    fn from(error: CompilationError) -> Self {
        let CompilationError { error, id_store } = error;
        match id_store {
            Some(id_store) => error.to_pyerr(|msg| resolve_ids(&msg, &id_store)),
            None => error.into(),
        }
    }
}

/// Turns interned IDs in an error into the names they stand for.
pub trait ResolveErr<T> {
    fn resolve_err(self, id_store: &NamedIdStore) -> Result<T>;
}

impl<T, E> ResolveErr<T> for std::result::Result<T, E>
where
    E: Into<Error>,
{
    fn resolve_err(self, id_store: &NamedIdStore) -> Result<T> {
        self.map_err(|e| Error::new(resolve_ids(&create_error_message(e), id_store)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_err() {
        let mut store = NamedIdStore::new();
        let id = store.get_or_insert("q0/drive");
        let err: std::result::Result<(), Error> =
            Err(Error::new(format!("Signal {id} has no calibration")));

        let resolved = err.resolve_err(&store).unwrap_err();

        assert_eq!(resolved.to_string(), "Signal q0/drive has no calibration");
    }

    #[test]
    fn test_context_keeps_the_error_as_cause() {
        let err = Error::new("frequency is not a multiple of 200 MHz").context("Cannot set the LO");

        assert_eq!(
            create_error_message(err),
            "Cannot set the LO\nCaused by:\n  frequency is not a multiple of 200 MHz"
        );
    }

    #[test]
    fn test_resolve_err_keeps_ok() {
        let store = NamedIdStore::new();
        let ok: std::result::Result<u8, Error> = Ok(7);

        assert_eq!(ok.resolve_err(&store).unwrap(), 7);
    }
}
