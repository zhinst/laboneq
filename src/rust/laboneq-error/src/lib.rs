// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//!
//! This crate introduces [`LabOneQError`] - a struct that holds a root cause error
//! and provides an easy way of stacking context messages while propagating the error.
//!
//! [LabOneQError] is an enum that represents all error conditions that one might need to handle
//! separately. The [`LabOneQError::Generic`] serves as a catch-all variant where we only care about
//! propagating a nice error to the user.
//!
//! Here are some guidelines on how to use [`LabOneQError`] when developing a crate:
//!
//!   - Most fallible functions in your crate return [`Result<whatever, LabOneQError>`]
//!   - Most internal errors are created as [`LabOneQError`] variants directly, or via laboneq_error::bail!(...)
//!   - For some internal errors you can still use specific error types (e.g. defined via thiserror), if it is
//!     needed for some local logic (e.g. you have a function call retry logic based on what error it returns).
//!     However, once it reaches the propagation stage, it should be transformed into [`LabOneQError`]
//!   - When an external error (i.e. originating from code in a different crate) happens, it is either
//!     immediately mapped to an appropriate variant of [`LabOneQError`], or is otherwise turned into an internal
//!     error type, and then processed according to the above guidelines for internal errors.
//!   - Once everything is [`LabOneQError`], error propagation is easy with the ? operator
//!   - While propagating, a stack of context messages can be built to better describe the error. In contrast
//!     to anyhow, this does not build an error chain.
//!   - Once an error bubbles up to the top of the propagation chain, we may need to process it before
//!     passing it forward through the crate boundary. In our codebase we have and we foresee only a handful
//!     of error types that need to be processed - most things are just sent to the user in their nice
//!     Display representation, without caring about the type. The mentioned handful of cases is represented
//!     by the variants of [`LabOneQError`], except [`LabOneQError::Generic`] which captures the rest.
//!   - If you are developing a new feature and existing variants do not cover your case, feel
//!     free to add new variants as necessary
//!   - anyhow is a nice library that among other things can achieve error propagation. [`ContextualError`]
//!     is a simplified alternative, so whenever you are considering to use anyhow::Error, you can instead use
//!     [`ContextualError`] directly or via [`LabOneQError`].
//!
pub mod resource_usage;

use core::error::Error as StdError;
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use thiserror::Error;

use crate::resource_usage::ResourceExhaustionError;

/// Basic type for unstructured error reporting
#[derive(Debug, Error)]
#[error("{0}")]
struct StringError(Cow<'static, str>);

#[derive(Debug)]
pub struct ContextualError<SourceError = Box<dyn StdError + Send + Sync + 'static>> {
    pub source: SourceError,
    context: Vec<Cow<'static, str>>,
}

impl ContextualError {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(msg: impl Into<Cow<'static, str>>) -> Self {
        ContextualError {
            source: Box::new(StringError(msg.into())),
            context: Vec::new(),
        }
    }

    // Instead of this, we could implement From<T>, but we don't do it on purpose. We don't
    // want automatic conversion of everything that qualifies as error into plain ContextualError,
    // but rather want it to be concious choice. See LabOneQError doc for more details.
    pub fn from_err<T>(err: T) -> Self
    where
        T: StdError + Send + Sync + 'static,
    {
        ContextualError {
            source: Box::new(err),
            context: vec![],
        }
    }
}

impl StdError for ContextualError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.source.as_ref())
    }
}

impl<E: Display> Display for ContextualError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)?;
        if !self.context.is_empty() {
            write!(f, "\n\nContext:\n")?;
            for ctx in self.context.iter() {
                writeln!(f, "    {}", ctx)?;
            }
        }
        Ok(())
    }
}

pub trait WithContext {
    fn add_context<F, C>(&mut self, f: F)
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>;

    fn with_context<F, C>(mut self, f: F) -> Self
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>,
        Self: Sized,
    {
        self.add_context(f);
        self
    }
}

impl<E> WithContext for ContextualError<E> {
    fn add_context<F, C>(&mut self, f: F)
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>,
    {
        self.context.push(f().into())
    }
}

impl<T, E: WithContext> WithContext for Result<T, E> {
    fn add_context<F, C>(&mut self, f: F)
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>,
    {
        let Err(e) = self else { return };
        e.add_context(f);
    }
}

/// A holder for a [`ContextualError`] that is guaranteed to wrap a [`pyo3::PyErr`].
///
/// It can be used without depending on pyo3, but constructing it or accessing the `PyErr` requires
/// enabling the `pyo3` feature on `laboneq_error`.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct PyErrorWithContext(ContextualError);

#[cfg(feature = "pyo3")]
impl PyErrorWithContext {
    pub fn py_err(&self) -> &pyo3::PyErr {
        self.0.source.downcast_ref::<_>().unwrap()
    }
}

#[cfg(feature = "pyo3")]
impl From<pyo3::PyErr> for PyErrorWithContext {
    fn from(py_err: pyo3::PyErr) -> Self {
        PyErrorWithContext(ContextualError::from_err(py_err))
    }
}

impl WithContext for PyErrorWithContext {
    fn add_context<F, C>(&mut self, f: F)
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>,
    {
        self.0.add_context(f);
    }
}

/// A concrete, typed error that enumerates error conditions that we (or a user) may need to handle
/// individually.
#[derive(Debug, Error)]
pub enum LabOneQError {
    #[error(transparent)]
    PulseSamplerCallback(#[from] PyErrorWithContext),

    #[error(transparent)]
    ResourceExhaustion(#[from] ResourceExhaustionError),

    #[error(transparent)]
    Generic(#[from] ContextualError),
}

#[cfg(feature = "pyo3")]
impl From<pyo3::PyErr> for LabOneQError {
    fn from(value: pyo3::PyErr) -> Self {
        Self::PulseSamplerCallback(value.into())
    }
}

impl LabOneQError {
    // Instead of this, we could implement From<T>, but we don't do it on purpose. The automatic
    // conversion is implemented for specific variants only, and From<T> would interfere with that.
    // If you need conversion to the generic variant, you have to do the concious choice of calling this method manually
    pub fn from_err<T>(err: T) -> Self
    where
        T: StdError + Send + Sync + 'static,
    {
        Self::Generic(ContextualError::from_err(err))
    }

    #[cfg(feature = "pyo3")]
    pub fn to_pyerr<F>(self, msg_updater: F) -> pyo3::PyErr
    where
        F: FnOnce(String) -> String,
    {
        pyo3::Python::attach(move |py| {
            let msg = self.to_string();
            let msg = msg_updater(msg);
            match self {
                LabOneQError::Generic(_) => LabOneQException::new_err(msg),
                LabOneQError::PulseSamplerCallback(err) => {
                    let py_err = LabOneQException::new_err(msg);
                    // If the error is a PyErr, mark it as the 'cause' of the new Python exception.
                    // Python will then properly display the traceback of both.
                    py_err.set_cause(py, Some(err.py_err().clone_ref(py)));
                    py_err
                }
                LabOneQError::ResourceExhaustion(err) => pyo3::PyErr::from_type(
                    ResourceLimitationError::type_object(py),
                    (msg, err.usage),
                ),
            }
        })
    }
}

impl WithContext for LabOneQError {
    fn add_context<F, C>(&mut self, f: F)
    where
        F: FnOnce() -> C,
        C: Into<Cow<'static, str>>,
    {
        match self {
            LabOneQError::PulseSamplerCallback(e) => e.add_context(f),
            LabOneQError::ResourceExhaustion(e) => e.add_context(f),
            LabOneQError::Generic(e) => e.add_context(f),
        }
    }
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::ContextualError::from_str(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! laboneq_error {
    ($($arg:tt)*) => {
        $crate::LabOneQError::from($crate::ContextualError::from_str(format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return core::result::Result::Err($crate::error!($($arg)*).into());
    };
}

#[macro_export]
macro_rules! bail_resource_usage {
    // Internal helper - terminal case when we hit the usage
    (@collect_args [$fmt:literal $(, $($args:tt)*)?], usage=$usage:expr) => {
        return core::result::Result::Err($crate::resource_usage::ResourceExhaustionError::new(format!($fmt $(, $($args)*)?), $usage).into());
    };

    // Internal helper - collect one more token and recurse
    (@collect_args [$($collected:tt)*] $next:tt $($rest:tt)*) => {
        bail_resource_usage!(@collect_args [$($collected)* $next] $($rest)*)
    };

    // Case 1: String literal and number (using semicolon separator)
    ($msg:literal, usage=$usage:expr) => {
        return core::result::Result::Err($crate::resource_usage::ResourceExhaustionError::new($msg, $usage).into());
    };

    // Case 2: Format specification with arguments and a usage
    // Entry point - delegate to the collecting helper
    ($fmt:literal, $($rest:tt)+) => {
        bail_resource_usage!(@collect_args [$fmt ,] $($rest)+)
    };
}

#[cfg(feature = "pyo3")]
use pyo3::PyTypeInfo;
#[cfg(feature = "pyo3")]
pyo3::import_exception!(laboneq.core.exceptions, LabOneQException);
#[cfg(feature = "pyo3")]
pyo3::import_exception!(
    laboneq.compiler.common.resource_usage,
    ResourceLimitationError
);

/// Conversion of [`LabOneQError`] into a Python exception.
#[cfg(feature = "pyo3")]
impl From<LabOneQError> for pyo3::PyErr {
    fn from(value: LabOneQError) -> pyo3::PyErr {
        value.to_pyerr(|x| x)
    }
}

#[cfg(test)]
mod test {
    use crate::{ContextualError, LabOneQError, WithContext};

    fn error() -> Result<(), std::num::ParseIntError> {
        let _ = "abc".parse::<i32>()?;
        Ok(())
    }

    #[test]
    fn test_laboneq_error_display() {
        let err = ContextualError::from_str("o la la");
        assert_eq!(err.to_string(), "o la la");
    }

    #[test]
    fn test_laboneq_error_with_context_display() {
        fn low_level_op() -> Result<(), ContextualError> {
            crate::bail!("Low-level error");
        }
        fn mid_level_op() -> Result<(), ContextualError> {
            low_level_op().with_context(|| "A mid-level context msg")
        }
        fn high_level_op() -> Result<(), ContextualError> {
            mid_level_op().with_context(|| "The highest-level context msg")?;
            Ok(())
        }

        let error_str = high_level_op().unwrap_err().to_string();
        assert_eq!(
            error_str,
            "\
Low-level error

Context:
    A mid-level context msg
    The highest-level context msg
"
        );
    }

    #[test]
    fn test_laboneq_error_from_error() {
        fn laboneq_code() -> Result<(), ContextualError> {
            error().map_err(ContextualError::from_err)?;
            Ok(())
        }
        let res = laboneq_code();
        assert_eq!(
            res.unwrap_err().to_string(),
            "invalid digit found in string"
        );
    }

    #[test]
    fn test_bail_resource_usage() {
        fn bail_basic() -> Result<(), LabOneQError> {
            bail_resource_usage!("foo bar", usage = 1.3);
        }

        fn bail_formatted_1() -> Result<(), LabOneQError> {
            bail_resource_usage!("{} {}", "foo", "bar", usage = 1.2);
        }

        fn bail_formatted_2() -> Result<(), LabOneQError> {
            bail_resource_usage!("{foo} {bar}", foo = "foo", bar = "bar", usage = 1.1);
        }

        for func in [bail_basic, bail_formatted_1, bail_formatted_2] {
            let res = func();
            assert!(res.is_err());
            assert_eq!(res.unwrap_err().to_string(), "foo bar");
        }
    }
}
