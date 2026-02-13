// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Error handling for the Python bindings.
//! This module provides functionality to translate Rust errors
//! into Python exceptions.

use codegenerator::Error as CodeGeneratorError;
use laboneq_common::named_id::NamedIdStore;
use laboneq_common::named_id::resolve_ids;
use pyo3::import_exception;
use pyo3::prelude::*;

import_exception!(laboneq.core.exceptions, LabOneQException);

pub(crate) type Result<T> = std::result::Result<T, CodeGeneratorError>;

/// Formatter for Python error.
///
/// The function is used to wrap Rust errors and convert them into Python exceptions.
/// If the root error is a Python exception, it will be extracted and set as the cause of the `LabOneQException`,
/// to enable full traceback in Python.
///
/// The messages in the error chain is converted into a string message that is included in the Python exception as
/// an additional context and all of the UIDs are resolved using the provided [`NamedIdStore`].
pub(crate) fn error_to_pyerr(error: CodeGeneratorError, id_store: &NamedIdStore) -> PyErr {
    let err_message = create_python_error_message(&error);
    let err_message = resolve_ids(&err_message, id_store);
    if let Some(py_err) = find_python_root_cause(&error) {
        let error = LabOneQException::new_err(err_message);
        Python::attach(|py| {
            error.set_cause(py, Some(py_err.clone_ref(py)));
            error
        })
    } else {
        LabOneQException::new_err(err_message)
    }
}

/// Collect the source errors and format them into a string.
///
/// The original error message is excluded from the context,
fn create_context_message(error: &anyhow::Error) -> Option<String> {
    let mut causes = error
        .chain()
        .skip(1)
        .map(|cause| format!("{cause}"))
        .collect::<Vec<_>>();
    if causes.is_empty() {
        return None;
    }
    // Reverse to show the most last cause first
    causes.reverse();
    let msg = format!("Caused by:\n  {:}", causes.join("\n  "));
    Some(msg)
}

fn get_anyhow_error(error: &CodeGeneratorError) -> &anyhow::Error {
    match error {
        CodeGeneratorError::Anyhow(e) => e,
    }
}

/// Format the error message for Python exceptions.
fn create_python_error_message(error: &CodeGeneratorError) -> String {
    if let Some(error_context) = create_context_message(get_anyhow_error(error)) {
        return format!("{error}\n{error_context}");
    }
    format!("{error}")
}

fn find_python_root_cause(error: &CodeGeneratorError) -> Option<&PyErr> {
    let err = get_anyhow_error(error).root_cause();
    if let Some(py_err) = err.downcast_ref::<PyErr>() {
        return Some(py_err);
    }
    if let Some(codegen_err) = err.downcast_ref::<CodeGeneratorError>() {
        match codegen_err {
            CodeGeneratorError::Anyhow(e) => {
                return e.root_cause().downcast_ref::<PyErr>();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn test_python_error_message() {
        // Test error without context
        let msg = create_python_error_message(&CodeGeneratorError::new("root"));
        let expected = "root";
        assert_eq!(msg, expected);

        // Test error with context
        let res = Err::<(), _>(CodeGeneratorError::new("root"));
        let res: Result<_> = res.context("mid").map_err(Into::into);
        let res: Result<_> = res.context("top").map_err(Into::into);
        let res: Result<_> = res.context("The most top level").map_err(Into::into);
        let err: CodeGeneratorError = res.unwrap_err();
        let msg = create_python_error_message(&err);
        let expected = "\
The most top level
Caused by:
  root
  mid
  top\
";
        assert_eq!(msg, expected);
    }
}
