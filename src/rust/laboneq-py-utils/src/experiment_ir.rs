// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use laboneq_dsl::types::ExternalParameterUid;
use laboneq_ir::ExperimentIr;
use pyo3::prelude::*;

use crate::{pulse::PulseDef, py_object_interner::PyObjectInterner};

/// A Python representation of the Experiment IR.
///
/// This class wraps the [`ExperimentIr`] struct and acts as a bridge between
/// the Rust components called from Python.
#[pyclass(name = "ExperimentIr", frozen)]
pub struct ExperimentIrPy {
    pub inner: ExperimentIr,
    pub py_object_store: Arc<PyObjectInterner<ExternalParameterUid>>,
    pub pulses: Vec<PulseDef>,
}
