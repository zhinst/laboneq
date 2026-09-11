// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Cap'n Proto `CompiledExperiment` -> Python deserializer.
//!
//! Reading a `CompiledExperiment` is backend-agnostic except for one step: the `artifacts`
//! pointer, which only its producer knows how to decode. [`compiled_experiment_from_capnp`]
//! builds the `ScheduledExperiment` and delegates that step to the backend's
//! [`ArtifactsDeserializer`]; backends own the Python bindings that call it.

mod compiled_experiment;
mod context;
mod execution;
mod result_shape;

use laboneq_error::LabOneQError;

pub use crate::context::{
    DecodedValue, DeserializerContext, decode_constant, decode_numeric_array, decode_value,
};
pub use compiled_experiment::{
    ArtifactsDeserializer, compiled_experiment_from_capnp, peek_backend_name,
};

/// The result of every fallible operation in this crate, and of an [`ArtifactsDeserializer`].
pub type Result<T> = std::result::Result<T, LabOneQError>;
