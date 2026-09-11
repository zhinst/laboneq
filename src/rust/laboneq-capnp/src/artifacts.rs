// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic interface for writing `CompiledExperiment.artifacts`.
//!
//! `artifacts` is an `AnyPointer`: its concrete schema is chosen by whichever
//! backend produced it, identified by the backend half of `CompiledExperiment.metadata.createdBy`.
//! Backends implement [`ArtifactsWriter`] so the core compiler can fill that pointer
//! without knowing the concrete artifacts type.

use capnp::any_pointer;

/// Writes a backend-specific codegen-artifacts struct into an `AnyPointer` field.
pub trait ArtifactsWriter {
    /// Initialize `builder` with this backend's artifacts struct.
    fn write(&self, builder: any_pointer::Builder<'_>) -> capnp::Result<()>;
}
