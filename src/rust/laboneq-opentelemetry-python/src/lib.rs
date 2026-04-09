// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! This crate provides a function to attach an active OpenTelemetry context from Python to the Rust context.
//! This allows Rust code called from Python to continue a trace started in Python.

mod context_carrier;
pub mod span_buffer;

pub use context_carrier::attach_otel_context;
