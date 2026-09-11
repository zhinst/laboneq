// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod artifacts_capnp;
mod backend;
mod device_kind;
mod device_setup_fingerprint;
mod device_traits;
mod experiment_view;
mod oscillator;
mod output_routing;
mod ppc_connections;
mod precompensation;
mod preprocessor;
mod prng;
mod qccs_feedback_calculator;
mod setup_processor;
mod validation;

pub mod setup_description;
pub mod setup_description_capnp;
pub mod setup_description_py;

pub mod ports;
pub use backend::QccsBackend;
pub use preprocessor::QccsBackendPreprocessedData;

pub type Result<T, E = laboneq_error::LabOneQError> = std::result::Result<T, E>;

/// Backend identifier, unique across backends. Tags the setup description this backend
/// compiles against (`DeviceSetup.backend`).
///
/// This identifier is also written to `CompiledExperiment.metadata.createdBy`; the read side
/// dispatches on it to pick this backend's artifacts reader.
pub const BACKEND_NAME: &str = "qccs";
