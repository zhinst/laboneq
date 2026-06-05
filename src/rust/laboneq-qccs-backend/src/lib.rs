// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod backend;
mod experiment_view;
mod output_routing;
mod precompensation;
mod preprocessor;
mod setup_processor;

pub mod ports;
pub use backend::QccsBackend;
pub use preprocessor::QccsBackendPreprocessedData;

pub type Result<T, E = laboneq_error::LabOneQError> = std::result::Result<T, E>;
