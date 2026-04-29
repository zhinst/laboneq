// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod backend;
mod preprocessor;

pub mod ports;
pub use backend::QccsBackend;
pub use preprocessor::QccsBackendPreprocessedData;

pub type Result<T, E = laboneq_error::LabOneQError> = std::result::Result<T, E>;
