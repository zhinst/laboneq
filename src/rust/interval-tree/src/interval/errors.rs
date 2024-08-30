// Copyright 2019 Johannes Köster, University of Duisburg-Essen.
// SPDX-License-Identifier: MIT

//! Error definitions for the `interval` module.
use thiserror::Error;

use serde::{Deserialize, Serialize};

#[derive(
    Error, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize,
)]
pub enum Error {
    #[error("an Interval must have a Range with a positive width")]
    InvalidRange,
}
pub type Result<T, E = Error> = std::result::Result<T, E>;
