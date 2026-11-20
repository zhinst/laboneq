// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use crate::uid::ParameterUid;

/// Identifies a point in a near-time sweep by the loop indices leading to it.
///
/// Shared between compilation (linking per-step code generator output) and the
/// controller/runtime (matching execution state and results back to the step
/// that produced them).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NtStepKey(pub Vec<usize>);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FixedValueOrParameter<T> {
    Value(T),
    Parameter(ParameterUid),
}

impl<T: fmt::Display> fmt::Display for FixedValueOrParameter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value) => value.fmt(f),
            Self::Parameter(uid) => write!(f, "parameter {}", uid.0),
        }
    }
}
