// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

/// Identifies a point in a near-time sweep by the loop indices leading to it.
///
/// Shared between compilation (linking per-step code generator output) and the
/// controller/runtime (matching execution state and results back to the step
/// that produced them).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NtStepKey(pub Vec<usize>);
