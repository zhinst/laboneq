// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, PartialEq, Eq, Copy, Default)]
pub enum SectionTimingMode {
    #[default]
    Relaxed,
    Strict,
}
