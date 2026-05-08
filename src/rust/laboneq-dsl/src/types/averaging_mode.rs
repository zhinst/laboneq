// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Default, Clone, PartialEq, Copy, Eq)]
pub enum AveragingMode {
    #[default]
    Cyclic,
    Sequential,
    SingleShot,
}
