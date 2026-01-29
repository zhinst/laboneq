// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_units::duration::{Duration, Second};

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum RepetitionMode {
    Fastest,
    Constant { time: Duration<Second, f64> },
    Auto,
}
