// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod acquisition;
mod repetition_time;

pub use acquisition::calculate_max_acquisition_time;
pub use repetition_time::{RepetitionInfo, resolve_repetition_time};
