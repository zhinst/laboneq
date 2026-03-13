// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

mod delays;
mod precompensation;
mod processor;

pub(crate) use crate::setup_processor::delays::DelayRegistry;
pub(crate) use processor::process_setup;
