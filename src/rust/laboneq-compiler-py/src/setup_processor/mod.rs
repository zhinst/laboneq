// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Turning a backend's setup description into the signals the compiler works
//! with.

mod delays;
mod processor;

pub(crate) use crate::setup_processor::delays::DelayRegistry;
pub(crate) use processor::process_setup;
