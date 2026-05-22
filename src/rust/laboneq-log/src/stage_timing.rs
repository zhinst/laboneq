// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

/// RAII guard that logs the start and elapsed time of a named compilation stage.
///
/// Logs at info level via `laboneq_log::info!` on creation and on drop.
/// Assign to a named binding (not `_`) to keep the guard alive for the full stage.
///
/// The format of the log message is: `"{stage_name} completed. [{elapsed_time:.3} s]"`
pub struct StageTiming {
    name: &'static str,
    start: Instant,
}

impl StageTiming {
    pub fn start(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for StageTiming {
    fn drop(&mut self) {
        log::info!(
            target: concat!("laboneq.rust::", module_path!()),
            "{} completed. [{:.3} s]",
            self.name,
            self.start.elapsed().as_secs_f64()
        );
    }
}
