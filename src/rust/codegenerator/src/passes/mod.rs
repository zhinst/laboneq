// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub mod analyze_awg;
pub mod handle_acquire;
pub mod handle_amplitude_registers;
pub mod handle_frame_changes;
pub mod handle_hw_phase_resets;
pub mod handle_loops;
pub mod handle_match;
pub mod handle_oscillators;
pub mod handle_playwaves;
pub mod handle_ppc_sweeps;
pub mod handle_precompensation_resets;
pub mod handle_prng;
mod handle_qa_events;
pub use handle_qa_events::handle_qa_events;
pub mod handle_signatures;
pub mod handle_triggers;
pub mod lower_for_awg;
pub use analyze_awg::{AwgCompilationInfo, analyze_awg_ir};
pub mod fanout_awg;
mod handle_measure_times;
pub use handle_measure_times::analyze_measurements;
