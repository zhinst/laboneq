// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod analyze_awg;
pub(crate) mod handle_acquire;
pub(crate) mod handle_amplitude_registers;
pub(crate) mod handle_frame_changes;
pub(crate) mod handle_hw_phase_resets;
pub(crate) mod handle_loops;
pub(crate) mod handle_match;
pub(crate) mod handle_oscillators;
pub(crate) mod handle_playwaves;
pub(crate) mod handle_ppc_sweeps;
pub(crate) mod handle_precompensation_resets;
pub(crate) mod handle_prng;
mod handle_qa_events;
pub(crate) use handle_qa_events::handle_qa_events;
pub(crate) mod fanout_awg;
mod handle_measure_times;
pub(crate) mod handle_signatures;
pub(crate) mod handle_triggers;
pub(crate) mod lower_for_awg;
pub(crate) use handle_measure_times::analyze_measurements;
