// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use num_complex::Complex;
use std::f64::consts::PI;
use std::sync::Arc;

use crate::ir;
use crate::ir::compilation_job as cjob;
use crate::utils::normalize_phase;

#[derive(Debug, Clone)]
pub struct PulseSignature {
    pub start: i64,
    pub pulse: Option<Arc<cjob::PulseDef>>,
    pub length: i64,
    pub amplitude: Option<Complex<f64>>,
    pub phase: f64,
    pub oscillator_phase: Option<f64>,
    pub oscillator_frequency: Option<f64>,
    pub increment_oscillator_phase: Option<f64>,
    pub channel: Option<u16>,
    pub sub_channel: Option<u8>,
    pub id_pulse_params: Option<usize>,
    pub markers: Vec<cjob::Marker>,
    // TODO: Perhaps this could be just 0?
    pub preferred_amplitude_register: Option<u16>,
    pub incr_phase_params: Vec<String>,
}

impl PulseSignature {
    pub fn end(&self) -> i64 {
        self.start + self.length
    }
}

#[derive(Debug, Clone)]
pub struct Waveform {
    pub length: ir::Samples,
    pub pulses: Vec<PulseSignature>,
}

impl Waveform {
    pub fn sort_pulses(&mut self) {
        self.pulses
            .sort_by(|a, b| (a.start, a.channel).cmp(&(b.start, b.channel)));
    }
}

/// Quantize amplitude baked in pulses for the given precision.
pub fn quantize_amplitude_pulse(value: f64, amplitude_resolution_range: u64) -> f64 {
    (value * amplitude_resolution_range as f64).round() / amplitude_resolution_range as f64
}

/// Quantize the command table amplitude.
///
/// For the amplitude specified by registers on the device (e.g. command table)
/// we quantize to a fixed precision of 18 bits. This
/// serves to avoid rounding errors leading to multiple command table entries
pub fn quantize_amplitude_ct(value: f64) -> f64 {
    static AMPLITUDE_RESOLUTION_CT: u32 = 1_u32 << 18;
    (value * AMPLITUDE_RESOLUTION_CT as f64).round() / AMPLITUDE_RESOLUTION_CT as f64
}

/// Quantize phase baked in pulses for the given precision.
pub fn quantize_phase_pulse(value: f64, phase_resolution_range: u64) -> f64 {
    let phase_resolution_range = phase_resolution_range as f64 / (2.0 * PI);
    let phase = (value * phase_resolution_range).round() / phase_resolution_range;
    normalize_phase(phase)
}

/// Quantize the command table phase.
///
/// For the phase specified by registers on the device (e.g. command table)
/// we quantize to a fixed precision of 24 bits. This
/// serves to avoid rounding errors leading to multiple command table entries
pub fn quantize_phase_ct(value: f64) -> f64 {
    static PHASE_RESOLUTION_CT: f64 = (1 << 24) as f64 / (2.0 * PI);
    let phase = (value * PHASE_RESOLUTION_CT).round() / PHASE_RESOLUTION_CT;
    normalize_phase(phase)
}
