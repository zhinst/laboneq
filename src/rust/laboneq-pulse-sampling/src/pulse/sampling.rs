// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use num_complex::Complex64;

use laboneq_error::LabOneQError;

use crate::builtin::Parameters;
use crate::envelope::{self, EnvelopeShape};
use crate::external::{EnvelopeFn, ExternalSampler};
use crate::modulate::{self, Carriers, Peak, SampleCheck};
use crate::samples::{SamplesMut, SamplesRef};

/// Adds every pulse into its target buffer. Pulses not supported by this crate
/// are evaluated by an [`ExternalSampler`].
///
/// Given a `magnitude_limit`, returns the index and peak magnitude of the first
/// pulse whose samples exceed it.
///
/// Panics when a pulse does not fit into its target buffer, or its envelope has
/// fewer samples than its request's slice.
pub fn sample_pulses<P>(
    pulses: &[PulseRequest<'_, P>],
    targets: &mut [SamplesMut<'_>],
    fs: f64,
    external: &dyn ExternalSampler<P>,
    magnitude_limit: Option<f64>,
) -> Result<Option<(usize, f64)>, LabOneQError> {
    let mut exceeding = None;
    let mut carriers = Carriers::new(fs);
    let callback: &mut EnvelopeFn<'_> = &mut |index, envelope| {
        let pulse = &pulses[index];
        let range = pulse.offset..pulse.offset + pulse.slice.len();
        let target = targets[pulse.target].slice_mut(range);
        match magnitude_limit {
            None => add_pulse(pulse, target, envelope, &mut carriers, &mut ()),
            Some(limit) => {
                let mut peak = Peak::default();
                add_pulse(pulse, target, envelope, &mut carriers, &mut peak);
                if peak.exceeds(limit) && exceeding.is_none_or(|(first, _)| index < first) {
                    exceeding = Some((index, peak.magnitude()));
                }
            }
        }
        Ok(())
    };
    envelope::for_each_envelope(pulses, fs, external, callback)?;
    Ok(exceeding)
}

fn add_pulse<P>(
    pulse: &PulseRequest<'_, P>,
    target: SamplesMut<'_>,
    envelope: SamplesRef<'_>,
    carriers: &mut Carriers,
    check: &mut impl SampleCheck,
) {
    let modulation = carriers.modulation(pulse.phase, pulse.modulation_frequency);
    modulate::modulate_envelope(target, envelope, pulse.amplitude, &modulation, check);
}

/// A pulse to add into one of the target buffers.
#[derive(Debug, Clone)]
pub struct PulseRequest<'a, P> {
    pub shape: EnvelopeShape<'a, P>,
    /// Pulse samples to evaluate. Starts at 0 for sampled pulses.
    pub slice: Range<usize>,
    /// Pulse amplitude, including any `amplitude` pulse parameter.
    /// Functional pulses also receive it as an argument.
    pub amplitude: Complex64,
    pub parameters: Option<&'a Parameters>,
    /// Phase offset in radians.
    pub phase: f64,
    /// Software-modulation frequency in Hz, 0 for none.
    pub modulation_frequency: f64,
    /// Index of the target buffer.
    pub target: usize,
    /// Target sample the first pulse sample is added to.
    pub offset: usize,
}
