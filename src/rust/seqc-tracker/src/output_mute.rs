// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use crate::{Result, Samples};
use anyhow::anyhow;
use codegenerator::{ir::compilation_job::DeviceKind, tinysample::length_to_samples};
use std::str::FromStr;

fn ceil(value: i64, grid: u16) -> Samples {
    assert!(value > 0, "Value must be greater than 0");
    let grid = grid as i64;
    // In Rust, the modulo operator works differently for negative numbers
    // compared to Python. In Python, -1 % 3 is 2, while in Rust it is -1.
    // Python code: value + (-value) % grid
    let mut vg = -value % grid;
    if vg < 0 {
        vg += grid;
    }
    (value + vg) as Samples
}

/// Mute played samples.
///
/// # Arguments:
///
/// * device_type: Device type
///   Given device must support output mute.
/// * duration_min: Minimum duration for mute to be fully engaged.
///   Therefore the minimum mute duration is: engage time + `duration_min` + disengage time
///
pub struct OutputMute {
    pub samples_min: Samples,
    pub delay_engage: Samples,
    pub delay_disengage: Samples,
}

impl OutputMute {
    pub fn new(device_type: &str, duration_min: f64) -> Result<Self> {
        let device_kind = DeviceKind::from_str(device_type)?;

        let device_traits = device_kind.traits();
        if !device_traits.supports_output_mute {
            return Err(anyhow!(
                "Unsupported device type: {0}. Supported types are: SHFQA, SHFSG and SHFQC.",
                device_type.to_uppercase()
            )
            .into());
        }

        // The minimum time for the muting required by the instrument
        let device_duration_min = device_traits.output_mute_engage_delay
            - device_traits.output_mute_disengage_delay  // latency of just turning on and off the blanking...
            + device_traits.min_play_wave as f64 / device_traits.sampling_rate; // ... plus a minimal playZero
        if duration_min <= device_duration_min {
            return Err(anyhow!(
                "Output mute duration must be larger than {} s.",
                device_duration_min
            )
            .into());
        }
        let samples_min = length_to_samples(duration_min, device_traits.sampling_rate);
        let samples_min = ((samples_min as f64 / device_traits.sample_multiple as f64).ceil()
            * device_traits.sample_multiple as f64) as Samples;
        let delay_engage = length_to_samples(
            device_traits.output_mute_engage_delay,
            device_traits.sampling_rate,
        );
        let delay_engage = ceil(delay_engage, device_traits.sample_multiple) as Samples;
        let delay_disengage = length_to_samples(
            -device_traits.output_mute_disengage_delay,
            device_traits.sampling_rate,
        );
        let delay_disengage = ceil(delay_disengage, device_traits.sample_multiple);

        Ok(OutputMute {
            samples_min,
            delay_engage,
            delay_disengage,
        })
    }
}
