// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Module for defining settings for the code generator.
use std::vec;

use crate::ir::compilation_job::DeviceKind;
use crate::tinysample::ceil_to_grid;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct SanitizationChange {
    pub field: &'static str,
    pub original: String,
    pub sanitized: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct CodeGeneratorSettings {
    hdawg_min_playwave_hint: u16,
    hdawg_min_playzero_hint: u16,
    shfsg_min_playwave_hint: u16,
    shfsg_min_playzero_hint: u16,
    uhfqa_min_playwave_hint: u16,
    uhfqa_min_playzero_hint: u16,
    amplitude_resolution_bits: u64,
    phase_resolution_bits: u64,
    use_amplitude_increment: bool,
    pub emit_timing_comments: bool,
    pub shf_output_mute_min_duration: f64,
}

impl CodeGeneratorSettings {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hdawg_min_playwave_hint: u16,
        hdawg_min_playzero_hint: u16,
        shfsg_min_playwave_hint: u16,
        shfsg_min_playzero_hint: u16,
        uhfqa_min_playwave_hint: u16,
        uhfqa_min_playzero_hint: u16,
        amplitude_resolution_bits: u64,
        phase_resolution_bits: u64,
        use_amplitude_increment: bool,
        emit_timing_comments: bool,
        shf_output_mute_min_duration: f64,
    ) -> Self {
        CodeGeneratorSettings {
            hdawg_min_playwave_hint,
            hdawg_min_playzero_hint,
            shfsg_min_playwave_hint,
            shfsg_min_playzero_hint,
            uhfqa_min_playwave_hint,
            uhfqa_min_playzero_hint,
            amplitude_resolution_bits,
            phase_resolution_bits,
            use_amplitude_increment,
            emit_timing_comments,
            shf_output_mute_min_duration,
        }
    }

    pub(crate) fn amplitude_resolution_range(&self) -> u64 {
        if self.amplitude_resolution_bits > 0 {
            1 << self.amplitude_resolution_bits
        } else {
            0
        }
    }

    pub(crate) fn phase_resolution_range(&self) -> u64 {
        if self.phase_resolution_bits > 0 {
            1 << self.phase_resolution_bits
        } else {
            0
        }
    }

    pub fn use_amplitude_increment(&self) -> bool {
        self.use_amplitude_increment
    }

    pub fn waveform_size_hints(&self, device: &DeviceKind) -> (u16, u16) {
        let (min_pw, min_pz) = match device {
            DeviceKind::HDAWG => (self.hdawg_min_playwave_hint, self.hdawg_min_playzero_hint),
            DeviceKind::SHFSG => (self.shfsg_min_playwave_hint, self.shfsg_min_playzero_hint),
            DeviceKind::UHFQA => (self.uhfqa_min_playwave_hint, self.uhfqa_min_playzero_hint),
            // On SHFQA there is no reason to have hints as there can be only one waveform per signal.
            // Use the lowest hints possible to ensure no extra padding is added to the waveforms to ensure
            // that a single waveform is created.
            DeviceKind::SHFQA => (
                device.traits().sample_multiple,
                device.traits().min_play_wave as u16,
            ),
        };
        (min_pw, min_pz)
    }

    pub fn sanitize(&mut self) -> Result<Vec<SanitizationChange>> {
        let mut changes = vec![];
        let min_pw_fields = [
            (
                "hdawg_min_playwave_hint",
                &mut self.hdawg_min_playwave_hint,
                DeviceKind::HDAWG.traits().sample_multiple,
            ),
            (
                "shfsg_min_playwave_hint",
                &mut self.shfsg_min_playwave_hint,
                DeviceKind::SHFSG.traits().sample_multiple,
            ),
            (
                "uhfqa_min_playwave_hint",
                &mut self.uhfqa_min_playwave_hint,
                DeviceKind::UHFQA.traits().sample_multiple,
            ),
        ];
        for (field, value, sample_multiple) in min_pw_fields {
            let sanitized_value = sanitize_min_playwave_hint(*value, sample_multiple)?;
            if sanitized_value != *value {
                changes.push(SanitizationChange {
                    field,
                    original: value.to_string(),
                    sanitized: sanitized_value.to_string(),
                    reason: format!("Not a multiple of {sample_multiple}."),
                });
                *value = sanitized_value;
            }
        }
        Ok(changes)
    }
}

fn sanitize_min_playwave_hint(value: u16, sample_multiple: u16) -> Result<u16> {
    if !value.is_multiple_of(sample_multiple) {
        return ceil_to_grid(value.into(), sample_multiple.into())
            .try_into()
            .map_err(|_| {
                Error::new("Expected `MIN_PLAY_WAVE_HINT` to fit into 16 bit unsigned int")
            });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_min_playwave_hint() {
        assert_eq!(sanitize_min_playwave_hint(0, 16).unwrap(), 0);
        assert_eq!(sanitize_min_playwave_hint(10, 1).unwrap(), 10);
        assert_eq!(sanitize_min_playwave_hint(10, 2).unwrap(), 10);
        assert_eq!(sanitize_min_playwave_hint(11, 2).unwrap(), 12);
        assert_eq!(sanitize_min_playwave_hint(15, 4).unwrap(), 16);
        assert!(sanitize_min_playwave_hint(65535, 1000).is_err());
    }

    #[test]
    fn test_sanization_change() {
        let offset: u16 = 1;
        let min_play_wave: u16 = DeviceKind::HDAWG.traits().min_play_wave.try_into().unwrap();
        let pw_original: u16 = min_play_wave + offset;
        let mut settings =
            CodeGeneratorSettings::new(pw_original, 0, 0, 0, 0, 0, 16, 16, true, true, 0.0);
        let changes = settings.sanitize().unwrap();
        assert_eq!(
            settings.hdawg_min_playwave_hint,
            pw_original - offset + DeviceKind::HDAWG.traits().sample_multiple
        );
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].field, "hdawg_min_playwave_hint");
        assert_eq!(changes[0].original, pw_original.to_string());
        assert_eq!(
            changes[0].sanitized,
            settings.hdawg_min_playwave_hint.to_string()
        );
    }
}
