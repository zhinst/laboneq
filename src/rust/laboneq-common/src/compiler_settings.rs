// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use laboneq_error::ContextualError;

#[derive(Debug, Clone)]
pub struct CompilerSettings {
    /// Optional event output configuration
    pub output_extras: bool,
    pub max_events_to_publish: usize,
    pub expand_loops_for_schedule: bool,

    /// Device-specific settings
    pub hdawg_min_playwave_hint: u16,
    pub hdawg_min_playzero_hint: u16,
    pub shfsg_min_playwave_hint: u16,
    pub shfsg_min_playzero_hint: u16,
    pub uhfqa_min_playwave_hint: u16,
    pub uhfqa_min_playzero_hint: u16,
    pub use_amplitude_increment: bool,
    pub shf_output_mute_min_duration: f64,
    /// Whether to emit timing comments in the generated code. This can be useful for debugging and performance analysis, but may add overhead to the compilation process.
    pub emit_timing_comments: bool,

    /// Resolution settings for the compiler.
    pub phase_resolution_bits: u64,
    pub amplitude_resolution_bits: u64,

    /// Whether to ignore resource exhaustion errors during compilation. This can be useful for testing and debugging, but should be used with caution in production.
    pub ignore_resource_exhaustion: bool,

    /// Whether the compilation should log detailed information about the output program.
    pub log_report: bool,
}

impl Default for CompilerSettings {
    fn default() -> Self {
        Self {
            output_extras: false,
            max_events_to_publish: 1000,
            expand_loops_for_schedule: true,
            hdawg_min_playwave_hint: 128,
            hdawg_min_playzero_hint: 128,
            shfsg_min_playwave_hint: 64,
            shfsg_min_playzero_hint: 64,
            uhfqa_min_playwave_hint: 64,
            uhfqa_min_playzero_hint: 64,
            use_amplitude_increment: true,
            shf_output_mute_min_duration: 280e-9,
            emit_timing_comments: false,
            amplitude_resolution_bits: 24,
            phase_resolution_bits: 24,
            ignore_resource_exhaustion: false,
            log_report: true,
        }
    }
}

impl CompilerSettings {
    /// Create settings from key-value pairs with validation
    pub fn from_key_value_pairs<I, K, V>(pairs: I) -> Result<Self, ContextualError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        // Define all supported keys
        let supported_keys: HashSet<&str> = [
            "OUTPUT_EXTRAS",
            "MAX_EVENTS_TO_PUBLISH",
            "EXPAND_LOOPS_FOR_SCHEDULE",
            "HDAWG_MIN_PLAYWAVE_HINT",
            "HDAWG_MIN_PLAYZERO_HINT",
            "SHFSG_MIN_PLAYWAVE_HINT",
            "SHFSG_MIN_PLAYZERO_HINT",
            "UHFQA_MIN_PLAYWAVE_HINT",
            "UHFQA_MIN_PLAYZERO_HINT",
            "USE_AMPLITUDE_INCREMENT",
            "SHF_OUTPUT_MUTE_MIN_DURATION",
            "EMIT_TIMING_COMMENTS",
            "PHASE_RESOLUTION_BITS",
            "AMPLITUDE_RESOLUTION_BITS",
            "IGNORE_RESOURCE_LIMITATION_ERRORS",
            "LOG_REPORT",
        ]
        .into_iter()
        .collect();

        let mut settings = Self::default();
        let mut unsupported_keys = Vec::new();

        for (key, value) in pairs {
            let key_str = key.as_ref();
            let value_str = value.as_ref();

            if !supported_keys.contains(key_str) {
                unsupported_keys.push(key_str.to_string());
                continue;
            }

            match key_str {
                "OUTPUT_EXTRAS" => {
                    settings.output_extras =
                        value_str.to_ascii_lowercase().parse().map_err(|_| {
                            ContextualError::from_str(format!(
                                "Invalid boolean for OUTPUT_EXTRAS: {}",
                                value_str
                            ))
                        })?;
                }
                "MAX_EVENTS_TO_PUBLISH" => {
                    settings.max_events_to_publish = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid usize for MAX_EVENTS_TO_PUBLISH: {}",
                            value_str
                        ))
                    })?;
                }
                "EXPAND_LOOPS_FOR_SCHEDULE" => {
                    settings.expand_loops_for_schedule =
                        value_str.to_ascii_lowercase().parse().map_err(|_| {
                            ContextualError::from_str(format!(
                                "Invalid boolean for EXPAND_LOOPS_FOR_SCHEDULE: {}",
                                value_str
                            ))
                        })?;
                }
                "HDAWG_MIN_PLAYWAVE_HINT" => {
                    settings.hdawg_min_playwave_hint = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u16 for HDAWG_MIN_PLAYWAVE_HINT: {}",
                            value_str
                        ))
                    })?;
                }
                "HDAWG_MIN_PLAYZERO_HINT" => {
                    settings.hdawg_min_playzero_hint = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u16 for HDAWG_MIN_PLAYZERO_HINT: {}",
                            value_str
                        ))
                    })?;
                }
                "SHFSG_MIN_PLAYWAVE_HINT" => {
                    settings.shfsg_min_playwave_hint = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u16 for SHFSG_MIN_PLAYWAVE_HINT: {}",
                            value_str
                        ))
                    })?;
                }
                "SHFSG_MIN_PLAYZERO_HINT" => {
                    settings.shfsg_min_playzero_hint = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u16 for SHFSG_MIN_PLAYZERO_HINT: {}",
                            value_str
                        ))
                    })?;
                }
                "UHFQA_MIN_PLAYWAVE_HINT" => {
                    settings.uhfqa_min_playwave_hint = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u16 for UHFQA_MIN_PLAYWAVE_HINT: {}",
                            value_str
                        ))
                    })?;
                }
                "UHFQA_MIN_PLAYZERO_HINT" => {
                    settings.uhfqa_min_playzero_hint = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u16 for UHFQA_MIN_PLAYZERO_HINT: {}",
                            value_str
                        ))
                    })?;
                }
                "USE_AMPLITUDE_INCREMENT" => {
                    settings.use_amplitude_increment =
                        value_str.to_ascii_lowercase().parse().map_err(|_| {
                            ContextualError::from_str(format!(
                                "Invalid boolean for USE_AMPLITUDE_INCREMENT: {}",
                                value_str
                            ))
                        })?;
                }
                "SHF_OUTPUT_MUTE_MIN_DURATION" => {
                    settings.shf_output_mute_min_duration = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid f64 for SHF_OUTPUT_MUTE_MIN_DURATION: {}",
                            value_str
                        ))
                    })?;
                }
                "EMIT_TIMING_COMMENTS" => {
                    settings.emit_timing_comments =
                        value_str.to_ascii_lowercase().parse().map_err(|_| {
                            ContextualError::from_str(format!(
                                "Invalid boolean for EMIT_TIMING_COMMENTS: {}",
                                value_str
                            ))
                        })?;
                }
                "PHASE_RESOLUTION_BITS" => {
                    settings.phase_resolution_bits = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u64 for PHASE_RESOLUTION_BITS: {}",
                            value_str
                        ))
                    })?;
                }
                "AMPLITUDE_RESOLUTION_BITS" => {
                    settings.amplitude_resolution_bits = value_str.parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid u64 for AMPLITUDE_RESOLUTION_BITS: {}",
                            value_str
                        ))
                    })?;
                }
                "IGNORE_RESOURCE_LIMITATION_ERRORS" => {
                    settings.ignore_resource_exhaustion =
                        value_str.to_ascii_lowercase().parse().map_err(|_| {
                            ContextualError::from_str(format!(
                                "Invalid boolean for IGNORE_RESOURCE_LIMITATION_ERRORS: {}",
                                value_str
                            ))
                        })?;
                }
                "LOG_REPORT" => {
                    settings.log_report = value_str.to_ascii_lowercase().parse().map_err(|_| {
                        ContextualError::from_str(format!(
                            "Invalid boolean for LOG_REPORT: {}",
                            value_str
                        ))
                    })?;
                }
                _ => unreachable!("Key should have been validated above"),
            }
        }

        if !unsupported_keys.is_empty() {
            let mut sorted_supported: Vec<_> = supported_keys.into_iter().collect();
            sorted_supported.sort();
            return Err(ContextualError::from_str(format!(
                "Unsupported compiler settings keys: {}. Supported keys are: {}",
                unsupported_keys.join(", "),
                sorted_supported.join(", ")
            )));
        }

        Ok(settings)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output_extras: bool,
        max_events_to_publish: usize,
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
        ignore_resource_exhaustion: bool,
        log_report: bool,
        expand_loops_for_schedule: bool,
    ) -> Self {
        Self {
            output_extras,
            max_events_to_publish,
            expand_loops_for_schedule,
            hdawg_min_playwave_hint,
            hdawg_min_playzero_hint,
            shfsg_min_playwave_hint,
            shfsg_min_playzero_hint,
            uhfqa_min_playwave_hint,
            uhfqa_min_playzero_hint,
            use_amplitude_increment,
            shf_output_mute_min_duration,
            emit_timing_comments,
            amplitude_resolution_bits,
            phase_resolution_bits,
            ignore_resource_exhaustion,
            log_report,
        }
    }
}
