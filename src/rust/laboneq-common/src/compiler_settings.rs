// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_error::ContextualError;

use crate::settings::{SettingKind, SettingSpec, SettingValue, Settings};

/// Setting keys the frontend reads. A backend declares the rest.
pub const FRONTEND_SETTINGS: &[SettingSpec] = &[
    SettingSpec {
        key: "OUTPUT_EXTRAS",
        kind: SettingKind::Bool,
        default: Some(SettingValue::Bool(false)),
        doc: "Publish the event list and the schedule alongside the compiled experiment.",
    },
    SettingSpec {
        key: "MAX_EVENTS_TO_PUBLISH",
        kind: SettingKind::Int,
        default: Some(SettingValue::Int(1000)),
        doc: "Upper bound on the number of published events.",
    },
    SettingSpec {
        key: "EXPAND_LOOPS_FOR_SCHEDULE",
        kind: SettingKind::Bool,
        default: Some(SettingValue::Bool(true)),
        doc: "Unroll loops in the published schedule, so that every iteration appears.",
    },
    SettingSpec {
        key: "IGNORE_RESOURCE_LIMITATION_ERRORS",
        kind: SettingKind::Bool,
        default: Some(SettingValue::Bool(false)),
        doc: "Continue past an exhausted device resource instead of failing the compile.",
    },
    SettingSpec {
        key: "LOG_REPORT",
        kind: SettingKind::Bool,
        default: Some(SettingValue::Bool(true)),
        doc: "Log a report on the compiled program.",
    },
];

#[derive(Debug, Clone)]
pub struct CompilerSettings {
    /// Optional event output configuration
    pub output_extras: bool,
    pub max_events_to_publish: usize,
    pub expand_loops_for_schedule: bool,

    /// Whether to ignore resource exhaustion errors during compilation. This can be useful for testing and debugging, but should be used with caution in production.
    pub ignore_resource_exhaustion: bool,

    /// Whether the compilation should log detailed information about the output program.
    pub log_report: bool,

    /// The values of the settings a backend declared. The frontend passes them
    /// through without interpreting them.
    pub backend: Settings,
}

impl CompilerSettings {
    /// Validate `pairs` against the frontend's settings and the backend's.
    ///
    /// # Errors
    /// If a key belongs to neither, or a value is not of the declared type.
    pub fn parse<I>(backend_settings: &[SettingSpec], pairs: I) -> Result<Self, ContextualError>
    where
        I: IntoIterator<Item = (String, SettingValue)>,
    {
        let values = Settings::parse(&[FRONTEND_SETTINGS, backend_settings], pairs)?;
        Ok(CompilerSettings {
            output_extras: values.bool("OUTPUT_EXTRAS"),
            max_events_to_publish: values.int("MAX_EVENTS_TO_PUBLISH") as usize,
            expand_loops_for_schedule: values.bool("EXPAND_LOOPS_FOR_SCHEDULE"),
            ignore_resource_exhaustion: values.bool("IGNORE_RESOURCE_LIMITATION_ERRORS"),
            log_report: values.bool("LOG_REPORT"),
            backend: values,
        })
    }

    /// The settings used by the compiler when the caller passes none.
    pub fn defaults(backend_settings: &[SettingSpec]) -> Self {
        CompilerSettings::parse(backend_settings, [])
            .expect("the declared settings should be consistent with each other")
    }
}
