// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

//! Compiler settings: the accepted keys and the values the caller passed.
//!
//! Whoever reads a setting declares it as a [`SettingSpec`], the frontend for
//! the settings it reads itself, each backend for its own. One validation pass
//! checks the caller's keys and values against all declarations.

use std::collections::HashMap;

use laboneq_error::{ContextualError, bail, error};

/// The type a setting takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingKind {
    Bool,
    Int,
    Float,
    Str,
}

impl SettingKind {
    fn name(&self) -> &'static str {
        match self {
            SettingKind::Bool => "boolean",
            SettingKind::Int => "integer",
            SettingKind::Float => "float",
            SettingKind::Str => "string",
        }
    }
}

/// A setting's value.
#[derive(Debug, Clone, PartialEq)]
pub enum SettingValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl std::fmt::Display for SettingValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SettingValue::Bool(value) => write!(f, "{value}"),
            SettingValue::Int(value) => write!(f, "{value}"),
            SettingValue::Float(value) => write!(f, "{value}"),
            SettingValue::Str(value) => write!(f, "{value:?}"),
        }
    }
}

impl SettingValue {
    fn kind(&self) -> SettingKind {
        match self {
            SettingValue::Bool(_) => SettingKind::Bool,
            SettingValue::Int(_) => SettingKind::Int,
            SettingValue::Float(_) => SettingKind::Float,
            SettingValue::Str(_) => SettingKind::Str,
        }
    }

    /// Convert to `kind`, or `None` if the value cannot be that kind.
    ///
    /// An integer converts to a float, so that `0` may be written for a duration.
    /// Nothing else converts, not even a boolean to an integer.
    fn coerce(self, kind: SettingKind) -> Option<Self> {
        match (self, kind) {
            (SettingValue::Int(i), SettingKind::Float) => Some(SettingValue::Float(i as f64)),
            (value, kind) if value.kind() == kind => Some(value),
            _ => None,
        }
    }
}

/// One setting, as declared by whoever reads it.
#[derive(Debug, Clone)]
pub struct SettingSpec {
    pub key: &'static str,
    pub kind: SettingKind,
    /// The value that applies when the caller passes none. `None` leaves the
    /// setting absent instead; its reader decides what absence means.
    pub default: Option<SettingValue>,
    pub doc: &'static str,
}

/// The values of the declared settings, validated against their specs.
///
/// Every declared setting with a default is present, so the readers below need
/// no fallback. A setting declared without one is absent until the caller
/// passes it, and is read through the `opt_*` readers.
#[derive(Debug, Clone)]
pub struct Settings {
    values: HashMap<&'static str, SettingValue>,
}

impl Settings {
    /// Validate `pairs` against `specs`, and fill in the declared defaults.
    ///
    /// # Errors
    /// If a key is in no spec, or its value cannot be the declared type, or two
    /// specs declare the same key.
    pub fn parse<I>(specs: &[&[SettingSpec]], pairs: I) -> Result<Self, ContextualError>
    where
        I: IntoIterator<Item = (String, SettingValue)>,
    {
        let mut declared: HashMap<&str, &SettingSpec> = HashMap::new();
        for spec in specs.iter().flat_map(|specs| specs.iter()) {
            if declared.insert(spec.key, spec).is_some() {
                bail!("Compiler setting declared twice: {}", spec.key);
            }
        }

        let mut values = HashMap::new();
        let mut unsupported = Vec::new();
        for (key, value) in pairs {
            let Some(spec) = declared.get(key.as_str()) else {
                unsupported.push(key);
                continue;
            };
            let value = value.clone().coerce(spec.kind).ok_or_else(|| {
                error!(
                    "Compiler setting {} expects a {}, got a {}: {}",
                    spec.key,
                    spec.kind.name(),
                    value.kind().name(),
                    value
                )
            })?;
            values.insert(spec.key, value);
        }

        if !unsupported.is_empty() {
            let mut supported: Vec<_> = declared.keys().collect();
            supported.sort();
            bail!(
                "Unsupported compiler settings keys: {}. Supported keys are: {}",
                unsupported.join(", "),
                supported
                    .iter()
                    .map(|key| key.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        for spec in declared.values() {
            if let Some(default) = &spec.default {
                values.entry(spec.key).or_insert_with(|| default.clone());
            }
        }
        Ok(Settings { values })
    }

    /// The declared defaults, with nothing passed in.
    pub fn defaults(specs: &[&[SettingSpec]]) -> Self {
        Settings::parse(specs, [])
            .expect("the declared settings should be consistent with each other")
    }

    pub fn opt_bool(&self, key: &str) -> Option<bool> {
        match self.values.get(key) {
            Some(SettingValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn opt_int(&self, key: &str) -> Option<i64> {
        match self.values.get(key) {
            Some(SettingValue::Int(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn opt_float(&self, key: &str) -> Option<f64> {
        match self.values.get(key) {
            Some(SettingValue::Float(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn opt_str(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(SettingValue::Str(value)) => Some(value),
            _ => None,
        }
    }

    /// # Panics
    /// If `key` was not declared with a boolean default. Reading a setting one
    /// did not declare is a bug in the reader, not something a caller can
    /// provoke.
    pub fn bool(&self, key: &str) -> bool {
        self.opt_bool(key).unwrap_or_else(|| undeclared(key))
    }

    /// # Panics
    /// As [`Settings::bool`].
    pub fn int(&self, key: &str) -> i64 {
        self.opt_int(key).unwrap_or_else(|| undeclared(key))
    }

    /// # Panics
    /// As [`Settings::bool`].
    pub fn float(&self, key: &str) -> f64 {
        self.opt_float(key).unwrap_or_else(|| undeclared(key))
    }

    /// # Panics
    /// As [`Settings::bool`].
    pub fn str(&self, key: &str) -> &str {
        self.opt_str(key).unwrap_or_else(|| undeclared(key))
    }
}

fn undeclared(key: &str) -> ! {
    panic!("Compiler setting {key} was not declared with a default of this type")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &[SettingSpec] = &[
        SettingSpec {
            key: "A_FLAG",
            kind: SettingKind::Bool,
            default: Some(SettingValue::Bool(false)),
            doc: "",
        },
        SettingSpec {
            key: "A_DURATION",
            kind: SettingKind::Float,
            default: Some(SettingValue::Float(1e-9)),
            doc: "",
        },
        SettingSpec {
            key: "A_LATENCY",
            kind: SettingKind::Float,
            default: None,
            doc: "",
        },
    ];

    fn parse(pairs: Vec<(&str, SettingValue)>) -> Result<Settings, ContextualError> {
        Settings::parse(
            &[SPEC],
            pairs
                .into_iter()
                .map(|(key, value)| (key.to_string(), value)),
        )
    }

    #[test]
    fn declared_defaults_are_in_force() {
        let settings = parse(vec![]).unwrap();
        assert!(!settings.bool("A_FLAG"));
        assert_eq!(settings.float("A_DURATION"), 1e-9);
        assert_eq!(settings.opt_float("A_LATENCY"), None);
    }

    #[test]
    fn a_passed_value_wins() {
        let settings = parse(vec![("A_FLAG", SettingValue::Bool(true))]).unwrap();
        assert!(settings.bool("A_FLAG"));
    }

    #[test]
    fn an_integer_stands_in_for_a_float() {
        let settings = parse(vec![("A_DURATION", SettingValue::Int(0))]).unwrap();
        assert_eq!(settings.float("A_DURATION"), 0.0);
    }

    #[test]
    fn an_undeclared_key_lists_the_declared_ones() {
        let error = parse(vec![("NO_SUCH_KEY", SettingValue::Bool(true))]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("NO_SUCH_KEY"), "{message}");
        assert!(
            message.contains("A_DURATION, A_FLAG, A_LATENCY"),
            "{message}"
        );
    }

    #[test]
    fn a_value_of_the_wrong_type_names_the_setting_and_the_type() {
        let error = parse(vec![("A_FLAG", SettingValue::Str("yes".into()))]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("A_FLAG"), "{message}");
        assert!(message.contains("boolean"), "{message}");
    }

    #[test]
    fn one_key_cannot_be_declared_twice() {
        let error = Settings::parse(&[SPEC, SPEC], []).unwrap_err();
        assert!(error.to_string().contains("declared twice"));
    }
}
