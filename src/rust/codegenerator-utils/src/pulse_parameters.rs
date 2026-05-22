// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use laboneq_dsl::operation::ExternalOrValue;
use laboneq_dsl::types::{PulseParameterUid, ValueOrParameter};

pub type PulseParametersId = u64;

pub fn hash_pulse_parameter_value(value: &ExternalOrValue) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match value {
        ExternalOrValue::ExternalParameter(v) => v.hash(&mut hasher),
        ExternalOrValue::ValueOrParameter(c) => match c {
            ValueOrParameter::Value(v) => {
                v.hash(&mut hasher);
            }
            ValueOrParameter::ResolvedParameter { value, .. } => {
                value.hash(&mut hasher);
            }
            ValueOrParameter::Parameter(uid) => uid.hash(&mut hasher),
        },
    }
    hasher.finish()
}

pub fn hash_hashmap_with_pulse_parameter_values(
    map: &HashMap<PulseParameterUid, ExternalOrValue>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Sort by key to ensure deterministic hashing regardless of HashMap iteration order
    let mut sorted_entries: Vec<_> = map.iter().collect();
    sorted_entries.sort_by_key(|a| a.0.0);

    for (key, value) in sorted_entries {
        key.hash(&mut hasher);
        hash_pulse_parameter_value(value).hash(&mut hasher);
    }

    hasher.finish()
}

pub struct PulseParameters {
    pub id: PulseParametersId,
    pub pulse_parameters: HashMap<PulseParameterUid, ExternalOrValue>,
    pub play_parameters: HashMap<PulseParameterUid, ExternalOrValue>,
    pub parameters: HashMap<PulseParameterUid, ExternalOrValue>,
}

/// Deduplicated for `PulseParameters` instances.
///
/// This struct allows to intern `PulseParameters` based on their content,
/// returning a unique `PulseParametersId` for each unique set of parameters.
#[derive(Default)]
pub struct PulseParameterDeduplicator {
    seen: HashMap<u64, PulseParameters>,
}

impl PulseParameterDeduplicator {
    pub fn new() -> Self {
        PulseParameterDeduplicator {
            seen: HashMap::new(),
        }
    }

    pub fn intern(
        &mut self,
        pulse_parameters: &HashMap<PulseParameterUid, ExternalOrValue>,
        play_parameters: &HashMap<PulseParameterUid, ExternalOrValue>,
    ) -> PulseParametersId {
        // Merge pulse_parameters and play_parameters, where play_parameters override pulse_parameters
        let mut merged = pulse_parameters.clone();
        merged.extend(play_parameters.clone());
        // Compute hash of the merged parameters
        let id = hash_hashmap_with_pulse_parameter_values(&merged);
        // Register if not seen before and check for hash collision
        if let Some(existing) = self.seen.get(&id) {
            assert_eq!(
                &existing.pulse_parameters, pulse_parameters,
                "Internal error: Hash collision detected for pulse parameters."
            );
        } else {
            let params = PulseParameters {
                id,
                pulse_parameters: pulse_parameters.clone(),
                play_parameters: play_parameters.clone(),
                parameters: merged,
            };
            self.seen.insert(id, params);
        }
        id
    }

    pub fn resolve(&self, id: &PulseParametersId) -> Option<&PulseParameters> {
        self.seen.get(id)
    }

    pub fn all_parameters(&self) -> impl Iterator<Item = &PulseParameters> {
        self.seen.values()
    }
}
