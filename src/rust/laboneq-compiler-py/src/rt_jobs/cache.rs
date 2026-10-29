// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_dsl::types::{NumericLiteral, ParameterUid};

use super::environment::Environment;

/// The indices of the compiled variants of the real-time program. A job runs a
/// variant when it is in the variant's chunk and holds the same values for the
/// parameters the variant read.
#[derive(Default)]
pub(crate) struct VariantCache {
    /// The variants grouped by the parameters they read.
    groups: Vec<(Vec<ParameterUid>, HashMap<Key, usize>)>,
}

impl VariantCache {
    /// Returns the index of the variant that matches `env` in `chunk`, if one is recorded.
    pub(crate) fn get(&self, env: &Environment, chunk: Option<usize>) -> Option<usize> {
        self.groups.iter().find_map(|(reads, variants)| {
            let key = Key::new(env, reads, chunk)?;
            variants.get(&key).copied()
        })
    }

    /// Records variant `index`, compiled for `env` and `chunk`, as reading the parameters
    /// `reads`. Panics when `env` lacks one of `reads`.
    pub(crate) fn insert(
        &mut self,
        reads: Vec<ParameterUid>,
        env: &Environment,
        chunk: Option<usize>,
        index: usize,
    ) {
        let key = Key::new(env, &reads, chunk)
            .expect("Internal error: A variant read a parameter out of scope");
        match self.groups.iter_mut().find(|(group, _)| *group == reads) {
            Some((_, variants)) => {
                variants.insert(key, index);
            }
            None => self.groups.push((reads, HashMap::from([(key, index)]))),
        }
    }
}

/// The values of a group's parameters in one job, and the job's chunk.
#[derive(PartialEq, Eq, Hash)]
struct Key {
    values: Vec<NumericLiteral>,
    chunk: Option<usize>,
}

impl Key {
    fn new(env: &Environment, reads: &[ParameterUid], chunk: Option<usize>) -> Option<Self> {
        let values = reads
            .iter()
            .map(|uid| env.get(*uid))
            .collect::<Option<_>>()?;
        Some(Self { values, chunk })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SELECTOR: u32 = 1;
    const A: u32 = 2;
    const B: u32 = 3;

    #[test]
    fn variants_that_read_different_parameters_run_by_their_own_reads() {
        let mut cache = VariantCache::default();
        cache.insert(
            uids(&[SELECTOR, A]),
            &env(&[(SELECTOR, 0.0), (A, 0.1), (B, 0.3)]),
            None,
            0,
        );
        cache.insert(
            uids(&[SELECTOR, B]),
            &env(&[(SELECTOR, 1.0), (A, 0.2), (B, 0.4)]),
            None,
            1,
        );

        let get = |selector, a, b| cache.get(&env(&[(SELECTOR, selector), (A, a), (B, b)]), None);
        assert_eq!(get(0.0, 0.1, 0.9), Some(0));
        assert_eq!(get(1.0, 0.9, 0.4), Some(1));
        assert_eq!(get(1.0, 0.1, 0.3), None);
        assert_eq!(get(0.0, 0.2, 0.4), None);
    }

    #[test]
    fn a_variant_runs_in_its_own_chunk_only() {
        let mut cache = VariantCache::default();
        cache.insert(vec![], &env(&[]), Some(0), 0);

        assert_eq!(cache.get(&env(&[]), Some(0)), Some(0));
        assert_eq!(cache.get(&env(&[]), Some(1)), None);
    }

    fn env(values: &[(u32, f64)]) -> Environment {
        values
            .iter()
            .map(|&(id, value)| (id.into(), NumericLiteral::Float(value)))
            .collect()
    }

    fn uids(ids: &[u32]) -> Vec<ParameterUid> {
        ids.iter().copied().map(ParameterUid::from).collect()
    }
}
