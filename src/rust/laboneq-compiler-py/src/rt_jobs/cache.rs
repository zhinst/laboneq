// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::hash::Hash;

use laboneq_dsl::types::{NumericLiteral, ParameterUid};

use super::environment::Environment;

/// The indices of the compiled variants of the real-time program, by what each depends on.
///
/// A variant depends on its program key, its chunk, and the values of the parameters that
/// compiling it read. A job runs a variant when it agrees with it on all three.
pub(crate) struct VariantCache<ProgramKey> {
    /// The index of each variant, by what it depends on.
    variants: HashMap<VariantKey<ProgramKey>, usize>,
    /// The sets of parameters that compiling the variants of each program read. A job
    /// looks up its variant once per set.
    reads: HashMap<ProgramKey, Vec<Vec<ParameterUid>>>,
}

#[derive(PartialEq, Eq, Hash)]
struct VariantKey<ProgramKey> {
    program: ProgramKey,
    chunk: Option<usize>,
    values: Vec<(ParameterUid, NumericLiteral)>,
}

impl<ProgramKey> Default for VariantCache<ProgramKey> {
    fn default() -> Self {
        Self {
            variants: HashMap::new(),
            reads: HashMap::new(),
        }
    }
}

impl<ProgramKey: Clone + Eq + Hash> VariantCache<ProgramKey> {
    /// Returns the index of the variant that a job with `program`, `chunk` and `env` runs, if
    /// one is recorded.
    pub(crate) fn get(
        &self,
        program: &ProgramKey,
        chunk: Option<usize>,
        env: &Environment,
    ) -> Option<usize> {
        self.reads.get(program)?.iter().find_map(|reads| {
            let values = reads
                .iter()
                .map(|&uid| Some((uid, env.get(uid)?)))
                .collect::<Option<_>>()?;
            let key = VariantKey {
                program: program.clone(),
                chunk,
                values,
            };
            self.variants.get(&key).copied()
        })
    }

    /// Records variant `index`, compiled for `program` and `chunk`, as depending on the
    /// parameter values in `values`.
    pub(crate) fn insert(
        &mut self,
        program: ProgramKey,
        chunk: Option<usize>,
        values: Vec<(ParameterUid, NumericLiteral)>,
        index: usize,
    ) {
        let reads: Vec<_> = values.iter().map(|&(uid, _)| uid).collect();
        let known = self.reads.entry(program.clone()).or_default();
        if !known.contains(&reads) {
            known.push(reads);
        }
        let key = VariantKey {
            program,
            chunk,
            values,
        };
        self.variants.insert(key, index);
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
        cache.insert((), None, values(&[(SELECTOR, 0.0), (A, 0.1)]), 0);
        cache.insert((), None, values(&[(SELECTOR, 1.0), (B, 0.4)]), 1);

        let get =
            |selector, a, b| cache.get(&(), None, &env(&[(SELECTOR, selector), (A, a), (B, b)]));
        assert_eq!(get(0.0, 0.1, 0.9), Some(0));
        assert_eq!(get(1.0, 0.9, 0.4), Some(1));
        assert_eq!(get(1.0, 0.1, 0.3), None);
        assert_eq!(get(0.0, 0.2, 0.4), None);
    }

    #[test]
    fn variants_that_read_equal_values_of_different_parameters_stay_apart() {
        let mut cache = VariantCache::default();
        cache.insert((), None, values(&[(A, 0.1)]), 0);
        cache.insert((), None, values(&[(B, 0.1)]), 1);

        assert_eq!(cache.get(&(), None, &env(&[(A, 0.1), (B, 0.2)])), Some(0));
        assert_eq!(cache.get(&(), None, &env(&[(A, 0.2), (B, 0.1)])), Some(1));
    }

    #[test]
    fn a_variant_runs_in_its_own_chunk_only() {
        let mut cache = VariantCache::default();
        cache.insert((), Some(0), vec![], 0);

        assert_eq!(cache.get(&(), Some(0), &env(&[])), Some(0));
        assert_eq!(cache.get(&(), Some(1), &env(&[])), None);
    }

    #[test]
    fn a_variant_runs_for_its_own_program_key_only() {
        let mut cache = VariantCache::default();
        cache.insert("a", None, values(&[(A, 0.1)]), 0);
        cache.insert("b", None, vec![], 1);

        assert_eq!(cache.get(&"a", None, &env(&[(A, 0.1)])), Some(0));
        assert_eq!(cache.get(&"a", None, &env(&[(A, 0.2)])), None);
        assert_eq!(cache.get(&"b", None, &env(&[(A, 0.2)])), Some(1));
        assert_eq!(cache.get(&"c", None, &env(&[(A, 0.1)])), None);
    }

    fn env(values: &[(u32, f64)]) -> Environment {
        self::values(values).into_iter().collect()
    }

    fn values(values: &[(u32, f64)]) -> Vec<(ParameterUid, NumericLiteral)> {
        values
            .iter()
            .map(|&(id, value)| (id.into(), NumericLiteral::Float(value)))
            .collect()
    }
}
