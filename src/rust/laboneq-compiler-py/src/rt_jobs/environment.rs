// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;

use laboneq_dsl::types::{NumericLiteral, ParameterUid};
use laboneq_ir::ParameterLookup;

/// The values of the near-time parameters in scope, innermost last. A later
/// value of a parameter shadows an earlier one.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Environment {
    values: Vec<(ParameterUid, NumericLiteral)>,
}

impl Environment {
    pub(crate) fn get(&self, uid: ParameterUid) -> Option<NumericLiteral> {
        self.position(uid).map(|index| self.values[index].1)
    }

    pub(crate) fn push(&mut self, uid: ParameterUid, value: NumericLiteral) {
        self.values.push((uid, value));
    }

    /// Returns the number of values in scope, shadowed ones included.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    fn position(&self, uid: ParameterUid) -> Option<usize> {
        self.values.iter().rposition(|(p, _)| *p == uid)
    }
}

impl FromIterator<(ParameterUid, NumericLiteral)> for Environment {
    fn from_iter<I: IntoIterator<Item = (ParameterUid, NumericLiteral)>>(iter: I) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }
}

/// A [`ParameterLookup`] over an [`Environment`] that records which parameters are read.
pub(crate) struct RecordingLookup<'a> {
    env: &'a Environment,
    read: Vec<Cell<bool>>,
}

impl<'a> RecordingLookup<'a> {
    pub(crate) fn new(env: &'a Environment) -> Self {
        let read = vec![Cell::new(false); env.values.len()];
        Self { env, read }
    }

    /// Returns the parameters read so far, in the order of the environment's values.
    pub(crate) fn reads(&self) -> Vec<ParameterUid> {
        self.env
            .values
            .iter()
            .zip(&self.read)
            .filter(|(_, read)| read.get())
            .map(|((uid, _), _)| *uid)
            .collect()
    }
}

impl ParameterLookup for RecordingLookup<'_> {
    fn value(&self, uid: ParameterUid) -> Option<NumericLiteral> {
        let index = self.env.position(uid)?;
        self.read[index].set(true);
        Some(self.env.values[index].1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_later_value_of_a_parameter_shadows_an_earlier_one() {
        let env = Environment::from_iter([(1.into(), float(0.1)), (1.into(), float(0.2))]);

        assert_eq!(env.get(1.into()), Some(float(0.2)));
    }

    #[test]
    fn truncating_restores_a_shadowed_value() {
        let mut env = Environment::default();
        env.push(1.into(), float(0.1));
        let len = env.len();
        env.push(1.into(), float(0.2));

        env.truncate(len);

        assert_eq!(env.get(1.into()), Some(float(0.1)));
    }

    #[test]
    fn a_lookup_records_the_parameters_it_returns() {
        let env = Environment::from_iter([(1.into(), float(0.1)), (2.into(), float(0.2))]);
        let lookup = RecordingLookup::new(&env);

        assert_eq!(lookup.value(2.into()), Some(float(0.2)));
        assert_eq!(lookup.value(2.into()), Some(float(0.2)));
        assert_eq!(lookup.value(3.into()), None);

        assert_eq!(lookup.reads(), vec![ParameterUid::from(2)]);
    }

    #[test]
    fn a_lookup_records_a_shadowed_parameter_once() {
        let env = Environment::from_iter([
            (1.into(), float(0.1)),
            (2.into(), float(0.2)),
            (1.into(), float(0.3)),
        ]);
        let lookup = RecordingLookup::new(&env);

        assert_eq!(lookup.value(1.into()), Some(float(0.3)));

        assert_eq!(lookup.reads(), vec![ParameterUid::from(1)]);
    }

    #[test]
    fn a_lookup_reports_reads_in_the_order_of_the_values() {
        let env = Environment::from_iter([(2.into(), float(0.2)), (1.into(), float(0.1))]);
        let lookup = RecordingLookup::new(&env);

        lookup.value(1.into());
        lookup.value(2.into());

        assert_eq!(
            lookup.reads(),
            vec![ParameterUid::from(2), ParameterUid::from(1)]
        );
    }

    fn float(value: f64) -> NumericLiteral {
        NumericLiteral::Float(value)
    }
}
