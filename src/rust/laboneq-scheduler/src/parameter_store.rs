// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_dsl::types::{NumericLiteral, ParameterUid};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Store for parameters to be used in the real-time part of an experiment.
///
/// The store keeps track of which parameters have been queried, allowing
/// detection of used parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterStore {
    /// Parameters per real-time part
    parameters: HashMap<ParameterUid, NumericLiteral>,
    /// Keep track of which parameters have been queried.
    /// This is used to detect used parameters.
    ///
    /// TODO: A better way to track the used parameters?
    /// Now we are following the current pattern from Python compiler.
    queries: RefCell<HashSet<ParameterUid>>,
}

impl ParameterStore {
    /// Get a parameter value by its UID, marking it as queried.
    pub fn get(&self, uid: &ParameterUid) -> Option<&NumericLiteral> {
        self.queries.borrow_mut().insert(*uid);
        self.parameters.get(uid)
    }

    /// Empties and returns the set of queried parameters since the last call.
    pub fn empty_queries(&mut self) -> HashSet<ParameterUid> {
        std::mem::take(&mut self.queries.borrow_mut())
    }

    pub fn available_parameters(&self) -> HashSet<ParameterUid> {
        self.parameters.keys().cloned().collect()
    }
}

#[derive(Default)]
pub struct ParameterStoreBuilder {
    parameters: HashMap<ParameterUid, NumericLiteral>,
}

impl ParameterStoreBuilder {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    pub fn with_parameter(mut self, uid: ParameterUid, value: NumericLiteral) -> Self {
        self.parameters.insert(uid, value);
        self
    }

    pub fn build(self) -> ParameterStore {
        ParameterStore {
            parameters: self.parameters,
            queries: RefCell::new(HashSet::new()),
        }
    }
}
