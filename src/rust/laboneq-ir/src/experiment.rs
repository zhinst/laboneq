// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{
    AcquisitionType, NumericLiteral, ParameterUid, PulseDef, PulseUid, SweepParameter,
};
use rustc_hash::FxHashMap;

use crate::node::IrNode;
use crate::signal::Signal;

pub struct ExperimentIr<'a> {
    pub root: IrNode,
    pub acquisition_type: AcquisitionType,
    pub id_store: &'a NamedIdStore,
    pub parameters: Vec<SweepParameter>,
    pub pulses: &'a FxHashMap<PulseUid, PulseDef>,
    pub signals: &'a [Signal],
    pub near_time_parameters: &'a dyn ParameterLookup,
}

/// A lookup of the values of the near-time parameters in scope of the real-time program.
pub trait ParameterLookup {
    /// Returns the value of `uid`, or `None` when `uid` is not a near-time parameter
    /// in scope.
    fn value(&self, uid: ParameterUid) -> Option<NumericLiteral>;
}

impl ExperimentIr<'_> {
    /// Resolve a near-time parameter value.
    pub fn resolve(&self, uid: ParameterUid) -> Option<NumericLiteral> {
        self.near_time_parameters.value(uid)
    }
}
