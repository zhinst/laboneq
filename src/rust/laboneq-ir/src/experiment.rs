// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{AcquisitionType, NumericLiteral, ParameterUid, PulseDef, SweepParameter};

use crate::node::IrNode;
use crate::signal::Signal;

pub struct ExperimentIr<'a> {
    pub root: IrNode,
    pub acquisition_type: AcquisitionType,
    pub id_store: &'a NamedIdStore,
    pub parameters: Vec<SweepParameter>,
    pub pulses: Vec<PulseDef>,
    pub signals: &'a [Signal],
    pub near_time_parameters: &'a dyn ParameterLookup,
}

/// A source of sweep parameter values.
pub trait ParameterLookup {
    fn value(&self, uid: ParameterUid) -> Option<NumericLiteral>;
}

impl ExperimentIr<'_> {
    /// Resolve a near-time parameter value.
    pub fn resolve(&self, uid: ParameterUid) -> Option<NumericLiteral> {
        self.near_time_parameters.value(uid)
    }
}
