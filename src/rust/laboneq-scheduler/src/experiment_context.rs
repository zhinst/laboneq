// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::experiment::sweep_parameter::SweepParameter;
use crate::experiment::types::{HandleUid, ParameterUid, SignalUid};
use crate::signal_info::SignalInfo;
use laboneq_common::named_id::{NamedId, NamedIdStore};

/// Experiment context.
///
/// The context stores the references to the components of the experiment tree and
/// use used to access the components.
pub struct ExperimentContext<'a, T: SignalInfo> {
    pub id_store: &'a NamedIdStore,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub signals: &'a HashMap<SignalUid, T>,
    pub handle_to_signal: &'a HashMap<HandleUid, SignalUid>,
}

impl<T: SignalInfo> ExperimentContext<'_, T> {
    pub fn signals(&self) -> impl Iterator<Item = &T> {
        self.signals.values()
    }

    pub fn get_signal(&self, uid: &SignalUid) -> Result<&T> {
        self.signals.get(uid).ok_or_else(|| {
            Error::new(format!(
                "Signal with uid {:?} not found in the experiment context",
                uid
            ))
        })
    }

    pub fn resolve_uid<K: Into<NamedId> + std::fmt::Debug + Copy>(&self, uid: K) -> Result<&str> {
        self.id_store
            .resolve(uid.into())
            .ok_or_else(|| Error::new(format!("Failed to resolve section ID {:?}", uid)))
    }

    pub fn sweep_parameter(&self, uid: &ParameterUid) -> Result<&SweepParameter> {
        self.parameters
            .get(uid)
            .ok_or_else(|| Error::new(format!("Undefined parameter '{}'.", uid.0)))
    }
}
