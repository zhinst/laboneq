// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use laboneq_common::types::SignalKind;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::signal_calibration::SignalCalibration;
use laboneq_dsl::types::{
    DeviceUid, ExternalParameterUid, ParameterUid, PulseDef, PulseUid, SignalUid, SweepParameter,
};
use laboneq_py_utils::py_object_interner::PyObjectInterner;

use crate::NamedIdStore;
use crate::error::{Error, Result};

pub(crate) struct Experiment {
    /// Root node of the experiment tree
    pub root: ExperimentNode,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub id_store: Arc<NamedIdStore>,
    pub parameters: HashMap<ParameterUid, SweepParameter>,
    pub pulses: HashMap<PulseUid, PulseDef>,
    pub py_object_store: Arc<PyObjectInterner<ExternalParameterUid>>,
}

impl Experiment {
    /// Get a sweep parameter by its UID.
    pub(crate) fn get_sweep_parameter(&self, uid: &ParameterUid) -> Result<&SweepParameter> {
        self.parameters.get(uid).ok_or_else(|| {
            Error::new(format!(
                "Parameter with UID {} not found in experiment",
                uid.0
            ))
        })
    }
}

/// Device signal definition, representing a signal in the device setup.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSignal {
    // Identification parameters
    pub uid: SignalUid,
    pub device_uid: DeviceUid,

    // Configuration parameters
    pub ports: Vec<String>,
    pub kind: SignalKind,
    pub calibration: SignalCalibration,
}
