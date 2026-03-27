// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use crate::NamedIdStore;

use laboneq_dsl::ExperimentNode;
use laboneq_dsl::types::{ExternalParameterUid, ParameterUid, PulseDef, PulseUid, SweepParameter};
use laboneq_py_utils::py_object_interner::PyObjectInterner;

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
