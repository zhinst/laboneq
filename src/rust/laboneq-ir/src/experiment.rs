// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{AcquisitionType, SweepParameter};

use crate::node::IrNode;

pub struct ExperimentIr {
    pub root: IrNode,
    pub acquisition_type: AcquisitionType,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub id_store: Arc<NamedIdStore>,
    pub parameters: Vec<SweepParameter>,
}
