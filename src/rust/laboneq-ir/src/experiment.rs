// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use laboneq_common::named_id::NamedIdStore;
use laboneq_dsl::types::{AcquisitionType, PulseDef, SweepParameter};

use crate::node::IrNode;
use crate::system::DeviceSetup;

pub struct ExperimentIr<'a> {
    pub root: IrNode,
    pub acquisition_type: AcquisitionType,
    pub id_store: &'a NamedIdStore,
    pub parameters: Vec<SweepParameter>,
    pub pulses: Vec<PulseDef>,
    pub device_setup: &'a DeviceSetup,
}
