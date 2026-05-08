// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_compiler_py::compiler_backend::ExperimentView;
use laboneq_dsl::device_setup::AuxiliaryDevice;
use laboneq_dsl::device_setup::DeviceSignal;

use laboneq_dsl::device_setup::Instrument;
use laboneq_dsl::types::DeviceUid;
use laboneq_error::laboneq_error;

use crate::Result;

pub(crate) struct ExperimentViewWrapper<'a> {
    pub id_store: &'a mut NamedIdStore,

    // Device setup properties
    pub instruments: Vec<Instrument>,
    pub auxiliary_devices: &'a [AuxiliaryDevice],
    pub signals: Vec<DeviceSignal>,

    // Lookup maps for efficient access to device properties by UID
    pub device_indices: HashMap<DeviceUid, usize>,
}

impl<'a> ExperimentViewWrapper<'a> {
    pub(crate) fn from_experiment_view(experiment: ExperimentView<'a>) -> Self {
        let device_indices = experiment
            .instruments
            .iter()
            .enumerate()
            .map(|(i, d)| (d.uid, i))
            .collect::<HashMap<_, _>>();

        ExperimentViewWrapper {
            id_store: experiment.id_store,
            instruments: experiment.instruments,
            auxiliary_devices: experiment.auxiliary_devices,
            signals: experiment.signals,
            device_indices,
        }
    }
}

impl ExperimentViewWrapper<'_> {
    pub(crate) fn get_device_by_uid(&self, uid: DeviceUid) -> Result<&Instrument> {
        let device = self
            .device_indices
            .get(&uid)
            .map(|&index| &self.instruments[index]);
        device.ok_or_else(|| laboneq_error!("Device with UID {} not found", uid.0))
    }
}
