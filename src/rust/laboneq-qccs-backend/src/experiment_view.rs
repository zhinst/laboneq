// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use indexmap::IndexMap;
use laboneq_common::named_id::NamedIdStore;
use laboneq_compiler_py::compiler_backend::ExperimentView;
use laboneq_compiler_py::compiler_backend::ParameterValues;
use laboneq_dsl::device_setup::Instrument;
use laboneq_dsl::device_setup::SetupDescription;
use laboneq_dsl::setup_description_qccs::AuxiliaryDevice;
use laboneq_dsl::signal_calibration::SignalCalibration;
use laboneq_dsl::types::DeviceUid;

use laboneq_dsl::types::ParameterUid;
use laboneq_dsl::types::SignalUid;
use laboneq_error::laboneq_error;

use crate::Result;
use crate::ports::Port;
use crate::ports::parse_port;

pub(crate) struct ExperimentSignal {
    pub uid: SignalUid,
    pub device_uid: DeviceUid,

    pub ports: Vec<Port>,
    pub calibration: SignalCalibration,
}

pub(crate) struct ExperimentViewWrapper<'a> {
    pub id_store: &'a mut NamedIdStore,

    // Device setup properties
    pub instruments: IndexMap<DeviceUid, Instrument>,
    pub auxiliary_devices: Vec<AuxiliaryDevice>,
    pub signals: Vec<ExperimentSignal>,

    parameters: HashMap<ParameterUid, &'a ParameterValues>,
}

impl<'a> ExperimentViewWrapper<'a> {
    pub(crate) fn from_experiment_view(experiment: ExperimentView<'a>) -> Result<Self> {
        let setup = match experiment.setup_description {
            SetupDescription::Qccs(setup) => setup,
            _ => panic!("Unsupported setup description type for QCCS backend"),
        };

        let instrument_map = setup
            .instruments
            .into_iter()
            .map(|instrument| (instrument.uid, instrument))
            .collect::<IndexMap<_, _>>();

        let physical_channel_map = setup
            .device_signals
            .into_iter()
            .map(|signal| (signal.uid.clone(), signal))
            .collect::<IndexMap<_, _>>();

        let mut signals = Vec::with_capacity(experiment.experiment_signals.len());
        for signal in experiment.experiment_signals {
            let physical_channel = physical_channel_map.get(&signal.maps_to).ok_or_else(|| {
                laboneq_error!(
                    "Experiment signal maps to unknown physical channel '{}'",
                    signal.maps_to
                )
            })?;

            let instrument = instrument_map
                .get(&physical_channel.device_uid)
                .ok_or_else(|| {
                    laboneq_error!(
                        "Physical channel maps to unknown device '{}'",
                        physical_channel.device_uid.0
                    )
                })?;

            signals.push(ExperimentSignal {
                uid: signal.uid,
                device_uid: physical_channel.device_uid,
                ports: physical_channel
                    .ports
                    .iter()
                    .map(|p| parse_port(p, instrument.kind))
                    .collect::<Result<Vec<_>>>()?,
                calibration: signal.calibration,
            });
        }
        let exp = ExperimentViewWrapper {
            id_store: experiment.id_store,
            instruments: instrument_map,
            auxiliary_devices: setup.auxiliary_devices,
            signals,
            parameters: experiment.parameters,
        };
        Ok(exp)
    }
}

impl ExperimentViewWrapper<'_> {
    pub(crate) fn get_device_by_uid(&self, uid: DeviceUid) -> Result<&Instrument> {
        let device = self.instruments.get(&uid);
        device.ok_or_else(|| laboneq_error!("Device with UID {} not found", uid.0))
    }

    pub(crate) fn get_signal_mut(
        &mut self,
        signal_uid: SignalUid,
    ) -> Result<&mut ExperimentSignal> {
        self.signals
            .iter_mut()
            .find(|s| s.uid == signal_uid)
            .ok_or_else(|| laboneq_error!("Signal with UID {} not found", signal_uid.0))
    }

    pub(crate) fn get_parameter_values(
        &self,
        parameter_uid: ParameterUid,
    ) -> Result<&ParameterValues> {
        self.parameters
            .get(&parameter_uid)
            .copied()
            .ok_or_else(|| laboneq_error!("Parameter with UID {} not found", parameter_uid.0))
    }
}
