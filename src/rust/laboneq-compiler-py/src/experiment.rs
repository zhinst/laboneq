// Copyright 2025 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::{num::NonZeroU16, sync::Arc};

use laboneq_common::device_traits::DeviceTraits;
use laboneq_common::types::{PhysicalDeviceUid, SignalKind};
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::signal_calibration::SignalCalibration;
use laboneq_dsl::types::{
    DeviceUid, ExternalParameterUid, PulseDef, PulseUid, SignalUid, SweepParameter,
};
use laboneq_py_utils::py_object_store::PyObjectStore;
use laboneq_units::duration::{Frequency, Hertz};
use rustc_hash::FxHashMap;

use laboneq_common::named_id::NamedIdStore;

pub(crate) struct Experiment {
    /// Root node of the experiment tree
    pub root: ExperimentNode,
    // NOTE: The usage of Arc here is to allow sharing the id_store across Python bindings
    // Remove when Python bindings are no longer needed
    pub id_store: Arc<NamedIdStore>,
    pub parameters: Vec<SweepParameter>,
    pub pulses: FxHashMap<PulseUid, PulseDef>,
    pub external_parameters: Arc<PyObjectStore<ExternalParameterUid>>,
}

/// Device signal definition, representing a signal in the device setup.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSignal {
    /// Identification parameters
    pub uid: SignalUid,
    pub device_uid: DeviceUid,
    /// Physical device the signal's device maps to.
    /// This UID is used to group virtual devices that share the same
    /// physical hardware, enabling proper device detection.
    pub physical_device_uid: PhysicalDeviceUid,
    /// Capabilities of the hardware the signal plays on.
    pub traits: &'static DeviceTraits,

    /// Configuration parameters
    pub kind: SignalKind,
    pub calibration: SignalCalibration,
    /// Delay signal in samples
    pub delay_signal: i64,
    pub sampling_rate: Frequency<Hertz>,
    /// The sequencer grid, in samples at `sampling_rate`.
    pub sample_multiple: NonZeroU16,
}
