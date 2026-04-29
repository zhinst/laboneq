// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use smallvec::SmallVec;
use std::any::Any;
use std::collections::HashMap;

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::AwgKey;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::device_setup::{AuxiliaryDevice, DeviceSignal};
use laboneq_dsl::types::{ParameterUid, PulseDef, PulseUid, SignalUid, SweepParameter};
use laboneq_ir::system::AwgDevice;

pub type CompilerBackendResult<T, E = laboneq_error::LabOneQError> = Result<T, E>;

/// A backend that performs hardware-specific preprocessing of an experiment
/// before it enters the generic compilation pipeline.
///
/// Each hardware family (QCCS, ZQCS, etc.) implements this trait to translate
/// the hardware-agnostic experiment representation into [`PreprocessedBackendData`]
/// that the rest of the compiler can query.
pub trait CompilerBackend {
    type Output: PreprocessedBackendData;

    /// Analyze the experiment against the target device setup and produce
    /// hardware-specific data needed for subsequent compilation stages.
    ///
    /// # Errors
    /// Returns an error if the experiment is incompatible with this backend
    /// (e.g. unsupported device combinations, missing ports, clock issues).
    fn preprocess_experiment(
        &self,
        experiment: ExperimentView,
    ) -> CompilerBackendResult<Self::Output>;
}

/// A view of the experiment and device setup passed to a [`CompilerBackend`].
pub struct ExperimentView<'a> {
    pub root: &'a ExperimentNode,
    pub id_store: &'a mut NamedIdStore,
    pub parameters: &'a HashMap<ParameterUid, SweepParameter>,
    pub pulses: &'a HashMap<PulseUid, PulseDef>,

    // Device setup properties
    pub awg_devices: &'a [AwgDevice],
    pub auxiliary_devices: &'a [AuxiliaryDevice],
    pub signals: &'a [DeviceSignal],
}

impl<'a> ExperimentView<'a> {
    pub fn new(
        root: &'a ExperimentNode,
        id_store: &'a mut NamedIdStore,
        parameters: &'a HashMap<ParameterUid, SweepParameter>,
        pulses: &'a HashMap<PulseUid, PulseDef>,
        awg_devices: &'a [AwgDevice],
        auxiliary_devices: &'a [AuxiliaryDevice],
        signals: &'a [DeviceSignal],
    ) -> Self {
        ExperimentView {
            root,
            id_store,
            parameters,
            pulses,
            awg_devices,
            auxiliary_devices,
            signals,
        }
    }
}

/// Hardware-specific data produced by a [`CompilerBackend`], queried
/// by later compilation stages
pub trait PreprocessedBackendData: Any {
    /// Get AWG key for signal. Required for all valid signals.
    fn awg_key(&self, signal_uid: SignalUid) -> CompilerBackendResult<AwgKey>;

    /// Get channels for signal. May not be available for all backends/devices.
    fn channels(&self, signal_uid: SignalUid) -> Option<&SmallVec<[u16; 4]>>;

    /// Get additional signals to be added to the device setup. This can be used to add virtual signals that are not part of the original experiment definition, e.g., for triggering/synchronization purposes.
    fn additional_signals(&self) -> &[DeviceSignal] {
        &[]
    }

    /// Returns `&dyn Any` to allow downcasting to the concrete type.
    ///
    /// NOTE: This is a temporary workaround for codegenerators that need to access backend-specific data.
    /// Once codegenerator is refactored to be part of the compiler backend, this can be removed and replaced with direct access to backend-specific data.
    fn as_any(&self) -> &dyn Any;
}
