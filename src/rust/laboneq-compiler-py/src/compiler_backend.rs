// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;

use laboneq_dsl::experiment_signal::ExperimentSignal;
use pyo3::prelude::*;
use smallvec::SmallVec;
use tracing::instrument;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_ir::ExperimentIr;
use laboneq_py_utils::py_object_interner::PyObjectInterner;
use numeric_array::NumericArray;

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::AwgKey;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::device_setup::SetupDescription;
use laboneq_dsl::types::{ExternalParameterUid, ParameterUid, PulseDef, PulseUid, SignalUid};
use laboneq_ir::system::AwgDevice;
use laboneq_units::duration::{Duration, Second};

// Re-export commonly used types for convenience
pub use crate::error::Error;
pub use crate::experiment::DeviceSignal;
pub use crate::qccs_feedback_calculator::QccsFeedbackCalculator;
pub use crate::signal_view::SignalView;
pub use laboneq_scheduler::FeedbackCalculator;

pub type CompilerBackendResult<T, E = laboneq_error::LabOneQError> = Result<T, E>;

// Re-exported so the compiler backend implementations can use these types without `numeric-array`
pub type ParameterValues = NumericArray;

/// A backend that performs hardware-specific preprocessing of an experiment
/// before it enters the generic compilation pipeline.
///
/// Each hardware family (QCCS, ZQCS, etc.) implements this trait to translate
/// the hardware-agnostic experiment representation into [`PreprocessedBackendData`]
/// that the rest of the compiler can query.
pub trait CompilerBackend {
    type Output: PreprocessedBackendData;
    type CodeGenArtifact: CodeGenArtifact;

    /// Analyze the experiment against the target device setup and produce
    /// hardware-specific data needed for subsequent compilation stages.
    ///
    /// # Errors
    /// Returns an error if the experiment is incompatible with this backend
    /// (e.g. unsupported device combinations, missing ports, clock issues).
    fn preprocess_experiment(
        &self,
        experiment: ExperimentView,
    ) -> CompilerBackendResult<PreprocessOutput<Self::Output>>;

    fn generate_code(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        py_object_store: &PyObjectInterner<ExternalParameterUid>,
        backend_data: &Self::Output,
    ) -> CompilerBackendResult<Self::CodeGenArtifact>;

    #[instrument(name = "laboneq.compiler.generate-code", skip_all)]
    fn generate_code_traced(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        py_object_store: &PyObjectInterner<ExternalParameterUid>,
        backend_data: &Self::Output,
    ) -> CompilerBackendResult<Self::CodeGenArtifact> {
        self.generate_code(experiment, compiler_settings, py_object_store, backend_data)
    }

    /// A numeric identifier for the device class this backend targets.
    ///
    /// NOTE: This is a temporary workaround as long as the compiler calls Python code that requires this information.
    fn device_class(&self) -> usize;

    /// Get a feedback calculator for the given backend. The default implementation returns `None`, indicating that no feedback calculator is available for this backend.
    fn feedback_calculator(
        &self,
        _signals: &[SignalView],
    ) -> Option<Box<dyn FeedbackCalculator<Error = Error> + Send + Sync + 'static>> {
        None
    }
}

/// A view of the experiment and device setup passed to a [`CompilerBackend`].
pub struct ExperimentView<'a> {
    pub root: &'a ExperimentNode,
    pub id_store: &'a mut NamedIdStore,
    pub parameters: HashMap<ParameterUid, &'a ParameterValues>,
    pub pulses: &'a HashMap<PulseUid, PulseDef>,

    pub experiment_signals: Vec<ExperimentSignal>,

    // Device setup properties
    pub setup_description: SetupDescription,
}

impl<'a> ExperimentView<'a> {
    pub fn new(
        root: &'a ExperimentNode,
        id_store: &'a mut NamedIdStore,
        parameters: HashMap<ParameterUid, &'a ParameterValues>,
        pulses: &'a HashMap<PulseUid, PulseDef>,
        experiment_signals: Vec<ExperimentSignal>,
        setup_description: SetupDescription,
    ) -> Self {
        ExperimentView {
            root,
            id_store,
            parameters,
            pulses,
            experiment_signals,
            setup_description,
        }
    }
}

/// Output of the [`CompilerBackend`] preprocessing stage, containing hardware-specific data and optionally modified device signals.
///
/// The backend can modify the device signals if needed (e.g. add virtual signals, modify existing signals, etc.).
/// For backends that do not need to modify the signals, the input signals can simply be returned as-is.
pub struct PreprocessOutput<T: PreprocessedBackendData> {
    pub(crate) backend_data: T,
    pub(crate) device_signals: Vec<DeviceSignal>,
    pub(crate) awg_devices: Vec<AwgDevice>,
}

impl<T: PreprocessedBackendData> PreprocessOutput<T> {
    /// Create a new `PreprocessOutput` with the given backend data and device signals.
    pub fn new(
        backend_data: T,
        device_signals: Vec<DeviceSignal>,
        awg_devices: Vec<AwgDevice>,
    ) -> Self {
        PreprocessOutput {
            backend_data,
            device_signals,
            awg_devices,
        }
    }
}

/// A code generation artifact produced by a [`CompilerBackend`].
pub trait CodeGenArtifact {
    /// Convert the artifact to a Python object for returning to the user.
    fn to_python(&self, py: Python) -> PyResult<Py<PyAny>>;
}

/// Object-safe version of [`CompilerBackend`] for storage in `ExperimentPy`.
///
/// `CompilerBackend` cannot be used as a trait object directly because its associated types
/// prevent object safety. This trait erases them so the backend can be stored as
/// `Arc<dyn DynCompilerBackend>`.
///
/// All concrete [`CompilerBackend`] implementors automatically implement this trait
/// via the blanket impl below.
pub(crate) trait DynCompilerBackend: Send + Sync {
    /// Equivalent to [`CompilerBackend::generate_code`] with type-erased `backend_data`.
    ///
    /// `backend_data` is a reference to the erased `B::Output` stored in `ExperimentPy`.
    /// The caller passes `&*experiment.backend_data` (deref of the stored `Arc<dyn Any>`).
    fn generate_code_dyn(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        py_object_store: &PyObjectInterner<ExternalParameterUid>,
        backend_data: &(dyn PreprocessedBackendData + Send + Sync),
    ) -> CompilerBackendResult<Box<dyn CodeGenArtifact + Send + Sync>>;

    fn feedback_calculator(
        &self,
        signals: &[SignalView],
    ) -> Option<Box<dyn FeedbackCalculator<Error = Error> + Send + Sync + 'static>>;
}

impl<B> DynCompilerBackend for B
where
    B: CompilerBackend + Send + Sync,
    B::Output: Send + Sync + 'static,
    B::CodeGenArtifact: Send + Sync + 'static,
{
    fn generate_code_dyn(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        py_object_store: &PyObjectInterner<ExternalParameterUid>,
        backend_data: &(dyn PreprocessedBackendData + Send + Sync),
    ) -> CompilerBackendResult<Box<dyn CodeGenArtifact + Send + Sync>> {
        let data = backend_data
            .as_any()
            .downcast_ref::<B::Output>()
            .expect("DynCompilerBackend: backend_data type does not match backend Output type");
        self.generate_code_traced(experiment, compiler_settings, py_object_store, data)
            .map(|a| Box::new(a) as Box<dyn CodeGenArtifact + Send + Sync>)
    }

    fn feedback_calculator(
        &self,
        signals: &[SignalView],
    ) -> Option<Box<dyn FeedbackCalculator<Error = Error> + Send + Sync + 'static>> {
        <Self as CompilerBackend>::feedback_calculator(self, signals)
    }
}

/// Hardware-specific data produced by a [`CompilerBackend`], queried
/// by later compilation stages
pub trait PreprocessedBackendData: Any {
    /// Get AWG key for signal. Required for all valid signals.
    fn awg_key(&self, signal_uid: SignalUid) -> CompilerBackendResult<AwgKey>;

    /// Get channels for signal. May not be available for all backends/devices.
    fn channels(&self, signal_uid: SignalUid) -> Option<&SmallVec<[u16; 4]>>;

    /// Get lead delay for signal.
    fn lead_delay(&self, signal_uid: SignalUid) -> Duration<Second>;

    /// Returns `&dyn Any` to allow downcasting to the concrete type.
    ///
    /// NOTE: This is a temporary workaround for codegenerators that need to access backend-specific data.
    /// Once codegenerator is refactored to be part of the compiler backend, this can be removed and replaced with direct access to backend-specific data.
    fn as_any(&self) -> &dyn Any;
}
