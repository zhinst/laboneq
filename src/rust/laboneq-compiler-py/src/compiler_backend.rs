// Copyright 2026 Zurich Instruments AG
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;

use laboneq_dsl::experiment_signal::ExperimentSignal;
use pyo3::prelude::*;
use tracing::instrument;

use laboneq_common::compiler_settings::CompilerSettings;
use laboneq_common::settings::SettingSpec;
use laboneq_ir::ExperimentIr;
use laboneq_py_utils::py_object_store::PyObjectStore;
use numeric_array::NumericArray;

use laboneq_common::named_id::NamedIdStore;
use laboneq_common::types::AwgKey;
use laboneq_dsl::ExperimentNode;
use laboneq_dsl::hqcs::{self, HqcsDeclarations};
use laboneq_dsl::types::{
    AcquisitionType, ExternalParameterUid, HandleUid, ParameterUid, PulseDef, PulseUid, SignalUid,
};
use laboneq_units::duration::{Duration, Second};

// Re-export commonly used types for convenience
pub use crate::capnp_deserializer::DeserializedExperiment;
pub use crate::capnp_setup_description::SetupDescriptionPayload;
pub use crate::error::Error;
pub use crate::experiment::DeviceSignal;
pub use crate::signal_view::SignalView;
pub use laboneq_scheduler::FeedbackCalculator;

pub type CompilerBackendResult<T, E = laboneq_error::LabOneQError> = Result<T, E>;

// Re-exported so the compiler backend implementations can use these types without `numeric-array`
pub type ParameterValues = NumericArray;

/// The values every sweep parameter of an experiment takes.
pub type ParameterMap<'a> = HashMap<ParameterUid, &'a ParameterValues>;

/// A processed experiment, as the rules of a [`CompilerBackend`] see it.
pub struct ProcessedExperiment<'a> {
    /// Root of the experiment tree, with every compiler pass applied.
    pub root: &'a ExperimentNode,
    pub parameters: ParameterMap<'a>,
    /// What the experiment's real-time loop acquires.
    pub acquisition_type: AcquisitionType,
}

/// A backend that performs hardware-specific preprocessing of an experiment
/// before it enters the generic compilation pipeline.
///
/// Each hardware family (QCCS, ZQCS, etc.) implements this trait to translate
/// the hardware-agnostic experiment representation into [`PreprocessedBackendData`]
/// that the rest of the compiler can query.
///
/// `Send + Sync + 'static`: the backend ends up stored in `Arc<dyn DynCompilerBackend>` on
/// a Python object shared across Python-bound calls.
pub trait CompilerBackend: Clone + Send + Sync + 'static {
    type Output: PreprocessedBackendData + Send + Sync;
    type CodeGenArtifact: CodeGenArtifact + Send + Sync + 'static;
    type CompilerArtifact: CompilerArtifact;
    type CombinedOutput: CombinedOutput;

    /// The compiler settings this backend accepts, each declared with its
    /// type, default and documentation. The frontend validates the caller's
    /// settings against these and its own in one pass; the backend reads the
    /// values back from [`CompilerSettings::backend`].
    fn settings(&self) -> &'static [SettingSpec] {
        &[]
    }

    /// Backend-specific program collected from HQCS content. `()` for backends
    /// that only refuse it; a real (non-emulated) coprocessor needs no compiler
    /// artifact — its program is the payload.
    type EmulatedHqcsProgram: Default + Send + Sync + 'static;

    /// Name of the backend.
    ///
    /// Must be unique across all backends: it is recorded in
    /// `CompiledExperiment.metadata.createdBy`, selects the reader that decodes
    /// `CompiledExperiment.artifacts`, and selects this backend's compiler hooks
    /// on the Python side.
    fn name(&self) -> &'static str;

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

    /// Check the experiment against this backend's own rules, on top of the
    /// ones the compiler checks for every backend.
    ///
    /// The compiler calls this once it has processed the experiment and checked
    /// its own rules, so the tree is the one that goes on to scheduling.
    fn validate(
        &self,
        _experiment: &ProcessedExperiment<'_>,
        _backend_data: &Self::Output,
    ) -> CompilerBackendResult<()> {
        Ok(())
    }

    /// Decide how to handle HQCS content right after deserialization, before
    /// any other processing sees the tree. Default is to refuse anything.
    fn process_hqcs(
        &self,
        experiment: &mut DeserializedExperiment,
    ) -> CompilerBackendResult<Self::EmulatedHqcsProgram> {
        refuse_hqcs(&experiment.root, &experiment.hqcs, &experiment.id_store)?;
        Ok(Self::EmulatedHqcsProgram::default())
    }

    /// Lower HQCS operations retained for result-shape extraction.
    fn lower_hqcs_for_scheduler(&self, _root: &mut ExperimentNode) -> CompilerBackendResult<()> {
        Ok(())
    }

    /// The sequence the PRNG draws for a given seed and range.
    ///
    /// One PRNG setup can hold several nested loops that draw from the same
    /// sequence, hence a generator rather than the values of one loop. The
    /// default fails: a backend without a PRNG cannot run a PRNG loop.
    fn prng_sequence(
        &self,
        _seed: u32,
        _range: u32,
    ) -> CompilerBackendResult<Box<dyn PrngSequence>> {
        Err(laboneq_error::laboneq_error!(
            "This backend has no PRNG, so it cannot run a PRNG loop."
        ))
    }

    fn generate_code(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        external_parameters: &PyObjectStore<ExternalParameterUid>,
        backend_data: &Self::Output,
    ) -> CompilerBackendResult<Self::CodeGenArtifact>;

    #[instrument(level = "debug", name = "laboneq.compiler.generate-code", skip_all)]
    fn generate_code_traced(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        external_parameters: &PyObjectStore<ExternalParameterUid>,
        backend_data: &Self::Output,
    ) -> CompilerBackendResult<Self::CodeGenArtifact> {
        self.generate_code(
            experiment,
            compiler_settings,
            external_parameters,
            backend_data,
        )
    }

    /// Get a feedback calculator for the given backend. The default implementation returns `Ok(None)`, indicating that no feedback calculator is available for this backend.
    fn feedback_calculator(
        &self,
        _signals: &[SignalView],
        _compiler_settings: &CompilerSettings,
        _backend_data: &Self::Output,
    ) -> Result<Option<Box<dyn FeedbackCalculator<Error = Error> + Send + Sync + 'static>>, Error>
    {
        Ok(None)
    }

    /// Extract this backend's [`CombinedOutput`] from the Python object (`CombinedOutput`)
    /// produced by the linking stage.
    ///
    /// NOTE: This is a temporary workaround as long as the compiler calls Python code that requires this information.
    /// Once the compiler is refactored to be fully Rust-based, this method can be removed
    /// and replaced with a more direct approach.
    fn combined_output_from_py(
        &self,
        obj: &Bound<'_, PyAny>,
    ) -> CompilerBackendResult<Self::CombinedOutput>;

    /// Finishes one compilation attempt on the linked [`CombinedOutput`].
    ///
    /// The compiler makes one attempt per chunk count it tries, so this method may run
    /// several times, each on a freshly linked output. Returning
    /// [`LabOneQError::ResourceExhaustion`] rejects the attempt, and automatic
    /// chunking retries with more chunks. `chunk_count` is the number of chunks the
    /// sweep is split into, or `None` when the sweep is not chunked. It is never `Some(1)`.
    /// Does nothing by default.
    ///
    /// [`LabOneQError::ResourceExhaustion`]: laboneq_error::LabOneQError::ResourceExhaustion
    fn finish_attempt(
        &self,
        _combined: &mut Self::CombinedOutput,
        _compiler_settings: &CompilerSettings,
        _chunk_count: Option<u32>,
    ) -> CompilerBackendResult<()> {
        Ok(())
    }

    /// Produces the [`CompilerArtifact`] from the [`CombinedOutput`] of the accepted
    /// attempt. Runs exactly once, after [`finish_attempt`](Self::finish_attempt)
    /// succeeded on that output.
    fn produce_output(
        &self,
        combined: Self::CombinedOutput,
        hqcs_program: Self::EmulatedHqcsProgram,
    ) -> CompilerBackendResult<CompilationOutput<Self::CompilerArtifact>>;
}

pub fn refuse_hqcs(
    root: &ExperimentNode,
    hqcs: &HqcsDeclarations,
    id_store: &NamedIdStore,
) -> CompilerBackendResult<()> {
    let construct = hqcs::find_hqcs_declaration(hqcs)
        .map(str::to_owned)
        .or_else(|| hqcs::find_hqcs_construct(root, id_store));
    match construct {
        Some(construct) => Err(laboneq_error::laboneq_error!(
            "HQCS constructs found in experiment ({construct}); not yet supported by the compiler."
        )),
        None => Ok(()),
    }
}

/// A combined (linked) real-time compilation output.
pub trait CombinedOutput {
    /// Resolve the raw acquisition length (in samples) for each requested signal/handle pair.
    fn raw_acquisition_lengths(
        &self,
        pairs: &[(SignalUid, HandleUid)],
        id_store: &NamedIdStore,
    ) -> CompilerBackendResult<Vec<(SignalUid, HandleUid, usize)>>;

    /// Total duration of the real-time steps, in seconds.
    fn total_execution_time(&self) -> CompilerBackendResult<f64>;

    /// Maximum duration of a single real-time step, in seconds.
    fn max_execution_time_per_step(&self) -> CompilerBackendResult<f64>;
}

/// A view of the experiment and device setup passed to a [`CompilerBackend`].
pub struct ExperimentView<'a> {
    pub root: &'a ExperimentNode,
    pub id_store: &'a mut NamedIdStore,
    pub parameters: ParameterMap<'a>,
    pub pulses: &'a HashMap<PulseUid, PulseDef>,

    pub experiment_signals: Vec<ExperimentSignal>,

    /// The target setup, in this backend's own Cap'n Proto schema.
    pub setup_description: SetupDescriptionPayload,
}

impl<'a> ExperimentView<'a> {
    pub fn new(
        root: &'a ExperimentNode,
        id_store: &'a mut NamedIdStore,
        parameters: ParameterMap<'a>,
        pulses: &'a HashMap<PulseUid, PulseDef>,
        experiment_signals: Vec<ExperimentSignal>,
        setup_description: SetupDescriptionPayload,
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
    pub(crate) device_setup_fingerprint: String,
}

impl<T: PreprocessedBackendData> PreprocessOutput<T> {
    /// Create a new `PreprocessOutput` with the given backend data and device signals.
    pub fn new(
        backend_data: T,
        device_signals: Vec<DeviceSignal>,
        device_setup_fingerprint: String,
    ) -> Self {
        PreprocessOutput {
            backend_data,
            device_signals,
            device_setup_fingerprint,
        }
    }
}

/// A code generation artifact produced by a [`CompilerBackend`].
pub trait CodeGenArtifact {
    /// Convert the artifact to a Python object for returning to the user.
    fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

/// A compiler artifact produced by a [`CompilerBackend`].
pub trait CompilerArtifact {
    /// Produce an [`ArtifactsWriter`] that fills `CompiledExperiment.artifacts` (an opaque
    /// `AnyPointer`) with this backend's own capnp struct.
    ///
    /// `ctx` contains the state needed to serialize the artifact consistently with the rest of the compiled experiment.
    /// It contains e.g. indexes of the parameters and pulses in the compiled experiment,
    /// so that the artifact can refer to them by index instead of by name.
    ///
    /// [`ArtifactsWriter`]: laboneq_capnp::artifacts::ArtifactsWriter
    fn to_capnp_writer<'a>(
        &'a self,
        ctx: &'a crate::capnp_artifacts::ArtifactSerializerContext<'a>,
    ) -> impl laboneq_capnp::artifacts::ArtifactsWriter + 'a;
}

pub struct CompilationOutput<T: CompilerArtifact> {
    pub artifacts: T,
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
        external_parameters: &PyObjectStore<ExternalParameterUid>,
        backend_data: &(dyn PreprocessedBackendData + Send + Sync),
    ) -> CompilerBackendResult<Box<dyn CodeGenArtifact + Send + Sync>>;

    fn feedback_calculator(
        &self,
        signals: &[SignalView],
        compiler_settings: &CompilerSettings,
        backend_data: &(dyn PreprocessedBackendData + Send + Sync),
    ) -> Result<Option<Box<dyn FeedbackCalculator<Error = Error> + Send + Sync + 'static>>, Error>;
}

impl<B> DynCompilerBackend for B
where
    B: CompilerBackend,
{
    fn generate_code_dyn(
        &self,
        experiment: ExperimentIr,
        compiler_settings: &CompilerSettings,
        external_parameters: &PyObjectStore<ExternalParameterUid>,
        backend_data: &(dyn PreprocessedBackendData + Send + Sync),
    ) -> CompilerBackendResult<Box<dyn CodeGenArtifact + Send + Sync>> {
        let data = backend_data
            .as_any()
            .downcast_ref::<B::Output>()
            .expect("DynCompilerBackend: backend_data type does not match backend Output type");
        self.generate_code_traced(experiment, compiler_settings, external_parameters, data)
            .map(|a| Box::new(a) as Box<dyn CodeGenArtifact + Send + Sync>)
    }

    fn feedback_calculator(
        &self,
        signals: &[SignalView],
        compiler_settings: &CompilerSettings,
        backend_data: &(dyn PreprocessedBackendData + Send + Sync),
    ) -> Result<Option<Box<dyn FeedbackCalculator<Error = Error> + Send + Sync + 'static>>, Error>
    {
        let data = backend_data
            .as_any()
            .downcast_ref::<B::Output>()
            .expect("DynCompilerBackend: backend_data type does not match backend Output type");
        <Self as CompilerBackend>::feedback_calculator(self, signals, compiler_settings, data)
    }
}

/// The pseudo-random sequence a device's PRNG produces.
///
/// The compiler replays it to label the result axis of a PRNG loop with the
/// values the loop draws.
pub trait PrngSequence {
    fn generate(&mut self) -> u32;
}

/// Hardware-specific data produced by a [`CompilerBackend`], queried
/// by later compilation stages
pub trait PreprocessedBackendData: Any {
    /// Get AWG key for signal. Required for all valid signals.
    fn awg_key(&self, signal_uid: SignalUid) -> CompilerBackendResult<AwgKey>;

    /// Get lead delay for signal.
    fn lead_delay(&self, signal_uid: SignalUid) -> Duration<Second>;

    /// Returns `&dyn Any` to allow downcasting to the concrete type.
    ///
    /// NOTE: This is a temporary workaround for codegenerators that need to access backend-specific data.
    /// Once codegenerator is refactored to be part of the compiler backend, this can be removed and replaced with direct access to backend-specific data.
    fn as_any(&self) -> &dyn Any;
}
