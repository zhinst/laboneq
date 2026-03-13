# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xc6cbf1a813ad1ae4;

using Common = import "common.capnp";
using PulseDef = import "pulse.capnp";
using Section = import "section.capnp";
using Sweep = import "sweep.capnp";

# Schema version. The schema is in 0.x development mode:
# no compatibility guarantees exist between 0.x releases.
const schemaVersion :Text = "0.1";

# ========================================================================================
# Experiment

struct Experiment {
  # A complete pulse-level quantum experiment definition.
  #
  # Contains everything the compiler needs to generate device code:
  # - A section tree defining the experiment's temporal structure.
  # - Pulse definitions (waveforms).
  # - Sweep parameters for parametric variation.
  #
  # The experiment is self-contained: no out-of-band context is required for compilation.

  metadata @0 :Metadata;

  deviceSetup @1 :AnyPointer;
  # Device setup definition. Reserved.

  signals @2 :List(ExperimentSignal);
  # Experiment signal declarations. Each signal is a logical channel used by operations
  # in the section tree.

  sweepParameters @3 :List(Sweep.SweepParameter);
  # All sweep parameter definitions referenced by `SweepSection` and `Value.parameterRef`.

  pulses @4 :List(PulseDef.Pulse);
  # All pulse (waveform) definitions referenced by `PlayOp` and `AcquireOp`.

  rootSection @5 :Section.Section;
  # Root of the experiment's timing hierarchy. All operations are nested inside this tree.

  acquisitionHandles @6 :List(AcquisitionHandle);
  # Pre-declared acquisition handles referenced by `AcquireOp.handle` and
  # `MatchSection.handle`.
}

struct AcquisitionHandle {
  # A pre-declared acquisition result handle.
  #
  # Handles are referenced by zero-based index in `Experiment.acquisitionHandles`.

  uid @0 :Text;
  # Text UID.
}

# ========================================================================================
# Metadata

struct Metadata {
  # Experiment metadata for identification and versioning.

  uid @0 :Text;
  # Unique identifier for this experiment instance.

  schemaVersion @1 :Text;
  # Schema version used by the producer (see `schemaVersion` const above). The compiler
  # uses this for compatibility validation: 0.x requires exact match; 1.x+ uses semver
  # (major must match, minor must be <= compiler).

  createdBy @2 :Text;
  # Producer identifier, e.g. "laboneq/26.4".

  annotations @3 :List(Common.StringEntry);
  # Arbitrary key-value annotations for user metadata. Not interpreted by the compiler.
}

# ========================================================================================
# Signal Declarations

struct ExperimentSignal {
  # Declares a logical signal used in the experiment.
  #
  # Experiment signals are abstract channels that operations reference by zero-based
  # index. They are connected to physical hardware ports via device setup configuration.

  uid @0 :Text;
  # Text UID, e.g. "q0_drive" or "q1_measure".
}
