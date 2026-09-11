# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xbe8258265fbab120;

using ResultShape = import "result_shape.capnp";
using RtProps = import "rt_loop_properties.capnp";
using Exec = import "execution.capnp";
using Values = import "values.capnp";
using PulseSheet = import "pulse_sheet.capnp";

const schemaVersion :Text = "0.1";
# Schema version. The schema is in 0.x development mode:
# no compatibility guarantees exist between 0.x releases.

struct CompiledExperiment {
  # A program ready for execution.

  # --- Identity ---

  metadata @0 :Metadata;

  hardwareFingerprint @1 :Text;
  # An identifier of the target hardware configuration.

  # --- Shared reference tables (defined before their users) ---

  parameters @2 :List(ParameterEntry);
  # Named parameter table. Referenced by index from SetParameter (execution),
  # AxisParameter (result shapes).

  acquisitionHandles @3 :List(AcquisitionHandle);
  # Pre-declared acquisition handles referenced by result shapes and artifacts.

  # --- Execution ---

  execution @4 :Exec.Execution;
  # Near-time execution program for the controller.

  realTimeProperties @5 :RtProps.RtLoopProperties;
  # Properties of the real-time averaging loop.

  artifacts @6 :AnyPointer;
  # Backend-specific compilation artifacts, opaque to this schema.
  # The concrete struct type is the one the backend named by `metadata.createdBy` writes;
  # decode with that backend's reader.

  # --- Results ---

  resultShapes @7 :ResultShape.ResultShapeInfo;
  # Per-handle result tensor metadata and source-to-handle routing.

  timing @8 :ExecutionTiming;

  # --- Debug ---

  pulseSheet @9 :PulseSheet.PulseSheet;
  # Scheduled event timeline. For debugging and visualization only.

  # --- Coprocessors ---

  coprocessors @10 :List(CoprocessorBinding);
  # Coprocessors the experiment declared, each bound to a device-setup inventory slot.
  # Coprocessors the experiment left unmapped are absent.
}

struct Metadata {
  # Experiment metadata for identification and versioning.

  uid @0 :Text;
  # Unique identifier for this experiment instance.

  schemaVersion @1 :Text;
  # Schema version used by the producer (see `schemaVersion` const above).

  createdBy @2 :Text;
  # Producer identifier, "<backend identifier>/<backend version>".
  #
  # The backend identifier is unique across backends and selects the reader for
  # `CompiledExperiment.artifacts`.
}

struct ExecutionTiming {
  totalDuration @0 :Float64;
  # Total duration of the real-time steps in seconds.

  maxStepDuration @1 :Float64;
  # Maximum duration of the real-time steps in seconds.
}

struct ParameterEntry {
  uid    @0 :Text;
  # Parameter UID.

  values @1 :Values.NumericArray;
  # The values array, serialized once for both execution and result shapes.
}

struct AcquisitionHandle {
  # A acquisition result handle.
  #
  # Handles are referenced by zero-based index in `CompiledExperiment.acquisitionHandles`.

  uid @0 :Text;
  # Text UID.
}

struct CoprocessorBinding {
  # An experiment-declared coprocessor, bound to a device-setup inventory slot.

  label @0 :Text;
  # Experiment-side coprocessor label, as declared in the experiment body. Keys
  # `compiled.set_coprocessor_payload(label, ...)` and error messages.

  inventoryKey @1 :Text;
  # Device-setup inventory key `label` is mapped to.

  payload @2 :Data;
  # Payload declared in the experiment body.
}
