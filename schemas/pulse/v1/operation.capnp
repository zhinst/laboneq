# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xad248f6ba75b927b;

using Common = import "common.capnp";

# ========================================================================================
# Operations

struct Operation {
  # A single operation within a section.
  #
  # Operations are the leaf-level actions in the experiment timing hierarchy. Each
  # operation acts on one or more signals and executes atomically within its parent
  # section's time span.

  union {
    none @0 :Void;
    # (Catches when someone forgets to specify the operation kind. Do not set this.)

    play @1 :PlayOp;
    # Play a pulse on a signal.

    delay @2 :DelayOp;
    # Insert an explicit delay on a signal.

    reserve @3 :ReserveOp;
    # Reserve a signal without producing output.

    acquire @4 :AcquireOp;
    # Perform an acquisition (measurement).

    call @5 :CallOp;
    # Invoke a near-time callback.

    setNode @6 :SetNodeOp;
    # Set an instrument node value at near-time.

    resetOscillatorPhase @7 :ResetOscillatorPhaseOp;
    # Reset oscillator phase to zero.
  }
}

# ========================================================================================
# Play

struct PlayOp {
  # Play a pulse on a signal.
  #
  # Plays a waveform defined by the referenced pulse, with optional per-operation
  # overrides for amplitude, phase, length, and shape parameters. The pulse must
  # reference a defined `Pulse` in the experiment.

  signal @0 :Common.Id;
  # `Experiment.signals` index. Must map to an output-capable port.

  pulse @1 :Common.Id = .Common.noneId;
  # Optional `Pulse` index to play (`Common.noneId` means no pulse).

  amplitude @2 :Common.Value;
  # Amplitude scaling (real or complex). Sweepable. Magnitude should not exceed 1.0.

  phase @3 :Common.Value;
  # Phase offset in radians. Sweepable.

  incrementOscillatorPhase @4 :Common.Value;
  # Increment the oscillator phase by this amount (radians). Sweepable.
  # Mutually exclusive with `setOscillatorPhase`.

  setOscillatorPhase @5 :Common.Value;
  # Set the oscillator phase to this absolute value (radians). Sweepable.
  # Mutually exclusive with `incrementOscillatorPhase`.

  length @6 :Common.Value;
  # Override pulse length in seconds. Sweepable. When set, stretches or compresses the
  # waveform to fit.

  markers @7 :Markers;
  # Marker signal configuration. See `Markers`.

  pulseParameters @8 :List(Common.ValueEntry);
  # Pulse shape parameter overrides, e.g. `{"sigma": 12e-9}`. These override the
  # default parameters defined in the `Pulse` definition.
}

struct Markers {
  # Marker signal configuration for a play operation.
  #
  # Configures up to two digital marker outputs synchronized with the pulse.

  marker1 @0 :MarkerSpec;
  marker2 @1 :MarkerSpec;
}

struct MarkerSpec {
  # Configuration for a single marker output.
  #
  # Markers are digital signals synchronized with pulse playback. They can be simple
  # on/off signals or custom waveforms.

  enable @0 :Bool;

  start :union {
    # Offset from pulse start in seconds.

    none @1 :Void;
    value @2 :Float64;
  }

  length :union {
    # Marker duration in seconds.

    none @3 :Void;
    value @4 :Float64;
  }

  waveform @5 :Common.Id = .Common.noneId;
  # Optional custom marker waveform `Pulse` index. When set, uses the pulse waveform
  # instead of a simple digital on/off signal.
}

# ========================================================================================
# Delay and Reserve

struct DelayOp {
  # Insert an explicit delay on a signal.
  #
  # Other signals may continue executing in parallel during the delay.

  signal @0 :Common.Id;
  # `Experiment.signals` index.

  time @1 :Common.Value;
  # Delay duration in seconds. Sweepable.

  precompensationClear @2 :Bool;
  # Reset the precompensation filter state during the delay. Use this when a long idle
  # period would cause the filter state to accumulate drift.
}

struct ReserveOp {
  # Reserve a signal without producing output.
  #
  # Prevents other operations from using the signal during this time slot. Useful for
  # maintaining timing alignment across signals without generating any waveform.

  signal @0 :Common.Id;
  # `Experiment.signals` index.
}

# ========================================================================================
# Acquire

struct AcquireOp {
  # Perform an acquisition on a signal.
  #
  # Acquires and integrates the signal using one or more integration kernels. For
  # multi-state discrimination, multiple kernels distinguish between quantum states.
  # The result is stored under `handle` and can be referenced by `MatchSection` for
  # real-time feedback.

  signal @0 :Common.Id;
  # `Experiment.signals` index. Must map to an acquisition-capable port.

  handle @1 :Common.Id;
  # `Experiment.acquisitionHandles` index. Can be referenced by `MatchSection` for
  # real-time feedback.

  kernels @2 :List(Common.Id);
  # Integration kernel `Experiment.pulses` indices. One kernel per state for
  # multi-state discrimination.

  length @3 :Common.Value;
  # Explicit integration length in seconds. Overrides the kernel's own length.

  kernelParameters @4 :List(PulseParameterMap);
  # Per-kernel pulse parameter overrides. Each entry corresponds to the kernel at the
  # same index in `kernels`.
}

struct PulseParameterMap {
  # Pulse parameter overrides for a single integration kernel.

  parameters @0 :List(Common.ValueEntry);
}

# ========================================================================================
# Near-Time Operations

struct CallOp {
  # Invoke a near-time callback.
  #
  # Callbacks are opaque to the compiler and resolved by the Controller at execution
  # time. The compiler includes this operation in its output, and the Controller invokes
  # the callback between real-time sequences. This is used for operations like instrument
  # reconfiguration or classical computation that cannot run on-device.

  callbackId @0 :Text;
  # Callback identifier. Should be namespaced to avoid collisions.

  arguments @1 :List(Common.ValueEntry);
  # Arguments passed to the callback. Each argument can be a constant or a sweep
  # parameter reference.
}

struct SetNodeOp {
  # Set an instrument node value at near-time.
  #
  # Directly sets a device node (e.g. oscillator frequency, gain) between real-time
  # sequences. The Controller executes this between shots.

  path @0 :Text;
  # Device node path.

  value @1 :Common.Value;
  # Value to set. Sweepable.
}

struct ResetOscillatorPhaseOp {
  # Reset oscillator phase to zero.

  signal @0 :Common.Id = .Common.noneId;
  # Signal whose oscillator phase should be reset. `Common.noneId` resets all
  # oscillators in the current execution context.
}

# ========================================================================================
# Acquisition Types

enum AcquisitionType {
  # Type of acquisition to perform. Determines how the raw ADC data is processed
  # on-device before returning results.

  unspecified @0;
  # Defaults to integration.

  raw @1;
  # Raw time-domain acquisition. Returns unprocessed waveform data.

  integration @2;
  # Integration with kernel(s). Returns integrated complex values.

  discrimination @3;
  # Multi-state discrimination. Returns discrete state indices.

  spectroscopyIq @4;
  # Spectroscopy with IQ demodulation.

  spectroscopyPsd @5;
  # Spectroscopy with power spectral density.
}
