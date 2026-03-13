# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xa069b2eda25218db;

using Common = import "common.capnp";

# ========================================================================================
# Pulse Definitions

struct Pulse {
  # A pulse (waveform) definition.
  #
  # Pulses are defined once at the experiment level and referenced by index from `PlayOp`
  # (for output pulses) and `AcquireOp` (for integration kernels). The same pulse
  # definition can be played multiple times with different per-operation parameter
  # overrides.

  uid @0 :Text;
  # Text UID.

  length :union {
    # Pulse duration in seconds. Must be > 0 when specified.
    #
    # Required for functional pulses. Optional for sampled pulses, where
    # the duration is derived from the sample count and sampling rate.

    none @1 :Void;
    value @2 :Float64;
  }

  amplitude @3 :Common.ComplexValue;
  # Default complex amplitude. Magnitude should not exceed 1.0. Can be overridden
  # per-operation in `PlayOp.amplitude`.

  canCompress @4 :Bool;
  # Whether the compiler may deduplicate identical waveform segments. When true, the
  # compiler can replace repeated constant-amplitude regions with shorter waveforms
  # and sequencer loops, saving waveform memory.

  union {
    none @5 :Void;

    sampled @6 :SampledPulse;
    # Pre-sampled waveform data.

    functional @7 :FunctionalPulse;
    # External sampler function called at compile time.
  }
}

# ========================================================================================
# Sampled Pulses

struct SampledPulse {
  # A pulse defined by pre-computed waveform samples.

  samples @0 :WaveformData;

  sampleType @1 :SampleType;
  # Whether samples are real-valued or complex (I/Q).
}

enum SampleType {
  # Sample data type for sampled pulses.

  unspecified @0;

  real @1;
  # Real-valued samples.

  complex @2;
  # Complex I/Q samples (interleaved real and imaginary components).
}

struct WaveformData {
  # Waveform sample data container.

  union {
    none @0 :Void;

    inline @1 :InlineData;
    # Inline waveform data embedded directly in the message.
  }
}

struct InlineData {
  # Inline waveform data. Recommended for waveforms smaller than 64 KB.

  data @0 :Data;
  # Raw sample bytes, little-endian. For complex data, real and imaginary components
  # are interleaved.

  sampleCount @1 :UInt64;
  # Number of samples. For complex data, this is the number of complex samples (not
  # the number of float values).

  dataType @2 :WaveformDataType;
}

enum WaveformDataType {
  # Numeric format of waveform sample data.
  #
  # All formats use little-endian byte order. Complex formats interleave real and
  # imaginary components per sample.

  unspecified @0;

  float32 @1;
  # 32-bit IEEE 754 floating point (real).

  float64 @2;
  # 64-bit IEEE 754 floating point (real).

  complex64 @3;
  # Two 32-bit floats (real, imag) per sample. 8 bytes per sample.

  complex128 @4;
  # Two 64-bit floats (real, imag) per sample. 16 bytes per sample.
}

# ========================================================================================
# Functional Pulses

struct FunctionalPulse {
  # A pulse defined by an external sampler function.
  #
  # The compiler calls the sampler at compile time to generate the waveform. The sampler
  # contract is: `sample(length, sampling_rate, parameters) -> ComplexArray`.
  #
  # Non-Python clients must pre-sample custom shapes and use `SampledPulse` instead.

  samplerUri @0 :Text;
  # Sampler URI. Supported schemes:
  #   "py://function_name" for in-process Python samplers (PyO3)
  #   "http://" or "https://" for REST API samplers (future)

  parameters @1 :List(Common.ValueEntry);
  # Parameters passed to the sampler function. These can be constants or sweep
  # parameter references.
}
