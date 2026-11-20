# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0x9d2d4a2c98bdf1e1;

using Common = import "common.capnp";
using Values = import "values.capnp";

struct ResultShapeInfo {
  shapes @0 :List(ResultShapeEntry);

  explicitValues @1 :List(Values.NumericArray);
  # Deduplicated explicit axis-value arrays (e.g. PRNG loop samples), referenced by
  # `AxisParameter.explicitRef`.
}

struct ResultShapeEntry {
  handle @0 :Common.Id;
  # `CompiledExperiment.acquisitionHandles` index.

  shape @1 :HandleResultShape;
}

struct HandleResultShape {
  shape @0 :List(UInt64);
  # Tensor extents for the result associated with one handle.

  axes @1 :List(Axis);
  # Axis descriptors, one entry per dimension.

  chunkedAxisIndex :union {
    # Index of the chunked axis. Absent when no axis is chunked.

    none  @2 :Void;
    value @3 :UInt32;
  }

  matchCaseMask @4 :List(MatchCaseMaskEntry);
  # Axis index -> selected row indices for match/case acquisitions.
  # Empty list means no mask.
}

struct Axis {
  parameters @0 :List(AxisParameter);
  # One entry per parameter swept on this axis.
  # Non-parallel: exactly one entry. Parallel: multiple entries.
}

struct AxisParameter {
  name @0 :Text;
  # Name of the axis.

  union {
    parameterRef @1 :Common.Id;
    # `CompiledExperiment.parameters` index.

    shots @2 :UInt64;
    # The axis is a plain `0..shots` index count -- e.g. averaging-loop iterations or
    # merged-handle indices -- with no backing parameter or explicit values. Unpacking
    # `0..shots` into concrete index values, if needed, is left to the consumer.

    explicitRef @3 :Common.Id;
    # `ResultShapeInfo.explicitValues` index.

    rawSamples :group {
      # The axis is the `0..count` sample index of a raw acquisition, acquired at
      # `samplingRate` Sa/s.

      count @4 :UInt64;
      samplingRate @5 :Float64;
    }
  }
}

struct MatchCaseMaskEntry {
  axis @0 :UInt32;
  rows @1 :List(UInt64);
}
