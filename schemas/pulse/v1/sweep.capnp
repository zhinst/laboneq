# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xc3aa2a6667731ae6;

using Common = import "common.capnp";

# ========================================================================================
# Sweep Parameters

struct SweepParameter {
  # A sweep parameter definition.
  #
  # Defines an array of values that are iterated when a `SweepSection` references this
  # parameter. Multiple parameters can be co-swept (iterated in lockstep) within the
  # same `SweepSection` by listing them together; all co-swept parameters must have
  # matching array lengths.

  uid @0 :Text;
  # Text UID.

  axisName @1 :Text;
  # Optional axis name for result labeling, e.g. "amplitude", "frequency".

  union {
    none @2 :Void;

    linear @3 :LinearSweep;
    # Linearly-spaced values from start to stop.

    explicitValues @4 :ExplicitSweep;
    # Explicitly-provided array of values.
  }
}

# ========================================================================================
# Sweep Value Specifications

struct LinearSweep {
  # Linearly-spaced sweep values.
  #
  # Generates `count` evenly-spaced values from `start` to `stop` (inclusive). Both
  # endpoints must be the same type (both real or both complex).

  start :union {
    none @0 :Void;
    real @1 :Float64;
    complex @2 :Common.ComplexValue;
  }

  stop :union {
    # Stop value (inclusive). Must match the type of `start`.

    none @3 :Void;
    real @4 :Float64;
    complex @5 :Common.ComplexValue;
  }

  count @6 :UInt32;
  # Number of points. Must be >= 1.
}

struct ExplicitSweep {
  # Explicitly-listed sweep values.

  union {
    realValues @0 :List(Float64);
    # Real-valued sweep points.

    complexValues @1 :List(Common.ComplexValue);
    # Complex-valued sweep points.

    intValues @2 :List(Int64);
    # Integer-valued sweep points.
  }
}
