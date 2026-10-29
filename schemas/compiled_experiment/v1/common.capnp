# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xc427c81f3306586f;

using Id = UInt32;
# Canonical type for all numeric references in the schema.

const noneId :Id = 0xffffffff;
# Sentinel for optional references. This value is outside valid zero-based
# index range for practical experiment sizes.

# ========================================================================================
# Scalar Types

struct ComplexValue {
  # A complex scalar value with real and imaginary parts.

  real @0 :Float64;
  imag @1 :Float64;
}

struct Constant {
  # A constant value (one of several scalar types).
  #
  # Used inside `Value` to represent a fixed (non-swept) quantity.

  union {
    real          @0 :Float64;
    complex       @1 :ComplexValue;
    integer       @2 :Int64;
    stringValue   @3 :Text;
    rawBytesValue @4 :Data;
    # Arbitrary binary data.

    pythonValue @5 :Data;
    # Serialized Python object, for custom functional pulse parameters or user callback
    # function arguments whose type isn't one of the other `Constant` variants.
    # Values that cannot be JSON-serialized are rejected. Producers needing an arbitrary
    # opaque payload should use `rawBytesValue` with a producer/consumer-agreed encoding instead.

    boolean @6 :Bool;
  }
}

# ========================================================================================
# Parametric Values

struct Value {
  # A value that is either a constant or a reference to a named parameter.
  #
  # Used for any quantity that can be controlled by a named parameter. When set to
  # `parameterRef`, the actual value is determined at each near-time step from the
  # referenced entry in `CompiledExperiment.parameters`.
  #
  # Example: an amplitude field might be a fixed `constant` of 0.5, or a `parameterRef`
  # pointing to a parameter that varies from 0.0 to 1.0.

  union {
    none @0 :Void;
    # No value specified. The consumer should use its default.

    constant @1 :Constant;
    # A fixed value.

    parameterRef @2 :Id;
    # `CompiledExperiment.parameters` index. The value is resolved at each near-time step.
  }
}

# ========================================================================================
# Map Entry Types

struct StringEntry {
  # Key-value pair for string-typed maps.

  key   @0 :Text;
  value @1 :Text;
}

struct ValueEntry {
  # Key-value pair for `Value`-typed maps.
  #
  # Used for pulse shape parameter overrides and callback arguments, where each
  # named parameter can be either a constant or a parameter reference.

  key   @0 :Text;
  value @1 :Value;
}

# ========================================================================================
# Near-time Step Key

using NearTimeStepKey = List(UInt32);
