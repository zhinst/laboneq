# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xec4efd1c1c8d5a6c;

# Canonical type for all numeric references in the schema.
using Id = UInt32;

# Sentinel for optional references. This value is outside valid zero-based
# index range for practical experiment sizes.
const noneId :Id = 0xffffffff;

# ========================================================================================
# Scalar Types

struct ComplexValue {
  # A complex scalar value with real and imaginary parts.

  real @0 :Float64;
  imag @1 :Float64;
}

# ========================================================================================
# Sweepable Values

struct Value {
  # A value that is either a constant or a reference to a sweep parameter.
  #
  # Used for any quantity that can be varied during a parameter sweep. When set to
  # `parameterRef`, the actual value is determined at each sweep step from the
  # referenced `SweepParameter`.
  #
  # Example: an amplitude field might be a fixed `real` of 0.5, or a `parameterRef`
  # pointing to a sweep parameter that varies from 0.0 to 1.0.

  union {
    none @0 :Void;
    # No value specified. The consumer should use its default.

    real @1 :Float64;
    complex @2 :ComplexValue;
    integer @3 :Int64;
    boolean @4 :Bool;
    stringValue @5 :Text;

    rawBytesValue @6 :Data;
    # Arbitrary binary data.

    pythonValue @7 :Data;
    # Serialized Python object, for custom functional pulse parameters or user callback
    # function arguments whose type isn't one of the other variants. Values that cannot be
    # JSON-serialized are rejected. Producers needing an arbitrary opaque payload should use
    # `rawBytesValue` with a producer/consumer-agreed encoding instead.

    parameterRef @8 :Id;
    # `Experiment.sweepParameters` index. The value is resolved at each sweep step.
  }
}

# ========================================================================================
# Map Entry Types

struct StringEntry {
  # Key-value pair for string-typed maps.

  key @0 :Text;
  value @1 :Text;
}

struct ValueEntry {
  # Key-value pair for `Value`-typed maps.
  #
  # Used for pulse shape parameter overrides and callback arguments, where each
  # named parameter can be either a constant or a sweep reference.

  key @0 :Text;
  value @1 :Value;
}
