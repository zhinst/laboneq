# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xf00580752be0804c;

using Common = import "common.capnp";


enum VarType {
  unspecified @0;
  int8 @1;
  int16 @2;
  int32 @3;
  int64 @4;
  uint8 @5;
  uint16 @6;
  uint32 @7;
  uint64 @8;
  phase @9;
  frequency @10;
  amplitude @11;
  discriminationDataPacked @12;
  iqDataPacked @13;
  scopeShot @14;
  waveformUpdate @15;
}

struct Coprocessor {
  # Experiment-scoped coprocessor handle. Referenced by zero-based index in
  # `Experiment.coprocessors`.

  label @0 :Text;
  # User-chosen local label. Keys `compiled.set_coprocessor_payload(label, ...)`
  # and error messages.

  payload @1 :Data;
  # Opaque kernel payload. Absent when unset.

  inventoryKey @2 :Text;
  # Device-setup inventory key that this handle is bound to. Absent when unmapped.
}

struct VariableValue {
  union {
    intValue @0 :Int64;
    floatValue @1 :Float64;
    phaseRadians @2 :Float64;
    frequencyHz @3 :Float64;
    amplitude @4 :Float64;
  }
}

struct Variable {
  # Typed runtime-valued DSL variable; the target of inbound stream-field
  # bindings. Referenced by zero-based index in `Experiment.variables`.

  type @0 :VarType;

  name @1 :Text;
  # Optional user-facing name for error messages. Absent when unset.

  initial @2 :VariableValue;
  # Initial value. Absent when the variable starts undefined.

  logHandle @3 :Text;
  # Results-capture handle. Absent when not logged.
}

struct StructField {
  name @0 :Text;

  type @1 :VarType;

  binding :union {
    unbound @2 :Void;
    # Outbound literal-at-send-time field, or inbound field nobody consumes.

    handles @3 :List(Common.Id);
    # Outbound acquisition field: `Experiment.acquisitionHandles` indices
    # feeding this field.

    variable @4 :Common.Id;
    # Inbound scalar field: `Experiment.variables` index it updates.

    pulse @5 :Common.Id;
    # Inbound waveform field: `Experiment.pulses` index of the Pulse it updates.
  }
}

struct Stream {
  # A declared, typed, directional packet stream.

  uid @6 :Text;
  # User-assigned identifier. Synthesized as `stream_<index>` when the user
  # leaves it unset, so it is always present.

  src :union {
    controlSystem @0 :Void;
    coprocessor @1 :Common.Id;
    # `Experiment.coprocessors` index.
  }

  dst :union {
    controlSystem @2 :Void;
    coprocessor @3 :Common.Id;
  }

  link @4 :Text;

  fields @5 :List(StructField);
}

enum CmpOp {
  unspecified @0;
  eq @1;
  ne @2;
  lt @3;
  le @4;
  gt @5;
  ge @6;
}

struct Predicate {
  # Runtime condition: the exit condition of a do-until loop, or the assertion of an
  # `ensure`.

  union {
    unspecified @0 :Void;

    comparison :group {
      # Value-based: `do_until(condition=(var != 0), ...)`.

      variable @1 :Common.Id;
      # `Experiment.variables` index (left-hand side).

      op @2 :CmpOp;

      rhs @3 :VariableValue;
      # Right-hand side literal.
    }

    isLive :group {
      # Arrival-based: `do_until(is_live(x), ...)`, `ensure(is_live(x))`.

      union {
        variable @4 :Common.Id;
        pulse @5 :Common.Id;
        # `Experiment.pulses` index.
      }
    }
  }
}
