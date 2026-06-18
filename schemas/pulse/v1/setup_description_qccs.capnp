# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xb3feeb8a3c8e97a3;

struct SetupDescriptionQccs {
  # QCCS hardware setup description

  instruments @0 :List(Instrument);
  # Instruments of the setup

  signals @1 :List(DeviceSignal);
  # Physical channels in the setup

  internalConnections @2 :List(InternalConnection);
  # Internal connections in the setup.
}

struct Instrument {
  # A physical instrument (device).

  uid @0 :Text;
  # Unique identifier of the device.

  deviceType @1 :Text;
  # Device type string.
  #   "HDAWG", "SHFQA", "SHFSG", "SHFQC", "UHFQA", "PQSC", "QHUB", "SHFPPC"

  options @2 :List(Text);
  # Device options
  # For "HDAWG", "SHFQA", "SHFSG", "SHFQC", "UHFQA" instruments the first value should be the model,
  # e.g. "SHFSG8"

  referenceClockSource @3 :ReferenceClock;
  # The instrument's reference clock source.
}

enum ReferenceClock {
  unspecified @0;
  # Decision is deferred to controller or not applicable

  internal @1;
  # Use instrument's own internal reference clock.

  external @2;
  # The reference clock is external, e.g. a PQSC instrument.
}

struct DeviceSignal {
  # A device-level signal with port mapping.
  #
  # DeviceSignals represent the physical ports on instruments. ExperimentSignals
  # are mapped to DeviceSignals.

  uid @0 :Text;
  # UID of the signal. 
  # The value can be mapped to ExperimentSignals.

  ports @1 :List(Text);
  # Physical port paths for this signal. (e.g. "SIGOUTS/0", "QACHANNELS/1")
  # Single-channel signals carry one entry; IQ signals carry two.

  instrumentUid @2 :Text;
  # Owning instrument UID.
}

struct InternalConnection {
  # A physical connection between ports on two instruments in the setup.
  #
  # Internal connections represent hardware wiring between instruments.
  # Examples:
  #   - SHFPPC pump output (PPCHANNELS/0) driving an SHFQC acquire channel (QACHANNELS/0/INPUT)

  fromInstrument @0 :Text;
  # UID of the source instrument.

  fromPort @1 :Text;
  # Port path on the source instrument (e.g. "PPCHANNELS/0").

  toInstrument @2 :Text;
  # UID of the destination instrument.

  toPort @3 :Text;
  # Port path on the destination instrument (e.g. "QACHANNELS/0/INPUT").
}
