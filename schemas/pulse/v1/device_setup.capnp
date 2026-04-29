# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xabd22f51d0b05b7b;

using Calibration = import "calibration.capnp";

struct DeviceSetup {
  instruments @0 :List(Instrument);
  # Instruments of the setup

  signals @1 :List(DeviceSignal);
  oscillators @2 :List(Calibration.Oscillator);
}

struct Instrument {
  # A physical instrument (device).

  uid @0 :Text;
  # Unique identifier of the device.

  deviceType @1 :Text;
  # Device type string.
  #   "HDAWG", "SHFQA", "SHFSG", "SHFQC", "UHFQA", "PQSC", "QHUB", "SHFPPC", "ZQCS"

  options @2 :List(Text);
  # Device options
  # For "HDAWG", "SHFQA", "SHFSG", "SHFQC", "UHFQA" instruments the first value should be the model,
  # e.g. "SHFSG8"

  referenceClockSource @3 :ReferenceClock;
  # The instrument's reference clock source.

  physicalDeviceUid @4 :UInt16; 
  # TODO: Temporary migration field, shall be removed
  isShfqc @5 :Bool; 
  # TODO: Temporary migration field, shall be removed
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
  # A device-level signal with port mapping and calibration.
  #
  # DeviceSignals represent the physical ports on instruments. ExperimentSignals
  # are mapped to DeviceSignals via SignalMapping.

  uid @0 :Text;

  ports @1 :List(Text);
  # Physical port paths for this signal. Format depends on hardware generation:
  #   Gen1/Gen2: device port path (e.g. "SIGOUTS/0", "QACHANNELS/1")
  #   ZQCS: geolocation (e.g. "1:3:1:2" for shelf:slot:frontend:port)
  # Single-channel signals carry one entry; IQ signals carry two.

  calibration @2 :Calibration.SignalCalibration;
  # Calibration of the signal.

  instrumentUid @3 :Text;
  # Owning instrument UID.

  channelType @4 :Text;
  # Optional signal type hint for validation (e.g. "IQ", "INTEGRATION", "RF").
  # The compiler infers the actual type from device metadata; this is a hint only.
}
