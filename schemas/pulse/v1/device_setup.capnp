# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xabd22f51d0b05b7b;

struct DeviceSetup {
  # The hardware setup an experiment targets.

  backend @0 :Text;
  # Backend identifier, unique across backends. Selects the reader that decodes
  # `setupDescription`, e.g. "qccs".

  setupDescription @1 :AnyPointer;
  # Backend-specific setup description, opaque to this schema.
  # The concrete struct type is the one the backend named by `backend` defines;
  # decode with that backend's reader.
}
