# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0xabd22f51d0b05b7b;

using Qccs = import "setup_description_qccs.capnp";
using Zqcs = import "setup_description_zqcs.capnp";

struct DeviceSetup {
  setupDescription :union {
    qccs @0 :Qccs.SetupDescriptionQccs;
    zqcs @1 :Zqcs.SetupDescriptionZqcs;
  }
}
