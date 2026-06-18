# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0x9e24ab8bad80c970;

struct SetupDescriptionZqcs {
  # ZQCS setup description.

  data @0 :Data;
  # The full description of the ZQCS system.

  uid @1 :Text;
  # Unique identifier of the system.

  channels @2 :List(ChannelConfig);
  # Channel configuration.
}

struct ChannelConfig {
  # Channel configuration

  geolocation @0 :Text;
  # Channel geolocation, e.g. "1:2:3:4".
  # The value can be mapped to ExperimentSignals.

  channelType @1 :ChannelType;
}

enum ChannelType {
  rf   @0;
  qa   @1;
  flux @2;
}
