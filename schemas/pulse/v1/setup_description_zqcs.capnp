# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

@0x9e24ab8bad80c970;

# ZQCS setup description.
#
# The structured description lives in the `laboneq_zqcs` backend and is
# threaded through the compiler as an opaque blob.

struct SetupDescriptionZqcs {
  data @0 :Data;
  # Backend-defined BLOB; only the matching backend decodes it.
}
