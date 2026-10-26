# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Backend-owned mutable state of a single experiment execution."""

from __future__ import annotations


class BackendRuntimeState:
    """Base for whatever state a backend (not an individual device!) carries across
    the near-time steps of a run."""
