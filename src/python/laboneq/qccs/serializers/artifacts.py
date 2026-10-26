# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Converts `ArtifactsCodegen` to and from the parts of `CodegenArtifactsQccs`."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import orjson

from laboneq.serializers.implementations._models._artifacts_qccs import (
    deserialize_artifacts_qccs,
    serialize_artifacts_qccs,
)

if TYPE_CHECKING:
    from laboneq.data.artifacts_qccs import ArtifactsCodegen


def artifacts_to_capnp_parts(
    artifacts: ArtifactsCodegen,
) -> tuple[bytes, list[dict[str, str | bytes]]]:
    """Splits `artifacts` into UTF-8 JSON of every field except `src`."""
    json = orjson.dumps(
        serialize_artifacts_qccs(replace(artifacts, src=None)),
        option=orjson.OPT_SERIALIZE_NUMPY,
    )
    return json, artifacts.src or []


def artifacts_from_capnp_parts(
    json: bytes, src: list[dict[str, str | bytes]]
) -> ArtifactsCodegen:
    """Joins the parts made by `artifacts_to_capnp_parts` into `ArtifactsCodegen`."""
    artifacts = deserialize_artifacts_qccs(orjson.loads(json))
    artifacts.src = src
    return artifacts
