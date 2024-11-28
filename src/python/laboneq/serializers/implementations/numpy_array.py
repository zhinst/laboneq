# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Serializer/Deserializer for np.array"""

from __future__ import annotations

import io

import numpy as np
import pybase64

from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    SerializationOptions,
    DeserializationOptions,
    JsonSerializableType,
)


@serializer(types=np.ndarray, public=True)
class NumpyArraySerializer(VersionedClassSerializer[np.ndarray]):
    SERIALIZER_ID = "laboneq.serializers.implementations.NumpyArraySerializer"
    VERSION = 1

    @staticmethod
    def _encode_npy(array):
        """Encode a numpy array as base-64 .npy data."""
        f = io.BytesIO()
        np.lib.format.write_array(f, array, version=(3, 0), allow_pickle=False)
        return pybase64.b64encode(f.getvalue()).decode("ascii")

    @staticmethod
    def _decode_npy(npy_binary):
        """Decode base-64 .npy data."""
        f = io.BytesIO(pybase64.b64decode(npy_binary.encode("ascii")))
        return np.lib.format.read_array(f)

    @classmethod
    def to_json_dict(
        cls, obj: np.ndarray, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": cls._encode_npy(obj),
        }

    @classmethod
    def to_dict(
        cls, obj: np.ndarray, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return obj

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> np.ndarray:
        return cls._decode_npy(serialized_data["__data__"])
