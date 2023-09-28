# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q serialization support for external package objects."""
from __future__ import annotations

from laboneq._optional_deps import (
    HAS_XARRAY,
    Xarray,
    XarrayDataArray,
    XarrayDataset,
    import_optional,
)


class XarrayDataArrayDeserializer:
    """Deserializer for `xarray.DataArray`."""

    _type_ = "xarrayDataArray"

    def __new__(cls, mapping: dict) -> XarrayDataArray:
        xr: Xarray = import_optional(
            "xarray", "Cannot deserialize `xarray` object with the package installed."
        )
        return xr.DataArray.from_dict(mapping)


class XarrayDatasetDeserializer:
    """Deserializer for `xarray.Dataset`."""

    _type_ = "xarrayDataset"

    def __new__(cls, mapping: dict) -> XarrayDataset:
        xr: Xarray = import_optional(
            "xarray", "Cannot deserialize `xarray` object with the package installed."
        )
        return xr.Dataset.from_dict(mapping)


def serialize_maybe_xarray(
    obj: object,
    serializer_function,
    entity_classes,
    entities_collector,
    emit_enum_types,
    omit_none_fields,
) -> dict | None:
    """Serialize `xarray` object.

    Checks if the object is from the optional `xarray` package and
    serialized it.

    Returns:
        Serialized object in JSON format if `xarray` is installed and the object
        belongs to it.
        `None` otherwise.
    """
    # To avoid circular imports
    from laboneq.core.serialization.simple_serialization import NumpyArrayRepr

    if HAS_XARRAY:
        import xarray as xr

        if obj.__class__ == xr.DataArray:
            as_dict = obj.to_dict()
            as_dict["__type"] = XarrayDataArrayDeserializer._type_
            as_dict["data"] = serializer_function(
                NumpyArrayRepr(array_data=obj.data),
                entity_classes,
                entities_collector,
                emit_enum_types,
                omit_none_fields,
            )
            for coord_name in obj.coords:
                as_dict["coords"][coord_name]["data"] = serializer_function(
                    NumpyArrayRepr(array_data=obj.coords[coord_name].data),
                    entity_classes,
                    entities_collector,
                    emit_enum_types,
                    omit_none_fields,
                )
            return as_dict

        if obj.__class__ == xr.Dataset:
            as_dict = obj.to_dict()
            as_dict["__type"] = XarrayDatasetDeserializer._type_
            for data_var_name in obj.data_vars:
                as_dict["data_vars"][data_var_name]["data"] = serializer_function(
                    NumpyArrayRepr(array_data=obj.data_vars[data_var_name].data),
                    entity_classes,
                    entities_collector,
                    emit_enum_types,
                    omit_none_fields,
                )
            for coord_name in obj.coords:
                as_dict["coords"][coord_name]["data"] = serializer_function(
                    NumpyArrayRepr(array_data=obj.coords[coord_name].data),
                    entity_classes,
                    entities_collector,
                    emit_enum_types,
                    omit_none_fields,
                )
            return as_dict
    return None
