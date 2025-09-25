# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A very basic serializer for use in the folder store's log."""

from __future__ import annotations

import datetime
from functools import singledispatch
from typing import Union

from typing_extensions import TypeAlias

from laboneq.dsl.session import Session
from laboneq.workflow.timestamps import utc_now


# A sentinel representing a large or complex value that should
# be saved to disk instead:
class NotSimpleType:
    pass


NOT_SIMPLE = NotSimpleType()

# A string to be returned if an object should not be serialized
# at all.
DONT_SERIALIZE = "..."

# A constant specifying the maximum number of items a container
# may have in order to be considered simple
MAX_CONTAINER_ITEMS = 10

# A constant specifying the maximum string length that is considered
# simple
MAX_STR_LENGTH = 1000

SimpleType: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    None,
    NotSimpleType,
    dict[str, "SimpleType"],
    list["SimpleType"],
    tuple["SimpleType"],
]


@singledispatch
def simple_serialize(obj: object) -> SimpleType:
    """Serialize an object.

    Arguments:
        obj:
            The object to serialize.
    """
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_none(obj: type(None)) -> SimpleType:
    return obj


@simple_serialize.register
def simple_serialize_int(obj: int) -> SimpleType:
    return obj


@simple_serialize.register
def simple_serialize_float(obj: float) -> SimpleType:
    return obj


@simple_serialize.register
def simple_serialize_complex(obj: complex) -> SimpleType:
    return {"real": obj.real, "imag": obj.imag}


@simple_serialize.register
def simple_serialize_bool(obj: bool) -> SimpleType:
    return obj


@simple_serialize.register
def simple_serialize_str(obj: str) -> SimpleType:
    if len(obj) <= MAX_STR_LENGTH:
        return obj
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_datetime(obj: datetime.datetime) -> SimpleType:
    return str(utc_now(obj))


@simple_serialize.register
def simple_serialize_date(obj: datetime.date) -> SimpleType:
    return str(obj)


@simple_serialize.register
def simple_serialize_list(obj: list) -> SimpleType:
    if len(obj) <= MAX_CONTAINER_ITEMS:
        simple_list = [simple_serialize(v) for v in obj]
        if all(v is not NOT_SIMPLE for v in simple_list):
            return simple_list
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_tuple(obj: tuple) -> SimpleType:
    if hasattr(obj, "_asdict"):
        # support namedtuples nicely:
        return simple_serialize_dict(obj._asdict())
    return simple_serialize_list(obj)


@simple_serialize.register
def simple_serialize_dict(obj: dict) -> SimpleType:
    if len(obj) <= MAX_CONTAINER_ITEMS and all(
        isinstance(k, str) and len(k) < 1000 for k in obj
    ):
        simple_dict = {k: simple_serialize(v) for k, v in obj.items()}
        if all(v is not NOT_SIMPLE for v in simple_dict.values()):
            return simple_dict
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_session(obj: Session):
    return DONT_SERIALIZE
