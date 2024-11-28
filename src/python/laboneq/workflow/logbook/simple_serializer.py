# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A very basic serializer for use in the folder store's log."""

from __future__ import annotations

import datetime
from functools import singledispatch

from laboneq.dsl.session import Session
from laboneq.workflow.timestamps import utc_now

# A sentinel representing a large or complex value that should
# be saved to disk instead:
NOT_SIMPLE = object()

# A string to be returned if an object should not be serialized
# at all.
DONT_SERIALIZE = "..."


@singledispatch
def simple_serialize(obj: object) -> object:
    """Serialize an object.

    Arguments:
        obj:
            The object to serialize.
    """
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_none(obj: type(None)):
    return obj


@simple_serialize.register
def simple_serialize_int(obj: int):
    return obj


@simple_serialize.register
def simple_serialize_float(obj: float):
    return obj


@simple_serialize.register
def simple_serialize_bool(obj: bool):
    return obj


@simple_serialize.register
def simple_serialize_str(obj: str):
    if len(obj) < 1000:
        return obj
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_datetime(obj: datetime.datetime):
    return str(utc_now(obj))


@simple_serialize.register
def simple_serialize_date(obj: datetime.date):
    return str(obj)


@simple_serialize.register
def simple_serialize_dict(obj: dict):
    if len(obj) < 10 and all(isinstance(k, str) and len(k) < 1000 for k in obj):
        simple_dict = {k: simple_serialize(v) for k, v in obj.items()}
        if all(v is not NOT_SIMPLE for v in simple_dict.values()):
            return simple_dict
    return NOT_SIMPLE


@simple_serialize.register
def simple_serialize_session(obj: Session):
    return DONT_SERIALIZE
