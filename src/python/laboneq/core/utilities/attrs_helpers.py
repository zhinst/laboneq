# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q helpers for working with attrs."""

from __future__ import annotations

import typing

import attrs
import typeguard


def validate_type(instance: object, attribute: attrs.Attribute, value: object) -> None:
    """An attrs validator that checks the value against the supplied type hint.

    Arguments:
        instance:
            The object the attribute belongs to.
        attribute:
            The attribute being set.
        value:
            The value to validate.

    Raises:
        TypeError:
            If the supplied value is not of the type given in the attributes
            type hint.

    !!! note

        The type check is currently performed using `typeguard.check_type` from
        the [typeguard](https://typeguard.readthedocs.io/) library.
    """
    attrs.resolve_types(type(instance))
    try:
        typeguard.check_type(value, attribute.type)
    except typeguard.TypeCheckError as err:
        raise TypeError(
            f"{attribute.name} must be of type {attribute.type} but received: {value!r}"
        ) from err


def validated_field(**kw) -> attrs.Field:
    """An attrs field with type hint validation.

    Calls `attrs.field` with the supplied keyword arguments,
    adding the `validate_type` function to the front of the
    list of validators.

    Arguments:
        **kw:
            Keyword arguments passed to `attrs.field`.

    Returns:
        A new `attrs.Field` instance.
    """
    validator = kw.pop("validator", None)
    if validator is None:
        validator = [validate_type]
    elif isinstance(validator, typing.Sequence):
        validator = [validate_type] + list(validator)
    else:
        validator = [validate_type, validator]
    return attrs.field(**kw, validator=validator)
