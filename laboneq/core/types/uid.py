# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

UID = str


def make_random_uid(obj: object = None, pre: str = None) -> UID:
    uid = UID(uuid4()) if obj is None else str(id(obj))

    if obj is not None and pre is None:
        pre = obj.__class__.__name__

    if pre is not None:
        uid = f"{pre}_{uid}"

    return uid


_current_uid = 12910802


def make_repeatable_uid(obj: object = None, pre: str = None) -> UID:
    global _current_uid
    uid = UID(_current_uid)
    _current_uid += 1

    if obj is not None and pre is None:
        pre = obj.__class__.__name__

    if pre is not None:
        uid = f"{pre}_{uid}"

    return uid


make_uid_creator = make_random_uid


def make_uid(obj: object = None, pre: str = None) -> UID:
    return make_uid_creator(obj, pre)


class SwitchUuidCreator:
    def __init__(self, repeatable: bool = True, reset_uid_count: bool = False):
        self._repeatable = repeatable
        self._reset_uid_count = reset_uid_count

    def __enter__(self):
        global make_uid_creator
        self._old_creater = make_uid_creator
        make_uid_creator = make_repeatable_uid if self._repeatable else make_random_uid
        if self._reset_uid_count and self._repeatable:
            global _current_uid
            _current_uid = 12910802

    def __exit__(self, exc_type, exc_val, exc_tb):
        global make_uid_creator
        make_uid_creator = self._old_creater
