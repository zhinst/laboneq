# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import functools
from enum import Enum
from typing import Iterable, Mapping

import numpy as np


@functools.lru_cache()
def _dir(cls):
    return dir(cls)


class DataHelper:
    @classmethod
    def accept_visitor(cls, data, visitor, visited=None):
        """
        Accepts a visitor function that is called for each element in the data structure.
        This can be used on any data structure that is a combination of lists, dicts, and objects.
        """
        if visited is None:
            visited = set()

        if id(data) in visited:
            return

        visited.add(id(data))

        if data is None:
            return

        if isinstance(data, str) or isinstance(data, float) or isinstance(data, int):
            return

        if isinstance(data, Enum):
            return

        if isinstance(data, Mapping) and not data:
            return

        visitor(data)

        if isinstance(data, np.ndarray):
            data = data.tolist()

        if isinstance(data, list):
            for item in data:
                cls.accept_visitor(item, visitor, visited, visited)
            return

        mapping = {}
        dir_list = _dir(data.__class__)
        is_object = False
        if hasattr(data, "__dict__"):
            is_object = True
            mapping = data.__dict__
        elif isinstance(data, Mapping):
            mapping = data

        for k, v in mapping.items():
            outkey = k
            outvalue = v
            if k[0] == "_" and is_object:
                # This is a private attribute, so we don't want to access it,
                # but if it is a property, we still want to access it
                if k[1:] in dir_list:
                    outkey = k[1:]
                    # we get the property value
                    outvalue = getattr(data, outkey)
                else:
                    # really private, so we don't access it
                    continue

            if isinstance(outvalue, Mapping):
                cls.accept_visitor(v, visitor, visited)

            elif isinstance(outvalue, Iterable) and not isinstance(outvalue, str):
                for item in outvalue:
                    cls.accept_visitor(item, visitor, visited)
            else:
                cls.accept_visitor(outvalue, visitor, visited)

    @classmethod
    def generate_uids(cls, data):
        # traverse all objects in the data structure
        # if the data items have a uid field and it is none, we generate a uid
        # we count up uid's per type
        # we need to make sure that we don't generate a uid which was already set before
        # first collect all uids which have been set

        uids = set()
        cls.accept_visitor(
            data, lambda x: uids.add(x.uid) if hasattr(x, "uid") else None
        )
        # remove None from the set
        uids.discard(None)

        # now we generate uids for all items which don't have a uid yet
        uid_counter = {}

        def uid_generator_visitor(item):
            if hasattr(item, "uid") and item.uid is None:
                if item.__class__ not in uid_counter:
                    uid_counter[item.__class__] = 0
                uid_counter[item.__class__] += 1
                while True:
                    uid = f"{item.__class__.__name__}_{uid_counter[item.__class__]}"
                    if uid not in uids:
                        uids.add(uid)
                        item.uid = uid
                        break

        cls.accept_visitor(data, uid_generator_visitor)
