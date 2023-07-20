# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import Dict, Set


class QueryTracker:
    """Tracks the queries made to the parameter store.

    This is useful for checking that all parameters are used.
    """

    def __init__(self):
        self._queries = set()

    def notify(self, key):
        self._queries.add(key)

    def queries(self):
        return self._queries


class ParameterStore(dict):
    """Key-value store for the sweep parameters.

    Tracks the parameters that are actually used in the experiment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_trackers: Set[weakref.ReferenceType[QueryTracker]] = set()

    def create_tracker(self):
        tracker = QueryTracker()
        self._query_trackers.add(weakref.ref(tracker))
        return tracker

    def _notify_trackers(self, key):
        for tracker_ref in self._query_trackers:
            tracker = tracker_ref()
            if tracker is not None:
                tracker.notify(key)
        self._query_trackers = {t for t in self._query_trackers if t() is not None}

    def __getitem__(self, item):
        self._notify_trackers(item)
        return super().__getitem__(item)

    def get(self, item, default=None):
        self._notify_trackers(item)
        super().get(item, default)

    def frozen(self):
        return frozenset(self.items())

    def __iter__(self):
        """To avoid all items being (incorrectly) flushed to the query tracker, we must
        not allow iteration over the parameter store.
        """
        raise NotImplementedError("ParameterStore is not iterable")

    def __contains__(self, item):
        self._notify_trackers(item)
        return super().__contains__(item)

    @contextmanager
    def extend(self, other: Dict):
        """Extend the parameter store with the parameters from another store.

        This is useful for tracking the parameters used in a sub-block.
        """

        assert self.keys().isdisjoint(other.keys())
        self.update(other)
        yield
        for key in other:
            self.pop(key)
