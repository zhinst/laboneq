# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

from laboneq.core.exceptions import LabOneQException


class UidGenerator:
    """An generator for unique identifiers for DSL objects within the context of
    an experiment.

    The unique identifiers generated have the form:

        `{prefix}_{count}`

    where the count for a particular prefix starts at zero and
    is incremented each time a prefix is used.

    One `UidGenerator` is created per `ExperimentContext` and is used to
    generated identifiers for objects within that experiment.

    Advantages of this approach over random identifiers are that these identifiers
    are human readable and allow for generating experiments with identical
    identifiers in a straight-forward manner.

    In addition, there is one global generator, `GLOBAL_UID_GENERATOR`,
    that is used when the `ExperimentContext` is not accessible. For
    example, if a `Section` is created before the experiment.

    The `GLOBAL_UID_GENERATOR` instead generates identifiers with
    an additional `__` at the start:

        `__{prefix}_{count}`

    So that the globally generated UIDs never conflict with the UIDs
    generated within an `ExperimentContext`, no prefix may start with `__`.

    Arguments:
        is_global:
            Whether this is the global unique identifier
            generator. The `GLOBAL_UID_GENERATOR` in this
            module should be the only instance created with
            `is_global=True`.

    Attributes:
        GLOBAL_NAMESPACE:
            The additional prefix used by the `GLOBAL_UID_GENERATOR`.
            It's value is `"__"`.
    """

    GLOBAL_NAMESPACE: str = "__"

    def __init__(self, is_global: bool = False):
        self.is_global = is_global
        self._uid_counts = defaultdict(int)

    def _reset(self):
        """Reset the all prefix counts used by the generator to zero."""
        self._uid_counts.clear()

    def uid(self, prefix: str) -> str:
        """Generate a unique identifier with the given prefix.

        Arguments:
            prefix:
                The prefix used to generate the unique identifier.
        """
        if prefix.startswith(self.GLOBAL_NAMESPACE):
            raise LabOneQException(
                f"Experiment context UID prefixes may not start with {self.GLOBAL_NAMESPACE!r}"
            )
        if self.is_global:
            prefix = f"{self.GLOBAL_NAMESPACE}{prefix}"
        count = self._uid_counts[prefix]
        self._uid_counts[prefix] += 1
        return f"{prefix}_{count}"


GLOBAL_UID_GENERATOR = UidGenerator(is_global=True)


def reset_global_uid_generator():
    """Reset prefix counts in the `GLOBAL_UID_GENERATOR` generator.

    This should be used with care and is primarily ended to be called before
    unit tests are run to allow global identifiers to be generated predictably.

    No single experiment should contain UIDS from both before and after a call
    to this function.
    """
    GLOBAL_UID_GENERATOR._reset()
