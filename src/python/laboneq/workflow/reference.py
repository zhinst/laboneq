# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Module for workflow reference."""

from __future__ import annotations

import operator
from typing import Any, Callable

from laboneq.workflow.exceptions import WorkflowError

notset = object()


def get_default(reference: Reference) -> object:
    """Get reference default value."""
    return reference._default


def unwrap(reference: Reference, value: object) -> object:
    """Unwrap the reference and any operations done on it."""
    res = value
    for op, *args in reference._ops:
        res = op(res, *args)
    return res


def get_ref(reference: Reference) -> object:
    """Return an object the reference points to."""
    return reference._ref


def are_equal(one: Reference, other: Reference) -> bool:
    """Check if two references are equal."""
    if not isinstance(one, Reference) or not isinstance(other, Reference):
        return NotImplemented
    return (
        one._ref == other._ref
        and one._head == other._head
        and one._ops == other._ops
        and one._default == other._default
        and one._overwrites == other._overwrites
    )


class Reference:
    """A reference class.

    A `Reference` is a placeholder for objects used while a workflow
    graph is being constructed.

    If tasks are the nodes in the workflow graph, then references define the edges.

    For example, in a workflow definition like:

    ```python
    @workflow
    def add_and_multiply(a, b):
        c = add_task(a, 2)
        d = multiply_task(c, 3)
    ```

    the values `a`, `b`, `c` and `d` are all references. By tracking where references
    are used, the graph builder can determine how inputs and outputs are passed between
    tasks in the graph.

    References also track some Python operations performed on them, so one can also use
    references like `a["value"]`. This creates a reference for "the item named 'value'
    from reference 'a'".

    When a workflow graph is run, the references are used to determine the input values
    to the next task from the outputs returned by the earlier tasks.

    The following operations are supported

    * __getitem__()
    * __getattr__()
    * __eq__()

    Notes on specific Python operations
    ---

    *Equality comparison*

    For equality comparison, especially with booleans, use `==` instead of `is`.
        Equality with `is` will always return `False`.

    Arguments:
        ref: An object this reference points to.
        default: Default value of `ref`.
    """

    # TODO: Which operators should be supported?
    #       To have a sensible error message on unsupported operation, on top of
    #       typical Python error message, the operators must either way be implemented.
    #       And therefore not supporting them might not be needed since they are
    #       implemented either way.
    def __init__(self, ref: object, default: object = notset):
        self._ref = ref
        self._default = default
        # Head of the reference
        self._head: Reference = self
        # Operations that was done on the reference
        self._ops: list[tuple[Callable, Any]] = []
        # A list of other references this reference overwrites
        self._overwrites: list[Reference] = []

    def _create_child_reference(
        self,
        head: Reference,
        ops: list[tuple[Callable, Any]],
    ) -> Reference:
        obj = Reference(self._ref)
        obj._head = head
        obj._ops = ops
        return obj

    def __getitem__(self, item: object):
        return self._create_child_reference(
            self._head,
            [*self._ops, (operator.getitem, item)],
        )

    def __eq__(self, other: object):
        return self._create_child_reference(
            self._head,
            [*self._ops, (operator.eq, other)],
        )

    def __getattr__(self, other: object):
        return self._create_child_reference(
            self._head,
            [*self._ops, (getattr, other)],
        )

    def __iter__(self):
        raise NotImplementedError("Iterating a workflow Reference is not supported.")

    def __repr__(self):
        return f"Reference(ref={self._ref}, default={self._default})"


def add_overwrite(one: Reference, other: Reference | object) -> None:
    """Add an overwrite to an reference."""
    if isinstance(other, Reference):
        one._overwrites.append(other)
    else:
        obj = Reference(None, other)
        one._overwrites.append(obj)


def resolve_to_value(ref: Reference, states: dict, *, only_ref: bool = False) -> Any:  # noqa: ANN401
    """Resolve reference.

    The resolve order is following:

        * Look value from `states`
        * Iterate over overwritten values and return the first one that
            exists in `states` or is constant

    Arguments:
        ref: Root reference
        states: A mapping of reference to actual values
        only_ref: Check only the root reference

    Raises:
        WorkflowError: Value cannot be resolved.
    """
    if not ref._overwrites or only_ref:
        try:
            return unwrap(ref, states[id(ref._head)])
        except KeyError as error:
            default = get_default(ref._head)
            if default != notset:
                return unwrap(ref, default)
            # Reference was never executed (e.g. undefined variable).
            msg = f"Result for '{get_ref(ref)}' is not resolved."
            raise WorkflowError(msg) from error
    else:
        to_try = [ref, *ref._overwrites]
        for x in to_try:
            ref_unwrapped = get_ref(x)
            # Constant, not from workflow construct
            if ref_unwrapped is None:
                return unwrap(x, get_default(x._head))
            if id(x._head) not in states:
                continue
            return unwrap(x, states[id(x._head)])
        return resolve_to_value(ref, states, only_ref=True)
