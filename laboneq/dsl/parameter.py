# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import ArrayLike

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

parameter_id = 0


def parameter_id_generator():
    global parameter_id
    retval = f"par{parameter_id}"
    parameter_id += 1
    return retval


def _compare_nested(a, b):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        if not (isinstance(b, list) or isinstance(b, np.ndarray)):
            return False
        if not len(a) == len(b):
            return False
        return all(map(lambda x: _compare_nested(x[0], x[1]), zip(a, b)))
    return a == b


@classformatter
@dataclass(init=True, repr=True, order=True)
class Parameter(ABC):
    """Parent class for sweep parameters in a LabOne Q Experiment.

    Attributes:
        uid (str):
            A unique ID for the parameter. If not supplied,
            one will be automatically generated.
    """

    uid: str = field(default_factory=parameter_id_generator)


class _ParameterArithmeticMixin(NDArrayOperatorsMixin):
    """A mixin that implments arithmetic using numpy's ufunc hooks.

    Classes that include this mixin should provide a `.values`
    attribute or property that gives an [ArrayLike][] containing
    the values to be swept.
    """

    values: ArrayLike

    def __array__(self, dtype=None):
        """numpy __array__ hook"""
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """numpy __array_ufunc__ hook"""
        # The out argument is not support because Parameters
        # are immutable. Operations are performed out-of-place
        # instead.
        kwargs.pop("out", None)
        if kwargs:
            # In general, the implementation does not yet support any
            # ufunc keyword arguments, but support might be added
            # for cases that prove useful.
            return NotImplemented

        if method == "__call__":
            new_inputs = []
            driven_by = []
            for input in inputs:
                if isinstance(input, self.__class__):
                    new_inputs.append(input.values)
                    driven_by.append(input)
                else:
                    new_inputs.append(input)

            values = ufunc(*new_inputs)
            return SweepParameter(values=values, driven_by=driven_by)
        else:
            return NotImplemented


@classformatter
@dataclass(init=True, repr=True, order=True)
class SweepParameter(_ParameterArithmeticMixin, Parameter):
    """An arbitrary sweep parameter.

    Attributes:
        values (ArrayLike):
            An arbitrary numpy array whose values are used as the sweep
            parameter.
        axis_name (str):
            The name of the sweep axis for this parameter used in the results.
            If this argument is not defined, the uid of the parameter will be
            used. Default `None`.
        driven_by (list[SweepParameter]):
            Optional and usually absent. If given, specifies the list of
            [SweepParameter][laboneq.dsl.parameter.SweepParameter] objects that
            this one is derived from. See the notes below for an example.
            Parameters should have the same shape as the ones they are driven
            by. Incorrect shapes will raise a [ValueError][]. Default `None`.

    Examples:
        The `driven_by` parameter is automatically set on the parameters created
        when we apply arithmetic operations. For example:

        >>> triple_param = 3 * param

        creates a new sweep parameter `triple_param` that is driven by `param`.

        Similarly, one may apply other numpy operations such as:

        >>> sin_param = np.sin(param)

        One may also manually create more complex derived parameters:

        >>> param = SweepParameter(np.linspace(0, np.pi, 10))
        >>> sin_param = SweepParameter(
                values=np.sin(param.values),
                driven_by=[param],
            )

        A sweep parameter may also be driven by multiple parameters, as is the
        case when adding parameters:

        >>> param = param_1 + param_2

        which creates a new parameter `param` that is the sum of `param_1` and
        `param_2` and is driven by both.

        When a sweep parameter is driven by multiple others, the others must all
        be swept simultaneously.

        Operations that create parameters of different shapes to the ones they
        are driven by will be rejected by raising a [ValueError][].

    !!! version-changed "Changed in 2.14.0"
        Support for applying numpy ufuncs (e.g. ``np.sin``) was added.
    """

    values: ArrayLike = field(default=None)
    axis_name: str = field(default=None)
    driven_by: list[SweepParameter] | None = field(default=None)

    def __post_init__(self):
        if self.driven_by:
            self_shape = np.shape(self.values)
            other_shapes = [np.shape(other.values) for other in self.driven_by]
            if any(other_shape != self_shape for other_shape in other_shapes):
                raise ValueError(
                    "Arithmetic and other operations on SweepParameters"
                    " and LinearSweepParameters should return values with"
                    " the same shape as those of the parameters they"
                    " are driven_by. Refusing to create parameter with"
                    f" shape {self_shape} that is driven by parameters"
                    f" with shapes: {other_shapes}."
                )

    def __eq__(self, other):
        if self is other:
            return True
        return (
            self.axis_name == other.axis_name
            and _compare_nested(self.values, other.values)
            and self.driven_by == getattr(other, "driven_by", None)
        )

    def __len__(self) -> int:
        return len(self.values)


@classformatter
@dataclass(init=True, repr=True, order=True)
class LinearSweepParameter(_ParameterArithmeticMixin, Parameter):
    """A linear sweep parameter.

    The parameter is swept through the values generated by
    `numpy.linspace(start, stop, count)`.

    Attributes:
        start (Number):
            The starting value of the parameter sweep.
        stop (Number):
            The final value of the parameter sweep.
        count (Number):
            The number of sweep steps in the parameter sweep.
        axis_name (str):
            The name of the sweep axis for this parameter used in
            the results. If this argument is not defined, the uid
            of the parameter will be used. Default `None`.

    Examples:
        As with [SweepParameter][laboneq.dsl.parameter.SweepParameter] one
        can perform arithmetic operations on linear sweep parameters:

        >>> param = 3 * linear_param
        >>> param = np.sin(param)

        See [SweepParameter][laboneq.dsl.parameter.SweepParameter] for a
        complete description.

    !!! version-changed "Changed in 2.14.0"
        Support for applying numpy ufuncs (e.g. ``np.sin``) was
        added.
    """

    # The starting value of the parameter sweep.
    start: Number = field(default=None)

    # The final value of the parameter sweep.
    stop: Number = field(default=None)

    # The number of sweep steps in the parameter sweep.
    count: int = field(default=None)

    # The name of the sweep axis for this parameter used in the results.
    #
    # If this argument is not defined, the uid of the object will be used instead.
    axis_name: str = field(default=None)

    def __eq__(self, other):
        if self is other:
            return True
        return self.axis_name == other.axis_name and _compare_nested(
            self.values, other.values
        )

    def __post_init__(self):
        if self.count is None or self.start is None or self.stop is None:
            raise RuntimeError(
                f"LinearSweepParameter {self.uid}: one of start, stop, count is None"
            )

    def __len__(self) -> int:
        return self.count

    @property
    def values(self) -> ArrayLike:
        """The values swept by the parameter."""
        return np.linspace(start=self.start, stop=self.stop, num=self.count)
