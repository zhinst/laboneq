# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Union
from typing_extensions import deprecated

import attrs

from laboneq.core.path import remove_logical_signal_prefix
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.device import DeviceSetup, LogicalSignalGroup
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.experiment import ExperimentSignal


@classformatter
@attrs.define(kw_only=True)
class QuantumParameters:
    """Calibration parameters for a [QuantumElement][laboneq.dsl.quantum.quantum_element.QuantumElement]."""

    def copy(self):
        """Returns a copy of the parameters."""
        return self.replace()

    def replace(self, **changes: dict[str, object]):
        """Return a new set of parameters with changes applied.

        Arguments:
            changes:
                Parameter changes to apply passed as keyword arguments.
                Dotted key names such as `a.b.c` update nested parameters
                or items within parameter values that are dictionaries.

        Return:
            A new parameters instance.
        """
        invalid_params = self._get_invalid_param_paths(**changes)
        if invalid_params:
            raise ValueError(
                f"Update parameters do not match the qubit "
                f"parameters: {invalid_params}",
            )

        return self._nested_evolve(self, **changes)

    def _get_invalid_param_paths(self, **changes) -> list[str]:
        invalid_params = []
        for param_path in changes:
            keys = param_path.split(".")
            obj = self
            for key in keys:
                if isinstance(obj, dict):
                    if key not in obj:
                        invalid_params.append(param_path)
                        break
                    obj = obj[key]
                elif not hasattr(obj, key):
                    invalid_params.append(param_path)
                    break
                else:
                    obj = getattr(obj, key)
        return invalid_params

    @classmethod
    def _dict_evolve(cls, d, **changes) -> dict[str, object]:
        return {**d, **changes}

    @classmethod
    def _nested_evolve(cls, obj, **changes) -> dict | QuantumParameters:
        obj_changes = {}
        nested_changes = defaultdict(dict)

        # separate object and nested changes
        for path, value in changes.items():
            key, _, sub_path = path.partition(".")
            if sub_path:
                nested_changes[key][sub_path] = value
            else:
                obj_changes[key] = value

        # apply nested changes:
        for key, value in nested_changes.items():
            sub_obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
            obj_changes[key] = cls._nested_evolve(sub_obj, **value)

        if isinstance(obj, dict):
            return cls._dict_evolve(obj, **obj_changes)
        return attrs.evolve(obj, **obj_changes)


def _signals_converter(signals: dict[str, str | LogicalSignal] | None, self_):
    """Convert signals to a mapping from experiment signal names to logical signals."""
    if signals is None:
        return {}
    alias = self_.SIGNAL_ALIASES
    return {
        alias.get(k, k): v.path if isinstance(v, LogicalSignal) else v
        for k, v in signals.items()
    }


def _parameters_converter(parameters: dict[str, Any] | QuantumParameters | None, self_):
    """Convert parameters to the required QuantumParameters type."""
    if parameters is None:
        return self_.PARAMETERS_TYPE()
    if isinstance(parameters, dict):
        return self_.PARAMETERS_TYPE(**parameters)
    return parameters


@classformatter
@attrs.define()
class QuantumElement:
    """A quantum element within a quantum device.

    For example, a qubit or a coupler.

    Attributes:
        uid:
            A unique identifier for the quantum element. E.g. `q0`.
        signals:
            A mapping for experiment signal names to logical signal paths.
        parameters:
            Parameter values for the quantum element. For example, qubit
            frequency, drive pulse lengths.

    Class attributes:
        PARAMETERS_TYPE:
            The type of the parameters used. Should be a sub-class of QuantumParameters.
        REQUIRED_SIGNALS:
            A tuple of experiment signal names that must be provided.
        OPTIONAL_SIGNALS:
            A tuple of optional signal names. Optional signals are used by the class
            but not required.
        SIGNAL_ALIASES:
            A mapping from alternative names for signals to their canonical names.
            Signal names are translated to canonical signal names when a
            quantum element is instantiated.

            This attribute is provided as a means to provide backwards
            compatibility with existing device configurations when signal naming
            conventions change.
    """

    # Class attributes: These cannot have type hints otherwise attrs will treat
    #                   them as instance attributes.

    PARAMETERS_TYPE = QuantumParameters

    REQUIRED_SIGNALS = ()

    OPTIONAL_SIGNALS = ()

    SIGNAL_ALIASES = {}

    # Instance attributes: These should have type hints and preferably be
    #                      assigned to `attrs.field(...)`.

    uid: str = attrs.field()
    signals: Dict[str, str] = attrs.field(
        converter=attrs.Converter(_signals_converter, takes_self=True),
        default=None,
    )
    parameters: QuantumParameters = attrs.field(
        converter=attrs.Converter(_parameters_converter, takes_self=True),
        default=None,
    )

    @signals.validator
    def _validate_signals(self, attribute: attrs.Attribute, value: dict[str, str]):
        invalid_signals = {
            k: v
            for k, v in value.items()
            if not isinstance(k, str) or not isinstance(v, str)
        }
        if invalid_signals:
            raise ValueError(
                f"Signals must map experiment signal names to logical signal paths."
                f" The following entries are not strings: {invalid_signals!r}"
            )

        missing_signals = [k for k in self.REQUIRED_SIGNALS if k not in value]
        if missing_signals:
            raise ValueError(
                f"The following required signals are absent: {missing_signals!r}"
            )

        allowed_signals = set(self.REQUIRED_SIGNALS + self.OPTIONAL_SIGNALS)
        unknown_signals = [k for k in value if k not in allowed_signals]
        if unknown_signals:
            raise ValueError(
                f"The following unknown signals are present: {unknown_signals!r}."
            )

    @parameters.validator
    def _validate_parameters(
        self, attribute: attrs.Attribute, value: QuantumParameters
    ):
        if not isinstance(value, self.PARAMETERS_TYPE):
            raise ValueError(
                f"The parameters must be an instance of {self.PARAMETERS_TYPE!r}"
            )

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} uid={self.uid!r}>"

    def __rich_repr__(self):
        yield "uid", self.uid
        yield "signals", self.signals
        yield "parameters", self.parameters

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: QuantumParameters | dict[str, Any] | None = None,
    ) -> QuantumElement:
        """Create a quantum element from a logical signal group.

        Arguments:
            uid:
                A unique identifier for the quantum element. E.g. `q0`.
            lsg:
                The logical signal group containing the quantum elements
                signals.
            parameters:
                A dictionary of quantum element parameters or an instance of
                QuantumParameters.

        Returns:
            A quantum element.

        !!! note
            The logical signal group prefix, `/logical_signal_group/`,
            will be removed from any signal paths if it is present.
        """
        signals = {
            name: remove_logical_signal_prefix(signal.path)
            for name, signal in lsg.logical_signals.items()
        }
        return cls(uid=uid, signals=signals, parameters=parameters)

    @classmethod
    def from_device_setup(
        cls,
        device_setup: DeviceSetup,
        qubit_uids: list[str] | None = None,
        parameters: dict[str, QuantumParameters | dict[str, Any]] | None = None,
    ) -> list[QuantumElement]:
        """Create a list of quantum elements from a device setup.

        Arguments:
            device_setup:
                The device setup to create the quantum elements from.
            qubit_uids:
                The set of logical signal group uids to create qubits from, or
                `None` to create qubits from all logical signal groups.
            parameters:
                A dictionary mapping quantum element UIDs to the parameters for
                that quantum element. Parameters may be specified either as
                a dictionary of parameter names and values, or as instances of
                the appropriate sub-class of `QuantumParameters`.

        Returns:
            A list of quantum elements.

        !!! version-changed "Changed in version 2.46.0

            The `parameters` argument was added.

            In earlier versions, parameters needed to be set separately
            after the quantum elements were created.
        """
        if qubit_uids is not None:
            qubit_uids = set(qubit_uids)
        else:
            qubit_uids = device_setup.logical_signal_groups.keys()
        if parameters is None:
            parameters = {}
        return [
            cls.from_logical_signal_group(
                uid, lsg, parameters=parameters.get(uid, None)
            )
            for uid, lsg in device_setup.logical_signal_groups.items()
            if uid in qubit_uids
        ]

    def calibration(self) -> Calibration:
        """Return the calibration for this quantum element.

        Calibration for each experiment signal is generated from the quantum element
        parameters.

        Returns:
            The experiment calibration.
        """
        return Calibration({})

    def experiment_signals(
        self,
    ) -> List[ExperimentSignal]:
        """Return the list of the experiment signals for this quantum element.

        Returns:
            A list of experiment signals.
        """
        return [ExperimentSignal(uid=k, map_to=k) for k in self.signals.values()]

    def copy(self):
        """Returns a copy of the qubit."""
        return self.replace()

    def replace(self, **parameter_changes: dict[str, object]) -> QuantumElement:
        """Return a new quantum element with the parameter changes applied.

        Arguments:
            parameter_changes:
                Parameter changes to apply passed as keyword arguments.

        Return:
            A new quantum element with the parameter changes applied.
        """
        params = self.parameters.replace(**parameter_changes)
        return attrs.evolve(self, parameters=params)

    def update(self, **parameter_changes: dict[str, object]):
        """Update this quantum element with supplied parameter changes.

        Arguments:
            parameter_changes:
                Parameter changes to apply passed as keyword arguments.

        !!! note
            The `parameters` attribute of this quantum element is replaced
            with a *new* parameter instance.
        """
        params = self.parameters.replace(**parameter_changes)
        self.parameters = params

    @classmethod
    def create_parameters(cls, **parameters: dict[str, object]) -> QuantumParameters:
        """Create a new instance of parameters for this qubit class.

        Arguments:
            parameters:
                Parameter values for the new parameter instance.

        Returns:
            A new parameter instance.
        """
        return cls.PARAMETERS_TYPE(**parameters)

    @classmethod
    @deprecated(
        "The .load method is deprecated. Use `q = laboneq.serializers.load(filename)` instead.",
        category=FutureWarning,
    )
    def load(cls, filename: Union[str, bytes, os.PathLike]) -> "QuantumElement":
        """
        Loads a QuantumElement object from a JSON file.

        Args:
            filename: The name of the JSON file to load the QuantumElement object from.

        !!! version-changed "Deprecated in version 2.43.0."
            Use `q = laboneq.serializers.load(filename)` instead.
        """
        from laboneq import serializers

        return serializers.load(filename)

    @deprecated(
        "The .save method is deprecated. Use `laboneq.serializers.save(q, filename)` instead.",
        category=FutureWarning,
    )
    def save(self, filename: Union[str, bytes, os.PathLike]):
        """
        Save a QuantumElement object to a JSON file.

        Args:
            filename: The name of the JSON file to save the QuantumElement object.

        !!! version-changed "Deprecated in version 2.43.0."
            Use `laboneq.serializers.save(q, filename)` instead.
        """
        from laboneq import serializers

        serializers.save(self, filename)
