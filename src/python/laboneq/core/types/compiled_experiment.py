# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from laboneq.core.exceptions import LabOneQException

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment


@dataclass(frozen=True)
class ResultProperties:
    """Data structure containing some information about measurement results.

    Attributes:
        shape: The shape of the result array.
        axis_names: This tuple has the same length as `shape`, and each entry describes what
                    the respective axis corresponds to. Usually they are ids of sweep parameters,
                    but can also contain other things - "samples" in case of raw acquisition, and the id of
                    the RT acquisition loop in case of single-shot readout. A dimension can also have more
                    than 1 associated names, if it corresponds to a parallel sweep over multiple parameters.
    """

    shape: tuple[int, ...]
    axis_names: tuple[str | tuple[str, ...], ...]


@dataclass(init=True, repr=True, order=True)
class CompiledExperiment:
    """Data structure to store the output of the compiler.

    Attributes:
        device_setup (DeviceSetup):
            The device setup the experiment was compiled for.
        experiment (Experiment):
            The (uncompiled) experiment.
        scheduled_experiment (internal):
            Internal. The internal representation of the compiled
            experiment. Available for debugging but subject to
            change in any LabOne Q release.

    !!! version-changed "Changed in version 26.10.0"
        The `scheduled_experiment` constructor argument was removed. Obtain a
        `CompiledExperiment` by compiling an experiment.

    !!! version-removed "Removed in version 26.10.0"
        The `.experiment_dict` attribute, deprecated since version 2.14.0,
        was removed. Use `.experiment` instead.

    !!! version-changed "Changed in version 2.54.0"
        The following deprecated methods for saving and loading were removed:

        - `load`
        - `save`

        Use the `load` and `save` functions from the `laboneq.simple` module instead.

    !!! version-changed "Changed in version 2.56.0"
        The `.device_setup` and `.experiment` attributes were no longer deprecated.

    !!! version-changed "Deprecated in version 2.51.0"
        The `.device_setup` and `.experiment` attributes were
        deprecated in version 2.51.0 and will be removed in a
        future release. Manage and track the device setup and experiment
        separately if they are needed.

        Note that the `.device_setup` and `.experiment` were no longer
        deprecated in version 2.56.0.

    !!! version-changed "Changed in version 2.14.0"
        The `.scheduled_experiment` attribute was documented to
        be internal and subject to change.
    """

    # Internal: the compiler's serialized output; `scheduled_experiment` is the Python
    # view of it. Excluded from `repr` because it is large and opaque.
    _compiler_output: bytes = field(repr=False)

    # The source device setup.
    device_setup: DeviceSetup | None = None

    # The source experiment.
    experiment: Experiment | None = None

    # Settings passed to the compiler
    compiler_settings: dict[str, Any] | None = None

    # Coprocessor payloads attached after compilation, keyed by the coprocessor label
    # declared in the experiment.
    coprocessor_payload_overrides: dict[str, bytes] = field(default_factory=dict)

    @property
    def estimated_runtime(self) -> float:
        """An estimation of the total runtime of the experiment in seconds.

        DISCLAIMER: This estimation does not include any overhead from network, IO,
        or python runtime.
        """
        return self.scheduled_experiment.total_execution_time

    @property
    def result_properties(self) -> dict[str, ResultProperties]:
        """Properties of results for each handle."""
        return {
            handle: ResultProperties(
                shape=properties.shape,
                axis_names=tuple(
                    x if isinstance(x, str) else tuple(x) for x in properties.axis_names
                ),
            )
            for handle, properties in self.scheduled_experiment.result_shape_info.shapes.items()
        }

    @cached_property
    def scheduled_experiment(self) -> ScheduledExperiment:
        """The internal representation of the compiled experiment.

        Available for debugging but subject to change in any LabOne Q release.
        """
        from laboneq.core import compiler_output

        return compiler_output.deserialize(self._compiler_output)

    def set_coprocessor_payload(self, coprocessor_label: str, payload: bytes) -> None:
        """Attach a payload to a coprocessor, post-compile.

        Takes precedence over the payload declared in the experiment body with
        the DSL function of the same name.

        Args:
            coprocessor_label: The coprocessor label, as declared with
                `@experiment(coprocessors=[...])`.
            payload: Opaque to LabOne Q.

        Raises:
            TypeError: If *payload* is not `bytes`.
            LabOneQException: If the label is not a coprocessor of this
                compiled experiment.
        """
        if not isinstance(payload, bytes):
            message = (
                f"The coprocessor payload must be bytes, not {type(payload).__name__}."
            )
            raise TypeError(message)
        mappings = self.scheduled_experiment.coprocessor_mappings
        if coprocessor_label not in mappings:
            raise LabOneQException(
                f"'{coprocessor_label}' is not a coprocessor of this compiled "
                f"experiment. Available: {sorted(mappings)}."
            )
        self.coprocessor_payload_overrides[coprocessor_label] = payload

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is not CompiledExperiment:
            return NotImplemented
        return (
            other.device_setup,
            other.experiment,
            other._compiler_output,
            other.coprocessor_payload_overrides,
        ) == (
            self.device_setup,
            self.experiment,
            self._compiler_output,
            self.coprocessor_payload_overrides,
        )


def compiler_output_bytes(compiled_experiment: CompiledExperiment) -> bytes:
    """LabOne Q internal use only.

    Get the compiler output bytes from a `CompiledExperiment`.

    Args:
        compiled_experiment: The compiled experiment.

    Returns:
        The compiler output bytes.
    """
    return compiled_experiment._compiler_output
