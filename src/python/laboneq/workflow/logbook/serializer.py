# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Logbook serializer for artifacts.

See the functions below named `serialize_` for the list of types that can be serialized.
"""

from __future__ import annotations

import abc
from functools import singledispatch
from typing import TYPE_CHECKING

import matplotlib.figure as mpl_figure
import numpy as np
import PIL

if TYPE_CHECKING:
    from typing import IO


class SerializationNotSupportedError(RuntimeError):
    pass


class SerializeOpener(abc.ABC):
    """An protocol allowing serializers to open files and access options.

    Serializers need to write out potentially multiple files without
    knowing precisely where the files will end up.

    Simultaneously the caller of serialize (e.g. the logbook) needs to
    keep a record of which files were created.

    This class allows all of this to happen by abstracting away the file
    creation interface.
    """

    @abc.abstractmethod
    def open(
        self,
        ext: str,
        *,
        encoding: str | None = None,
        suffix: str | None = None,
        description: str | None = None,
        binary: bool = False,
    ) -> IO:
        """Return an open file handle.

        Arguments:
            ext:
                The file extension to use (without a starting period).
            encoding:
                The encoding to use for text files.
            suffix:
                A suffix to add to the name of the file before the extension.
                This allows serializers that save multiple files to distinguish
                the files saved in a human-readable fashion.
            description:
                A description of the file contents. For example, a serializer
                saving a figure might save a `.png` file with the description
                "The plotted figure." and a `.json` file with the description
                "Metadata for the plot.".
            binary:
                If true, files are opened for writing in binary mode. If false,
                the default, files are opened in text mode.
        """

    @abc.abstractmethod
    def options(self) -> dict:
        """Return the serialization options."""

    @abc.abstractmethod
    def name(self) -> str:
        """Return a name for the object being serialized."""


@singledispatch
def serialize(obj: object, opener: SerializeOpener) -> None:
    """Serialize an object.

    Arguments:
        obj:
            The object to serialize.
        opener:
            A `SerializeOpener` for retrieving options and opening
            files to write objects to.
    """
    raise SerializationNotSupportedError(
        f"Type {type(obj)!r} not supported by the serializer [name: {opener.name()}]."
    )


@serialize.register
def serialize_str(obj: str, opener: SerializeOpener) -> None:
    """Serialize a Python `str` object.

    String objects are saved as a text file with extension `.txt` and UTF-8 encoding.

    No options are supported.
    """
    with opener.open("txt", encoding="utf-8") as f:
        f.write(obj)


@serialize.register
def serialize_bytes(obj: bytes, opener: SerializeOpener) -> None:
    """Serialize a Python `bytes` object.

    Bytes objects are saved as a binary file with extension `.dat`.

    No options are supported.
    """
    with opener.open("dat", binary=True) as f:
        f.write(obj)


@serialize.register
def serialize_pil_image(obj: PIL.Image.Image, opener: SerializeOpener) -> None:
    """Serialize a PIL image.

    PIL images are saved with `PIL.Image.save`.

    The format to save in is passed in the `format` option which defaults to `png`.
    The format `jpg` is automatically converted to `jpeg` for PIL.

    The remaining options are passed directly to `PIL.Image.save` as keyword
    arguments.
    """
    options = opener.options()
    ext = options.pop("format", "png")

    # Determine the PIL image format from the extension:
    image_format = ext.upper()
    if image_format == "JPG":
        image_format = "JPEG"

    with opener.open(ext, binary=True) as f:
        obj.save(f, format=image_format, **options)


@serialize.register
def serialize_matplotlib_figure(
    obj: mpl_figure.Figure,
    opener: SerializeOpener,
) -> None:
    """Serialize a matplotlib Figure.

    Matplotlib figures are saved with `matplotlib.figure.Figure.savefig`.

    The format to save in is passed in the `format` option which defaults to `png`.

    The remaining options are passed as the `pil_kwargs` argument to `.savefig`.
    """
    options = opener.options()
    ext: str = options.pop("format", "png")
    bbox = options.pop("bbox_inches", "tight")

    with opener.open(ext, binary=True) as f:
        obj.savefig(f, format=ext, bbox_inches=bbox, pil_kwargs=options)


@serialize.register
def serialize_numpy_array(obj: np.ndarray, opener: SerializeOpener) -> None:
    """Serialize a NumPy `ndarray`.

    NumPy arrays are saved with `numpy.save` and the extension `.npy`.

    Any options are passed directly as keyword arguments to `.save`.
    """
    with opener.open("npy", binary=True) as f:
        np.save(f, obj, allow_pickle=False, **opener.options())


from laboneq.dsl.experiment.experiment import Experiment
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.quantum import QPU, QuantumElement, Transmon
from laboneq.dsl.result.results import Results
from laboneq.workflow.tasks import RunExperimentResults
from laboneq.serializers.implementations.quantum_element import QuantumElementContainer
from laboneq.workflow import TaskOptions, WorkflowOptions


@serialize.register(CompiledExperiment)
@serialize.register(DeviceSetup)
@serialize.register(Experiment)
@serialize.register(QPU)
@serialize.register(QuantumElement)
@serialize.register(Transmon)
@serialize.register(Results)
@serialize.register(RunExperimentResults)
@serialize.register(QuantumElementContainer)
@serialize.register(TaskOptions)
@serialize.register(WorkflowOptions)
def serialize_laboneq_object(obj: object, opener: SerializeOpener) -> None:
    """Serialize LabOne Q types."""
    from laboneq.serializers import to_json

    with opener.open("json", binary=True) as f:
        f.write(to_json(obj))


@serialize.register
def serialize_list(obj: list, opener: SerializeOpener) -> None:
    """Serialize lists."""
    if all(isinstance(x, QuantumElement) for x in obj):
        serialize_laboneq_object(QuantumElementContainer(obj), opener)
    else:
        # TODO: We should be more careful about what we try serialize
        #       using numpy (e.g. lists of LabOne Q objects)
        #       Perhaps we can have a better heuristic than this.
        serialize_numpy_array(obj, opener)
