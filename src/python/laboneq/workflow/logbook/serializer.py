# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Logbook serializer for artifacts.

See the functions below named `serialize_` for the list of types that can be serialized.
"""

from __future__ import annotations

import abc
from functools import singledispatch, singledispatchmethod
from typing import TYPE_CHECKING

import matplotlib.figure as mpl_figure
import numpy as np
import orjson
import PIL

try:
    from lmfit.model import ModelResult as LmfitModelResult

    LMFIT_IMPORTED = True
except ImportError:
    LMFIT_IMPORTED = False


try:
    from uncertainties.core import Variable as UncertaintiesVariable
    from uncertainties.core import AffineScalarFunc as UncertaintiesAffineScalarFunc

    UNCERTAINTIES_IMPORTED = True
except ImportError:
    UNCERTAINTIES_IMPORTED = False


try:
    from sklearn.base import BaseEstimator as SklearnBaseEstimator

    SKLEARN_IMPORTED = True
except ImportError:
    SKLEARN_IMPORTED = False


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


class OrjsonSerializer:
    """A helper class for using `orjson`.

    It applies consistent settings when calling `orjson.dumps`.
    """

    # Types supported by orjson or by provided default handler:
    SUPPORTED_TYPES = (
        type(None),
        int,
        float,
        complex,
        bool,
        str,
        dict,
        list,
        tuple,
        np.integer,
        np.ndarray,
    )

    if LMFIT_IMPORTED:
        SUPPORTED_TYPES += (LmfitModelResult,)

    if UNCERTAINTIES_IMPORTED:
        SUPPORTED_TYPES += (UncertaintiesVariable, UncertaintiesAffineScalarFunc)

    if SKLEARN_IMPORTED:
        SUPPORTED_TYPES += (SklearnBaseEstimator,)

    COMPLEX_DTYPE_MAP = {
        # complex-dtype: float-view-dtype
        np.dtype(np.complex128): np.dtype(np.float64),
        np.dtype(np.complex64): np.dtype(np.float32),
    }

    def supported_type(self, obj) -> bool:
        """Return true if the *type* of the object is supported."""
        return isinstance(obj, self.SUPPORTED_TYPES)

    def supported_object(self, obj) -> bool:
        """Return true if the whole object is supported."""
        if isinstance(obj, dict):
            if not all(isinstance(k, str) for k in obj):
                return False
            return all(self.supported_object(o) for o in obj.values())
        elif isinstance(obj, (list, tuple)):
            return all(self.supported_object(o) for o in obj)
        elif isinstance(obj, np.ndarray) and obj.dtype is np.dtype(object):
            return False
        else:
            return self.supported_type(obj)

    @singledispatchmethod
    def default(self, obj) -> object:
        """A `default` handler for objects `orjson` does not support directly."""
        raise TypeError(
            f"{type(obj).__name__!r} is not supported by the logbook JSON serializer."
        )

    @default.register
    def _default_ndarray(self, obj: np.ndarray) -> object:
        """A `default` orjson handler for numpy arrays."""
        # orjson can handle most numpy arrays natively but not
        # complex or non-contiguous arrays:
        obj = np.ascontiguousarray(obj)
        if obj.dtype in self.COMPLEX_DTYPE_MAP:
            float_dtype = self.COMPLEX_DTYPE_MAP[obj.dtype]
            float_view = obj.view(dtype=float_dtype)
            return {
                "description": "Array of complex values arranged as: [real(v0), imag(v0), real(v1), imag(v1), ...]",
                "data": orjson.Fragment(
                    orjson.dumps(float_view, option=orjson.OPT_SERIALIZE_NUMPY)
                ),
            }
        else:
            return orjson.Fragment(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY))

    @default.register
    def _default_complex(self, obj: complex) -> object:
        """A `default` orjson handler for lists."""
        return {"real": obj.real, "imag": obj.imag}

    if LMFIT_IMPORTED:

        @default.register
        def _default_lmfit_model_result(self, obj: LmfitModelResult) -> object:
            """A `default` orjson handler for LmfitModelResults."""
            return obj.summary()

    if UNCERTAINTIES_IMPORTED:

        @default.register
        def _default_uncertainties_variable(self, obj: UncertaintiesVariable) -> object:
            """A `default` orjson handler for UncertaintiesVariable."""
            return {
                "value": obj.nominal_value,
                "std_dev": obj.std_dev,
                "tag": obj.tag,
            }

        @default.register
        def _default_uncertainties_affine_scalar_func(
            self, obj: UncertaintiesAffineScalarFunc
        ) -> object:
            """A `default` orjson handler for UncertaintiesAffineScalarFunc."""
            return {
                "value": obj.nominal_value,
                "std_dev": obj.std_dev,
            }

    if SKLEARN_IMPORTED:

        @default.register
        def _default_sklearn_base_estimator(self, obj: SklearnBaseEstimator) -> object:
            """A `default` orjson handler for SklearnBaseEstimator."""
            return {k: v for k, v in obj.__dict__.items() if k.endswith("_")}

    def dumps(self, obj) -> bytes:
        """Call orjson.dumps with the appropriate settings.

        Current options set are:

            * OPT_SORT_KEYS
            * OPT_SERIALIZE_NUMPY

        The method `self.default` is passed to `orjson.dump(..., default=...)`.
            * `self.default`
        """
        return orjson.dumps(
            obj,
            option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
            default=self.default,
        )

    def serialize(self, obj, opener):
        """Serialize the provided JSON-serializable object."""
        json_data = self.dumps(obj)
        with opener.open("json", binary=True) as f:
            f.write(json_data)


_ORJSON_SERIALIZER = OrjsonSerializer()


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
    if obj.dtype is np.dtype(object):
        raise SerializationNotSupportedError(
            f"NumPy arrays with dtype {obj.dtype!r} are not supported"
            f" by the serializer [name: {opener.name()}]."
        )
    with opener.open("npy", binary=True) as f:
        np.save(f, obj, allow_pickle=False, **opener.options())


if LMFIT_IMPORTED:

    @serialize.register
    def serialize_lmfit_model_result(
        obj: LmfitModelResult, opener: SerializeOpener
    ) -> None:
        """Serialize an lmfit `ModelResult`.

        `ModelResult`s are saved as JSON using their `.summary` method.
        """
        _ORJSON_SERIALIZER.serialize(obj, opener)


if SKLEARN_IMPORTED:

    @serialize.register
    def serialize_scikit_learn_base_estimator(
        obj: SklearnBaseEstimator, opener: SerializeOpener
    ) -> None:
        """Serialize scikit-learn estimators.

        Estimators parameter values are saved as JSON.
        """
        _ORJSON_SERIALIZER.serialize(obj, opener)


if UNCERTAINTIES_IMPORTED:

    @serialize.register
    def serialize_uncertainties_variable(
        obj: UncertaintiesVariable,
        opener: SerializeOpener,
    ) -> None:
        """Serialize uncertainties `Variable`.

        The `Variable` has its `value`, `std_dev` and `tag` saved as JSON.
        """
        _ORJSON_SERIALIZER.serialize(obj, opener)

    @serialize.register
    def serialize_uncertainties_affince_scalar_func(
        obj: UncertaintiesAffineScalarFunc,
        opener: SerializeOpener,
    ) -> None:
        """Serialize uncertainties `AffineScalarFunc`.

        `AfficeScalarFunc` objects are created when performing arithmetic on
        uncertainties `Variable` objects.

        The `AffineScalarFunc` has its `value`, `std_dev` and `tag` saved as JSON.
        """
        _ORJSON_SERIALIZER.serialize(obj, opener)


from laboneq.dsl.experiment.experiment import Experiment
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.quantum import QPU, QuantumParameters, QuantumElement, Transmon
from laboneq.dsl.result.results import Results
from laboneq.serializers.implementations.quantum_element import (
    QuantumParametersContainer,
    QuantumParametersDict,
    QuantumElementContainer,
    QuantumElementDict,
)
from laboneq.workflow import TaskOptions, WorkflowOptions


@serialize.register(CompiledExperiment)
@serialize.register(DeviceSetup)
@serialize.register(Experiment)
@serialize.register(QPU)
@serialize.register(QuantumParameters)
@serialize.register(QuantumElement)
@serialize.register(Transmon)
@serialize.register(Results)
@serialize.register(QuantumParametersContainer)
@serialize.register(QuantumElementContainer)
@serialize.register(QuantumParametersDict)
@serialize.register(QuantumElementDict)
@serialize.register(TaskOptions)
@serialize.register(WorkflowOptions)
def serialize_laboneq_object(obj: object, opener: SerializeOpener) -> None:
    """Serialize LabOne Q types."""
    from laboneq.serializers import to_json

    with opener.open("json", binary=True) as f:
        f.write(to_json(obj))


@serialize.register
def serialize_list(obj: list, opener: SerializeOpener) -> None:
    """Serialize lists.

    Supports lists of quantum parameters and lists that can be
    serialized by converting them to numpy arrays and using
    `.serialize_numpy_array`.
    """
    if obj and QuantumParametersContainer.supports(obj):
        serialize_laboneq_object(QuantumParametersContainer(obj), opener)
    elif obj and QuantumElementContainer.supports(obj):
        serialize_laboneq_object(QuantumElementContainer(obj), opener)
    else:
        # serialize_numpy_array passes allow_pickle=False to numpy
        # write_array, so the line below rejects lists that numpy
        # will convert to an array with a dtype of object:
        obj_array = np.array(obj)
        if obj_array.dtype is np.dtype(object):
            raise SerializationNotSupportedError(
                f"Lists that convert to NumPy arrays with dtype dtype('O') are"
                f" not supported by the serializer [name: {opener.name()}]."
            )
        serialize_numpy_array(obj_array, opener)


@serialize.register
def serialize_tuple(obj: tuple, opener: SerializeOpener) -> None:
    """Serialize tuples.

    Supports tuples of quantum parameters and values that can be
    serialized to JSON.
    """
    if obj and QuantumParametersContainer.supports(obj):
        serialize_laboneq_object(QuantumParametersContainer(obj), opener)
    elif obj and QuantumElementContainer.supports(obj):
        serialize_laboneq_object(QuantumElementContainer(obj), opener)
    else:
        if _ORJSON_SERIALIZER.supported_object(obj):
            _ORJSON_SERIALIZER.serialize(obj, opener)
        else:
            unsupported = ", ".join(
                f"{i}: value={v!r} type={type(v).__qualname__}"
                for i, v in enumerate(obj)
                if not _ORJSON_SERIALIZER.supported_object(v)
            )
            supported_types = ", ".join(
                t.__qualname__ for t in _ORJSON_SERIALIZER.SUPPORTED_TYPES
            )
            raise SerializationNotSupportedError(
                f"Type {type(obj)!r} has the following unsupported values: ({unsupported})."
                f" Only tuples with values of the following types are supported: [{supported_types}]."
                f" [name: {opener.name()}]."
            )


@serialize.register
def serialize_dict(obj: dict, opener: SerializeOpener) -> None:
    """Serialize dicts."""
    if obj and QuantumParametersDict.supports(obj):
        serialize_laboneq_object(QuantumParametersDict(obj), opener)
    elif obj and QuantumElementDict.supports(obj):
        serialize_laboneq_object(QuantumElementDict(obj), opener)
    else:
        if _ORJSON_SERIALIZER.supported_object(obj):
            _ORJSON_SERIALIZER.serialize(obj, opener)
        else:
            non_string_keys = [k for k in obj if not isinstance(k, str)]
            if non_string_keys:
                raise SerializationNotSupportedError(
                    f"Type {type(obj)!r} has the following non-string keys: {non_string_keys!r}."
                    f" Only dictionaries with str keys are supported by the folder store serializer."
                    f" [name: {opener.name()}]."
                )
            unsupported = ", ".join(
                f"{k!r}: value={v!r} type={type(v).__qualname__}"
                for k, v in obj.items()
                if not _ORJSON_SERIALIZER.supported_object(v)
            )
            supported_types = ", ".join(
                t.__qualname__ for t in _ORJSON_SERIALIZER.SUPPORTED_TYPES
            )
            raise SerializationNotSupportedError(
                f"Type {type(obj)!r} has the following unsupported values: {{{unsupported}}}."
                f" Only dictionaries with values of the following types are supported: [{supported_types}]."
                f" [name: {opener.name()}]."
            )
