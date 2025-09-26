# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A wrapper for accessing members of a dict in dot notation."""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
)

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


@classformatter
class AttributeWrapper:
    """A wrapper for result data that provides access to results via attribute lookup.

    The wrapper treats data keys as if they were paths (like in a filename or URL) where
    folder names as separated by ``/`` (or other separator if specified).

    The wrapper provides access to the data in three different ways:

    - when used as a sequence (e.g ``list(wrapper)`` or ``len(wrapper)``) it behaves
      as an ordinary sequence of keys in the data dictionary.

    - when used for item lookup (e.g. ``wrapper[key]`` or ``key in wrapper``) it
      behaves as if the key were either a complete path (in which case it returns
      the original value from the data) or a folder path (in which case it returns
      a new ``AttributeWrapper`` with the subset of the data that starts with the
      given path).

    - when used for attribute lookup (e.g. ``wrapper.key`` or ``dir(wrapper)``) it
      behaves as above for item lookup, but the key may not include the separator
      and ``dir`` returns only top-level folder paths and complete paths that do
      not include the separator.

    If any complete key in data is also a top-level "folder" name, an ambiguity
    results. For example, if data contains ``{"a": 5, "a/b": 6}`` then ``wrapper.a``
    could return either ``5`` or a wrapper for``{"b": 6}``. In such cases the
    wrapper always returns the data for the complete key, i.e. ``5`` in the
    preceding example. To access the keys inside the folder, using a more complete
    key, e.g. ``wrapper["a/b"]``.

    Attributes:
        data:
            The dictionary of result data to wrap.
        separator:
            The path separator to use.


    Raises:
        ValueError:
            When the keys are not strings.

    Example:
        ```python
        wrapper = AttributeWrapper(
            {
                "cal_trace/q0/g": 123,
                "cal_trace/q1/g": 456,
            }
        )

        # used as a sequence:

        assert list(wrapper) == [
            "cal_trace/q0/g",
            "cal_trace/q1/g",
        ]
        assert len(wrapper) == 2

        # used with item lookup:

        assert wrapper["cal_trace/q0/g"] == 123
        assert list(wrapper["cal_trace"]) == ["q0", "q1"]
        assert list(wrapper["cal_trace/q0"]) == ["g"]
        assert "cal_trace" in wrapper
        assert "cal_trace/q0" in wrapper
        assert "cal_trace/q0/g" in wrapper

        # used with attribute lookup:

        assert list(wrapper.cal_trace) == ["q0", "q1"]
        assert list(wrapper.cal_trace.q0) == ["g"]
        assert wrapper.cal_trace.q0.g == 123
        assert "cal_trace" in dir(wrapper)
        ```
    """

    def __init__(
        self,
        data: Mapping[str, object],
        *,
        separator: str = "/",
    ):
        self._separator = separator
        self._check_keys_are_strings(data)
        self._data = data

        # find prefixes and leaves
        prefixes_and_leaves = {k.partition(separator)[0:2] for k in self._data}
        self._prefixes = {k[0] for k in prefixes_and_leaves if k[1]}
        self._leaves = {k[0] for k in prefixes_and_leaves if not k[1]}

    def _check_keys_are_strings(self, data: Mapping[str, object]):
        """Check that keys are strings."""
        non_str_keys = {k for k in data if not isinstance(k, str)}
        if non_str_keys:
            raise ValueError(
                f"The result keys must be strings."
                f" The following keys are not strings: {non_str_keys}."
            )

    # Sequence interface

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    # Item interface

    def __getitem__(self, key: object) -> object:
        if not isinstance(key, str):
            raise KeyError(key)
        if key in self._data:
            return self._data[key]
        folder_path = key + self._separator
        folder_path_length = len(folder_path)
        folder_data = {
            k[folder_path_length:]: v
            for k, v in self._data.items()
            if k.startswith(folder_path)
        }
        if folder_data:
            return AttributeWrapper(
                folder_data,
                separator=self._separator,
            )
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._data:
            return True
        folder_path = key + self._separator
        return any(k.startswith(folder_path) for k in self._data)

    # Attribute interface

    def __getattr__(self, key: str) -> object:
        if key in self._leaves or key in self._prefixes:
            return self[key]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {key!r}"
        ) from None

    def __dir__(self) -> Iterable[str]:
        attrs = list(super().__dir__())
        attrs.extend(self._attr_keys())
        return attrs

    def _attr_keys(self) -> Iterable[str]:
        """Return just the attribute keys.

        This method is called by Results.__dir__.
        """
        return sorted(self._leaves) + sorted(self._prefixes)

    # Comparison interface

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AttributeWrapper):
            return NotImplemented
        return self._data == value._data and self._separator == value._separator

    # Reprs

    def __repr__(self) -> str:
        return (
            f"<{type(self).__qualname__}"
            f" len(data)={len(self._data)},"
            f" separator={self._separator!r})"
            ">"
        )

    def __rich_repr__(self):
        for key in sorted(self._leaves):
            yield key, self._data[key]
        for key in sorted(self._prefixes):
            yield key, getattr(self, key)
