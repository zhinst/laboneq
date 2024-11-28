# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A wrapper for accessing members of a dict in dot notation."""

from __future__ import annotations

from collections.abc import (
    Collection,
    ItemsView,
    Iterable,
    Iterator,
    Mapping,
    ValuesView,
)
from collections.abc import KeysView as BaseKeysView
from typing import Any, cast

from laboneq.workflow.tasks.common.classformatter import classformatter


def find_common_prefix(keys: set[str], separator: str) -> tuple[str, str] | None:
    """Check if there is no key which is also a prefix of another.

    Args:
        keys: A set of keys.
        separator: The symbol used to separate levels in the keys.

    Returns:
        The shorter and longer key if there is a key which is also the prefix of
        another, None otherwise
    """
    # Sort. Appending the separator helps to avoid cases if the key name
    # contains a character which is sorts before the separator, like
    # "a/a", "a/a/b" and "a/a.".
    sorted_keys = sorted(k + separator for k in keys)
    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i + 1]
        if k2.startswith(k1):
            return k1[:-1], k2[:-1]
    return None


def _check_prefix(keys: set[str], separator: str) -> None:
    """Check if there is no key which is also a prefix of another.

    Args:
        keys: A set of keys.
        separator: The symbol used to separate levels in the keys.

    Raises:
        ValueError: If there is a key which is also the prefix of another.
    """
    k = find_common_prefix(keys, separator)
    if k is not None:
        raise ValueError(
            f"Handle '{k[0]}' is a prefix of handle '{k[1]}', which is not"
            " allowed, because a results entry cannot contain both data and "
            "another results subtree. Please rename one of the handles.",
        )


def _check_attrs(keys: set[str], attrs: list[str], separator: str) -> None:
    """Check if no subkey matches an attribute of the class.

    Args:
        keys: A set of keys.
        attrs: A list of attribute names.
        separator: The symbol used to separate levels in the keys.

    Raises:
        ValueError: If a key is also an attribute of the class.
    """
    attrs_set = set(attrs)
    if not all(key is not None for key in keys):
        raise ValueError(
            f"The acquire handle cannot be None. Please check the handles: {keys}."
        )
    subkeys = {subkey for key in keys for subkey in key.split(separator)}
    if attrs_set & subkeys:
        raise ValueError(
            f"Handles {subkeys & attrs_set} aren't allowed names.",
        )


@classformatter
class AttributeWrapper(Collection[str]):
    """A wrapper for accessing members of a dict in dot notation.

    Input data is a dict, where each key is a string, where levels are separated by
    slashes. The wrapper allows to access the data in dot notation, where the levels
    are separated by dots. The wrapper also provides a read-only dict interface.

    Attributes:
        data: The dict to wrap.
        path: The path to the current level in the dict.

    Example:
    ```python
    data = {
        "cal_trace/q0/g": 12345,
    }
    wrapper = AttributeWrapper(data)
    assert wrapper.cal_trace.q0.g == 12345
    assert len(wrapper.cal_trace) == 1
    assert set(wrapper.cal_trace.keys()) == {"q0"}
    ```
    """

    class KeysView(BaseKeysView[str]):
        """A view of the keys of an AttributeWrapper."""

        def __str__(self) -> str:
            return f"AttributesView({list(self._mapping.keys())})"

        def __repr__(self) -> str:
            return str(self)

    def _get_subkey(self, key: str) -> str:
        if len(self._path) == 0:
            return key.split(self._separator, 1)[0]
        prefix_len = len(self._path)
        return key[prefix_len + 1 :].split(self._separator, 1)[0]

    def _get_subkeys(self, key: str) -> str:
        if len(self._path) == 0:
            return key.replace(self._separator, ".")
        prefix_len = len(self._path)
        return key[prefix_len + 1 :].replace(self._separator, ".")

    def _add_path(self, key: str) -> str:
        return (self._path + self._separator + key) if self._path else key

    def __init__(
        self,
        data: Mapping[str, object] | None,
        path: str | None = None,
        separator: str | None = None,
    ) -> None:
        super().__init__()
        self._data: Mapping[str, object] = data or {}
        self._path = path or ""
        self._separator = separator if separator is not None else "/"
        self._key_cache: set[str] = set()
        keys_set = set(self._data.keys())
        _check_attrs(keys_set, dir(self), self._separator)
        _check_prefix(keys_set, self._separator)
        self._key_cache = {
            self._get_subkey(k) for k in self._data if k.startswith(self._path)
        }

    # Partial Mapping interface
    def __len__(self) -> int:
        return len(self._key_cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._key_cache)

    def __getitem__(self, key: object) -> object:
        if not isinstance(key, str):
            raise TypeError(f"Key {key} has to be of type str.")
        path = self._add_path(key)
        try:
            return self._data[path]
        except KeyError as e:
            path_sep = path + self._separator
            if not any(k.startswith(path_sep) for k in self._data):
                raise KeyError(f"Key '{self._path}' not found in the data.") from e
            return AttributeWrapper(self._data, path)

    def __contains__(self, key: object) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def _keys(self) -> AttributeWrapper.KeysView:
        """A set-like object providing a view on the available attributes."""
        return AttributeWrapper.KeysView(cast(Mapping[str, Any], self))

    def _items(self) -> ItemsView[str, Any]:
        """A set-like object providing a view on wrapper's items."""
        return ItemsView(cast(Mapping[str, Any], self))

    def _values(self) -> ValuesView[Any]:
        """An object providing a view on the wrapper's values."""
        return ValuesView(cast(Mapping[str, Any], self))

    # End of Mapping interface

    def __getattr__(self, key: str) -> object:
        try:
            return self.__getitem__(key)
        except KeyError as e:
            raise AttributeError(
                f"Key '{self._add_path(key)}' not found in the data.",
            ) from e

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AttributeWrapper):
            return NotImplemented
        return (
            self._data == value._data
            and self._path == value._path
            and self._separator == value._separator
        )

    def _as_str_dict(self) -> dict[str, Any]:
        return {
            key: (attr._as_str_dict() if isinstance(attr, AttributeWrapper) else attr)
            for key, attr in ((key, getattr(self, key)) for key in self._key_cache)
        }

    def __dir__(self) -> Iterable[str]:
        return list(super().__dir__()) + list(self._key_cache)

    def __repr__(self) -> str:
        return (
            f"AttributeWrapper(data={self._data!r}, path={self._path!r}, "
            f"separator={self._separator!r})"
        )
