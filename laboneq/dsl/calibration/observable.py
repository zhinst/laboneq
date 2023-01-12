# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""
Mixin for dataclasses (!), makes the fields observable.

The class will expose two new methods, `about_to_change()` and `has_changed()`,
Both methods return a `Signal` object. Add or remove callbacks via its `connect()`
and `disconnect()` methods.

The signature of the callback is:

     def callback(observable: Observable, field: str, value):
        ...

When implementing `__post_init__()`, remember to call `super().__post_init__()`!

Example
=======

    >>> @dataclasses.dataclass
    ... class Bar(Observable):
    ...     a: int = 2
    ...     b: int = 4
    ...
    >>> b = Bar()
    >>> def has_changed_callback(observable, key, value):
    ...     print(f"Changed: {observable}, {key}={value}")
    >>> b.has_changed().connect(has_changed_callback)
    >>> b.a = 123
    Changed: Bar(a=123, b=4), a=123

Non-fields are not observed:
    >>> b.c = 1

Unsubscribe:
    >>> b.has_changed().disconnect(has_changed_callback)
    >>> b.a = 0
"""

import weakref
from typing import Any, Callable, List

CallbackType = Callable[[Any, str, Any], None]


class Signal:
    def __init__(self, observable: Any):
        self._callbacks: List[CallbackType] = []
        self._observable = weakref.ref(observable)

    def connect(self, callback: CallbackType):
        # Need to check for identity, cannot use `callback in self._callbacks`, which
        # checks for equality.
        if not any(callback is element for element in self._callbacks):
            self._callbacks.append(callback)

    def disconnect(self, callback):
        self._callbacks.remove(callback)

    def observable(self):
        return self._observable()

    def fire(self, key, value):
        # We must create a copy here. To avoid recursion, the callbacks might
        # unsubscribe themselves temporarily. However, since we iterate over the list,
        # it must be immutable.
        callbacks = self._callbacks.copy()
        for callback in callbacks:
            callback(self.observable(), key, value)


class Observable:
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            # Other than in multiple inheritance the base class likely does not
            # implement `__post_init__()`
            super().__post_init__()

        self._signal_changed = Signal(self)

    def observed_fields(self) -> List[str]:
        return list(self.__dataclass_fields__.keys())

    def has_changed(self):
        return self._signal_changed

    def __setattr__(self, key, value):
        if (
            not (
                # Early on  in `dataclass.__init__()`, `_signal_changed` has not been
                # set yet.
                hasattr(self, "_signal_changed")
            )
            or key not in self.observed_fields()
        ):
            return super().__setattr__(key, value)

        super().__setattr__(key, value)
        self._signal_changed.fire(key, value)


class RecursiveObservable(Observable):
    def __setattr__(self, key, value):
        if isinstance(value, Observable):
            try:
                value.has_changed().connect(
                    lambda obs, k, v: self.has_changed().fire(k, v)
                )
            except TypeError:
                #  `_signal_changed` has not been set yet.
                pass
        super().__setattr__(key, value)
