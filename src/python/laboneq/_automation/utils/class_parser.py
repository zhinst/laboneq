# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq._automation.logic import AutomationLogic


def find_logic_class(class_name: str) -> type[AutomationLogic]:
    """Return the AutomationLogic subclass with the given name.

    Arguments:
        class_name: The name of the AutomationLogic subclass.

    Returns:
        The AutomationLogic subclass.
    """

    def _all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _all_subclasses(sub)

    for sub in _all_subclasses(AutomationLogic):
        if sub.__name__ == class_name:
            return sub

    raise ValueError(f"The AutomationLogic subclass {class_name!r} is not defined.")
