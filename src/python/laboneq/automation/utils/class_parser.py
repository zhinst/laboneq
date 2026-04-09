# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import importlib

from laboneq.automation.logic import AutomationLogic


def find_logic_class(class_path: str) -> type[AutomationLogic]:
    """Return the AutomationLogic subclass identified by a fully qualified path.

    Arguments:
        class_path: The fully qualified class path in the form
            ``"package.module:ClassName"``, or ``"package.module:Outer.Inner"``
            for nested classes.

    Returns:
        The AutomationLogic subclass.

    Raises:
        ValueError: If the class cannot be found or is not an AutomationLogic subclass.
    """
    try:
        module_name, qualname = class_path.rsplit(":", 1)
    except ValueError:
        raise ValueError(
            f"Invalid class path {class_path!r}. "
            "Expected format: 'package.module:ClassName'."
        ) from None

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Cannot import module {module_name!r} for logic class {class_path!r}."
        ) from exc

    cls = module
    for part in qualname.split("."):
        try:
            cls = getattr(cls, part)
        except AttributeError:  # noqa: PERF203
            raise ValueError(
                f"Cannot find {part!r} in {module_name!r} for logic class {class_path!r}."
            ) from None

    if not (
        isinstance(cls, type)
        and issubclass(cls, AutomationLogic)
        and cls is not AutomationLogic
    ):
        raise ValueError(
            f"{class_path!r} does not refer to an AutomationLogic subclass."
        )

    return cls
