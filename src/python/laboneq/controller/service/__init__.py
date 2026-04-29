# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Remote controller service for LabOne Q.

This module provides a FastAPI-based REST service that allows running the
LabOne Q controller as a remote service. Clients can connect to this service
to execute experiments without needing direct access to the hardware.

Example:
    Starting the service::

        python -m laboneq.controller.service --dataserver 192.168.1.50 --host 0.0.0.0 --port 8080

    With pre-registered callbacks::

        python -m laboneq.controller.service --dataserver 192.168.1.50 --callbacks my_callbacks.py
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DeviceSetupLoader",
    "create_app",
]


def __getattr__(name: str) -> Any:
    """Lazy imports to avoid importing fastapi at module load time."""
    if name in ("create_app"):
        from laboneq.controller.service import app as _app  # noqa: PLC0415

        return getattr(_app, name)
    if name == "DeviceSetupLoader":
        from laboneq.controller.service import (
            controller_container as _cc,  # noqa: PLC0415
        )

        return _cc.DeviceSetupLoader
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
