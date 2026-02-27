# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=wrong-import-order


"""Main functionality of the LabOne Q Software."""

import os

from laboneq._version import get_version

__version__ = get_version()


def _load_backend_plugin(ep):
    """Load and register a single backend plugin.

    Args:
        ep: Entry point to load

    Raises:
        RuntimeError: If plugin fails to load in strict mode
    """
    import traceback
    import warnings

    try:
        plugin_class = ep.load()
        plugin_instance = plugin_class()
        plugin_instance.register()
    except Exception as e:
        strict_mode = os.environ.get("LABONEQ_STRICT_BACKEND_PLUGINS", "").lower()
        if strict_mode in {"1", "true", "yes", "on"}:
            raise RuntimeError(f"Failed to load backend plugin {ep.name}") from e

        warnings.warn(
            (f"Failed to load backend plugin {ep.name}: {e}\n{traceback.format_exc()}"),
            RuntimeWarning,
            stacklevel=3,
        )


def _load_backend_plugins():
    """Load and register all backend plugins via entry points.

    Backends register themselves in the "laboneq.backends" entry point group.
    Each backend's plugin class is instantiated and its register() method is called.
    """
    from importlib.metadata import entry_points

    backend_eps = entry_points().select(group="laboneq.backends")

    for ep in backend_eps:
        _load_backend_plugin(ep)


_load_backend_plugins()
