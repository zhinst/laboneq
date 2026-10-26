# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=wrong-import-order


"""Main functionality of the LabOne Q Software."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from laboneq._version import __version__, get_version  # noqa F401

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.metadata import EntryPoint


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


def _check_for_superseded_backends(backend_eps: Iterable[EntryPoint]) -> None:
    """Refuse to load when several distributions provide the same backend.

    A renamed backend package keeps its predecessor's entry point registered
    until the obsolete distribution is uninstalled, so both claim the same
    backend name. Entry point ordering alone decides which one loads second.
    That one either registers over the first or dies on a module that the old
    install no longer ships.

    Reporting the clash rather than picking a winner is deliberate. A wrong
    pick compiles experiments against a stale backend, which is worse than
    refusing to start, and no environment is better off running with two.

    Which distribution is obsolete is left unsaid. Versions do not settle it:
    a superseded pair can share one, and the newer install is not reliably the
    surviving one. Naming the wrong package would send the user to uninstall
    the one they need.

    Keys off distribution metadata instead of package names, so it needs no
    upkeep as backends come and go.

    Raises:
        RuntimeError: If two or more installed distributions provide the same
            backend name.
    """
    providers: dict[str, list[EntryPoint]] = {}
    for ep in backend_eps:
        providers.setdefault(ep.name, []).append(ep)

    for name, eps in sorted(providers.items()):
        if len(eps) < 2:
            continue
        dists = [getattr(ep, "dist", None) for ep in eps]
        labels = sorted(
            f"{d.name} {d.version}" if d is not None else "<unknown>" for d in dists
        )
        raise RuntimeError(
            f"The {name!r} backend is provided by more than one installed "
            f"distribution ({', '.join(labels)}). Uninstall whichever of them "
            f"is obsolete."
        )


def _load_backend_plugins():
    """Load and register all backend plugins via entry points.

    Backends register themselves in the "laboneq.backends" entry point group.
    Each backend's plugin class is instantiated and its register() method is called.
    """
    from importlib.metadata import entry_points

    backend_eps = entry_points().select(group="laboneq.backends")

    _check_for_superseded_backends(backend_eps)

    for ep in backend_eps:
        _load_backend_plugin(ep)


_load_backend_plugins()
