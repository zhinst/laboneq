# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System description cache management.

System descriptions are cached locally to enable offline compilation.
Cache location is platform-specific and can be overridden via environment variable.

This is temporary until we have the new API for instantiating the controller with a pre-loaded device setup.
"""

from __future__ import annotations

import glob
import logging
import os
import platform
from pathlib import Path

from laboneq.dsl.device import SystemDescription
from laboneq.serializers.core import SerializerFormat
from laboneq.serializers.core import load as core_load
from laboneq.serializers.core import save as core_save


def get_cache_dir() -> Path:
    """Get platform-appropriate cache directory for system descriptions.

    Environment:
        LABONEQ_SYSTEM_DESCRIPTION_CACHE: Override default cache location
    """
    if env_path := os.environ.get("LABONEQ_SYSTEM_DESCRIPTION_CACHE"):
        return Path(env_path).expanduser()

    system = platform.system()

    if system == "Windows":
        homedir = os.environ.get("LOCALAPPDATA")
        if homedir is not None:
            base = Path(homedir).expanduser()
        else:
            base = Path.home() / "AppData" / "Local"
        cache_dir = base / "laboneq" / "system_descriptions"
    elif system == "Darwin":
        cache_dir = (
            Path.home() / "Library" / "Caches" / "laboneq" / "system_descriptions"
        )
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "~/.cache")
        cache_dir = Path(xdg_cache).expanduser() / "laboneq" / "system_descriptions"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load(setup_uid: str, setup_hash: str | None = None) -> SystemDescription | None:
    """Load system description from cache.

    Args:
        setup_uid: Device setup UID
        setup_hash: Optional specific hash to load

    Returns:
        system description if found in cache, None otherwise
    """
    cache_dir = get_cache_dir()
    resolved_cache_dir = cache_dir.resolve()

    if not (cache_dir / setup_uid).resolve().is_relative_to(resolved_cache_dir):
        raise ValueError(f"setup_uid {setup_uid!r} escapes the cache directory.")

    if setup_hash:
        description_path = cache_dir / f"{setup_uid}-{setup_hash}.yaml"
        if not description_path.resolve().is_relative_to(resolved_cache_dir):
            raise ValueError(f"setup_hash {setup_hash!r} escapes the cache directory.")
        if not description_path.exists():
            return None
    else:
        # Take the latest one as defined by the file modification time.
        # This assumes that users did not manually tamper with the cache files.
        pattern = f"{glob.escape(setup_uid)}-*.yaml"
        matches = list(cache_dir.glob(pattern))
        if not matches:
            return None
        description_path = max(matches, key=lambda p: p.stat().st_mtime)

    try:
        description = core_load(description_path, format=SerializerFormat.YAML)
        if not isinstance(description, SystemDescription):
            raise TypeError("Loaded object is not a SystemDescription")
        return description
    except Exception:
        # Failed to load description, possibly due to corruption or a version downgrade. Ignore it, but notify user.
        logging.info(
            "Failed to load system description from cache: %s",
            description_path,
            exc_info=True,
        )
        return None


def save(description: SystemDescription) -> Path:
    """Save system description to cache.

    Args:
        description: system description to save

    Returns:
        Path where description was saved
    """
    cache_dir = get_cache_dir()

    fingerprint = description.get_fingerprint()
    description_path = cache_dir / f"{description.uid}-{fingerprint}.yaml"
    if not description_path.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(
            f"Description UID {description.uid!r} escapes the cache directory."
        )

    core_save(
        description,
        description_path,
        format=SerializerFormat.YAML,
    )

    return description_path


def install(source_path: Path, setup_uid: str) -> Path:
    """Install a system description from external source into cache.

    Args:
        source_path: Path to description YAML file
        setup_uid: Setup UID to install under

    Returns:
        Path where description was installed
    """
    description = core_load(source_path, format=SerializerFormat.YAML)
    assert isinstance(description, SystemDescription)

    description.uid = setup_uid

    return save(description)


def flush(*, setup_uid: str | None = None, setup_hash: str | None = None) -> None:
    """Clear the system description cache for the given setup UID and/or hash.

    If no arguments are provided, the entire cache is cleared.

    Args:
        setup_uid: If provided, only clear descriptions matching this setup UID.
        setup_hash: If provided, only clear descriptions matching this specific hash.
    """
    cache_dir = get_cache_dir()
    for file in cache_dir.glob("*.yaml"):
        filename = file.stem
        if "-" not in filename:
            continue
        uid_part, hash_part = filename.rsplit("-", 1)
        if setup_uid and uid_part != setup_uid:
            continue
        if setup_hash and hash_part != setup_hash:
            continue
        file.unlink()
