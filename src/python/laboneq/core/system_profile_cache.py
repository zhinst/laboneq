# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System profile cache management.

System profiles are cached locally to enable offline compilation.
Cache location is platform-specific and can be overridden via environment variable.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

from laboneq.dsl.device import SystemProfile
from laboneq.serializers.core import SerializerFormat
from laboneq.serializers.core import load as core_load
from laboneq.serializers.core import save as core_save


def get_cache_dir() -> Path:
    """Get platform-appropriate cache directory for system profiles.

    Environment:
        LABONEQ_PROFILE_CACHE: Override default cache location
    """
    if env_path := os.environ.get("LABONEQ_PROFILE_CACHE"):
        return Path(env_path).expanduser()

    system = platform.system()

    if system == "Windows":
        homedir = os.environ.get("LOCALAPPDATA")
        if homedir is not None:
            base = Path(homedir).expanduser()
        else:
            base = Path.home() / "AppData" / "Local"
        cache_dir = base / "laboneq" / "profiles"
    elif system == "Darwin":
        cache_dir = Path.home() / "Library" / "Caches" / "laboneq" / "profiles"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "~/.cache")
        cache_dir = Path(xdg_cache).expanduser() / "laboneq" / "profiles"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load(setup_uid: str, setup_hash: str | None = None) -> SystemProfile | None:
    """Load system profile from cache.

    Args:
        setup_uid: Device setup UID
        setup_hash: Optional specific hash to load

    Returns:
        System profile if found in cache, None otherwise
    """
    cache_dir = get_cache_dir()

    if setup_hash:
        profile_path = cache_dir / f"{setup_uid}-{setup_hash}.yaml"
        if not profile_path.exists():
            return None
    else:
        # Take the latest one as defined by the file modification time.
        # This assumes that users did not manually tamper with the cache files.
        pattern = f"{setup_uid}-*.yaml"
        matches = list(cache_dir.glob(pattern))
        if not matches:
            return None
        profile_path = max(matches, key=lambda p: p.stat().st_mtime)

    try:
        profile = core_load(profile_path, format=SerializerFormat.YAML)
        if not isinstance(profile, SystemProfile):
            raise TypeError("Loaded object is not a SystemProfile")
        return profile
    except Exception:
        # Failed to load profile, possibly due to corruption or a version downgrade. Ignore it, but notify user.
        logging.info(
            "Failed to load system profile from cache: %s", profile_path, exc_info=True
        )
        return None


def save(profile: SystemProfile) -> Path:
    """Save system profile to cache.

    Args:
        profile: System profile to save

    Returns:
        Path where profile was saved
    """
    cache_dir = get_cache_dir()

    fingerprint = profile.get_fingerprint()
    profile_path = cache_dir / f"{profile.uid}-{fingerprint}.yaml"

    core_save(
        profile,
        profile_path,
        format=SerializerFormat.YAML,
    )

    return profile_path


def install(source_path: Path, setup_uid: str) -> Path:
    """Install a system profile from external source into cache.

    Args:
        source_path: Path to profile YAML file
        setup_uid: Setup UID to install under

    Returns:
        Path where profile was installed
    """
    profile = core_load(source_path, format=SerializerFormat.YAML)
    assert isinstance(profile, SystemProfile)

    profile.uid = setup_uid

    return save(profile)


def flush(*, setup_uid: str | None = None, setup_hash: str | None = None) -> None:
    """Clear the system profile cache for the given setup UID and/or hash.

    If no arguments are provided, the entire cache is cleared.

    Args:
        setup_uid: If provided, only clear profiles matching this setup UID.
        setup_hash: If provided, only clear profiles matching this specific hash.
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
