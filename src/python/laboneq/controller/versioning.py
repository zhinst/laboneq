# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs


@attrs.define(order=True, kw_only=True)
class LabOneVersion:
    """Class to represent LabOne product versions.

    If build is 0, str conversion of LabOneVersion will omit the build
    part, which is useful for clean error messages.
    """

    major: int
    minor: int
    patch: int | None  # QLUE1: Remove `None` as a possible type here.
    build: int

    def __str__(self):
        # Use zero-padding for minor to distinguish CalVer.
        if self.patch is None:
            return f"{self.major}.{self.minor:02d}.{self.build}"
        else:
            return f"{self.major}.{self.minor:02d}.{self.patch}.{self.build}"

    def as_tuple(
        self, *, omit_build: bool = False
    ) -> tuple[int, int, int, int] | tuple[int, int, int] | tuple[int, int]:
        """Return version as a tuple of integers. Particularly useful if one
        wants to compare up to the build number.

        Args:
            omit_build: If the version supports patch number, returns (major,
                minor, build) or (major, minor). Otherwise returns (major, minor,
                patch, build) or (major, minor, patch).

        """
        if omit_build:
            if self.patch is None:
                return (self.major, self.minor)
            else:
                return (self.major, self.minor, self.patch)
        else:
            if self.patch is None:
                return (self.major, self.minor, self.build)
            else:
                return (self.major, self.minor, self.patch, self.build)

    def as_dataserver_revision(self) -> int:
        """Pack revision information similar to how LabOne data server does it."""
        if self.patch is None:
            return int(f"{self.major}{self.minor:02d}{self.build % 100000:05d}")
        else:
            return int(
                f"{self.major}{self.minor:02d}{self.patch % 10:01d}{self.build % 10000:04d}"
            )

    def as_dataserver_version(self) -> str:
        """Pack version information similar to how LabOne data server does it."""
        return f"{self.major}.{self.minor:02d}"

    @classmethod
    def from_version_string(cls, s: str):
        version_fields = [int(fld) for fld in s.split(".")]

        if len(version_fields) == 3:
            return cls(
                major=version_fields[0],
                minor=version_fields[1],
                build=version_fields[2],
                patch=None,
            )
        elif len(version_fields) == 4:
            return cls(
                major=version_fields[0],
                minor=version_fields[1],
                patch=version_fields[2],
                build=version_fields[3],
            )
        else:
            raise ValueError(f"Unrecognized version string. ({s})")

    @classmethod
    def from_dataserver_version_information(cls, version: str, revision: int):
        """Constructs a version object using information that can be retrieved
        from a running LabOne data server instance.

        Args:
            version: Version string of the form: {major}.{minor}.
            revision: An integer packing containing information about the full
                LabOne version.

        Raises:
            ValueError: If one of the assumptions for version and revision fail.
        """
        revision_str = str(revision)
        if revision_str.find(version.replace(".", "")) != 0:
            raise ValueError(
                "Data server revision does not contain version information."
            )
        try:
            major, minor = map(int, version.split("."))
        except ValueError as e:
            raise ValueError(
                "Data server version string is not '<major>.<version>'."
            ) from e

        if version < "25.01":
            patch = None
            build = int(revision_str[len(version) - 1 :])  # -1 for the dot.
        else:
            patch = int(revision_str[len(version) - 1])  # -1 for the dot.
            build = int(revision_str[len(version) :])  # -1 for the dot +1 for patch
        return cls(major=major, minor=minor, patch=patch, build=build)


RECOMMENDED_LABONE_VERSION = LabOneVersion(major=24, minor=10, patch=None, build=0)
"""This variable holds the version what we currently support and actively test against."""

MINIMUM_SUPPORTED_LABONE_VERSION = LabOneVersion(
    major=24, minor=10, patch=None, build=0
)
"""This variable holds the minimum version that we expect LabOne Q to work
reliably, but may not be testing against anymore. Most of the time, this will
be equal to `RECOMMENDED_LABONE_VERSION` with the exceptions happening
typically around new LabOne releases to be able to support previous release in
case an issue is found."""


# LabOne Q version (major,minor) marked to remove support for all LabOne
# versions less than RECOMMENDED_LABONE_VERSION.
DROP_SUPPORT_FOR_PREVIOUS_L1 = (2, 41)


class SetupCaps:
    def __init__(self, version: LabOneVersion):
        self._version = version
