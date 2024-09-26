# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs


@attrs.define(order=True)
class LabOneVersion:
    """Class to represent LabOne product versions.

    If build is 0, str conversion of LabOneVersion will omit the build
    part, which is useful for clean error messages.
    """

    major: int
    minor: int
    build: int

    def __str__(self):
        # Use zero-padding for minor to distinguish CalVer.
        version = f"{self.major}.{self.minor:02d}"
        if self.build == 0:
            # Omit build number if it's zero
            return version
        else:
            return f"{version}.{self.build}"

    def as_tuple(
        self, *, omit_build: bool = False
    ) -> tuple[int, int, int] | tuple[int, int]:
        """Return version as a tuple of integers. Particularly useful if one
        wants to compare up to the build number.

        Args:
            omit_build: If `True` returned tuple is (major, minor).
                Otherwise, returns (major, minor, build).

        """
        if omit_build:
            return (self.major, self.minor)
        else:
            return (self.major, self.minor, self.build)

    def as_dataserver_revision(self) -> int:
        """Pack revision information similar to how LabOne data server does it."""
        return int(f"{self.major}{self.minor:02d}{self.build}")

    def as_dataserver_version(self) -> str:
        """Pack version information similar to how LabOne data server does it."""
        return f"{self.major}.{self.minor:02d}"

    @classmethod
    def from_version_string(cls, s: str):
        major, minor, build = map(int, s.split("."))
        return cls(major, minor, build)

    @classmethod
    def from_dataserver_version_information(cls, version: str, revision: int):
        """Constructs a version object using information that can be retrieved
        from a running LabOne data server instance.

        Args:
            version: Version string of the form: {major}.{minor}.
            revision: An integer containing the version and the build number
                information. When represented in decimal as a string, must have the form:
                {major}{minor}{build}.

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
        build = int(revision_str[len(version) - 1 :])  # -1 for the dot.
        return cls(major, minor, build)


RECOMMENDED_LABONE_VERSION = LabOneVersion(24, 7, 0)
"""This variable holds the version what we currently support and actively test against."""

MINIMUM_SUPPORTED_LABONE_VERSION = RECOMMENDED_LABONE_VERSION
"""This variable holds the minimum version that we expect LabOne Q to work
reliably, but may not be testing against anymore. Most of the time, this will
be equal to `RECOMMENDED_LABONE_VERSION` with the exceptions happening
typically around new LabOne releases to be able to support previous release in
case an issue is found."""


# LabOne Q version (major,minor) marked to remove support for all LabOne
# versions less than RECOMMENDED_LABONE_VERSION.
DROP_SUPPORT_FOR_PREVIOUS_L1 = (2, 37)


class SetupCaps:
    def __init__(self, version: LabOneVersion):
        self._version = version
