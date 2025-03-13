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

    year: int
    month: int
    patch: int
    build: int

    def __str__(self):
        return f"{self.year}.{self.month:02d}.{self.patch}.{self.build}"

    @classmethod
    def from_version_string(cls, s: str):
        """Parse a version string from this format: YEAR.MONTH.PATCH.BUILD"""
        version_fields = [int(fld) for fld in s.split(".")]

        if len(version_fields) == 4:
            return cls(
                year=version_fields[0],
                month=version_fields[1],
                patch=version_fields[2],
                build=version_fields[3],
            )
        if len(version_fields) == 3:
            # Only required to parse pre-25.01 version strings.
            return cls(
                year=version_fields[0],
                month=version_fields[1],
                patch=0,
                build=version_fields[2],
            )
        else:
            raise ValueError(f"Unrecognized version string. ({s})")


RECOMMENDED_LABONE_VERSION = LabOneVersion(year=25, month=1, patch=0, build=0)
"""This variable holds the version what we currently support and actively test against."""

MINIMUM_SUPPORTED_LABONE_VERSION = LabOneVersion(year=25, month=1, patch=0, build=0)
"""This variable holds the minimum version that we expect LabOne Q to work
reliably, but may not be testing against anymore. Most of the time, this will
be equal to `RECOMMENDED_LABONE_VERSION` with the exceptions happening
typically around new LabOne releases to be able to support previous release in
case an issue is found."""


# LabOne Q version (major,minor) marked to remove support for all LabOne
# versions less than RECOMMENDED_LABONE_VERSION.
DROP_SUPPORT_FOR_PREVIOUS_L1 = (2, 47)


class SetupCaps:
    def __init__(self, version: LabOneVersion):
        self._version = version
