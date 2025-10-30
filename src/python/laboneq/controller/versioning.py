# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass

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
        return self.show()

    def show(self, omit_build: bool = False):
        if omit_build:
            return f"{self.year}.{self.month:02d}.{self.patch}"
        else:
            return f"{self.year}.{self.month:02d}.{self.patch}.{self.build}"

    @classmethod
    def from_version_string(cls, s: str):
        """Parse a version string from this format: YEAR.MONTH.PATCH.BUILD"""
        version_fields = [int(fld) for fld in s.split(".")]

        if len(version_fields) != 4:
            raise ValueError(f"Unrecognized version string. ({s})")

        return cls(
            year=version_fields[0],
            month=version_fields[1],
            patch=version_fields[2],
            build=version_fields[3],
        )


RECOMMENDED_MINIMUM_LABONE_VERSION = LabOneVersion(
    year=25, month=10, patch=0, build=271
)
"""This variable holds the L1 version where the latest stable release of LabOne Q was fully tested against."""

MIN_LABONE_VERSION_SHF_BUSY = LabOneVersion(year=25, month=4, patch=0, build=0)


@dataclass
class SetupCaps:
    client_version: LabOneVersion
    server_version: LabOneVersion | None = None

    def for_server(self, server_version: LabOneVersion) -> SetupCaps:
        return SetupCaps(
            client_version=self.client_version, server_version=server_version
        )

    @property
    def supports_shf_busy(self) -> bool:
        if self.server_version is None:
            return False
        return self.server_version >= MIN_LABONE_VERSION_SHF_BUSY
