# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TypedDict

import attr
import orjson

FIELDS = [
    "pump_frequency",
    "pump_power",
    "probe_frequency",
    "probe_power",
    "cancellation_phase",
    "cancellation_attenuation",
]


class SweepCommand(TypedDict, total=False):
    pump_frequency: float
    pump_power: float
    probe_frequency: float
    probe_power: float
    cancellation_phase: float
    cancellation_attenuation: float


@attr.define(kw_only=True, slots=True)
class SHFPPCSweeperConfig:
    count: int
    commands: list[SweepCommand]
    ppc_device: str
    ppc_channel: int

    def swept_fields(self):
        return sorted(
            {field for field in FIELDS for command in self.commands if field in command}
        )

    def build_table(self):
        swept_fields = self.swept_fields()

        active_values: dict[str, float] = {}

        # We start by finding the 'default' values, ie. those that will be set first.
        for command in self.commands:
            active_values = command | active_values  # note: RHS takes precedence
            if len(active_values) == len(swept_fields):
                break

        # Next, we fill all the swept fields in all the commands.
        for command in self.commands:
            for field, default in active_values.items():
                active_values[field] = command.setdefault(field, default)

        # Finally, construct the flat list of all values.
        flat_list = [
            [
                float(command[field])
                if field != "cancellation_phase"
                else float(command[field]) * 180 / math.pi
                for field in swept_fields
            ]
            for command in self.commands
        ]

        table_dict = {
            "header": {"version": "1.0"},
            "dimensions": swept_fields,
            "flat_list": flat_list,
            "repetitions": self.count,
        }
        table_json = orjson.dumps(table_dict).decode()
        return table_json
