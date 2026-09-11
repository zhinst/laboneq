# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.data.artifacts_qccs import (
    COMPLEX_USAGE,
    CommandTableMapEntry,
)

if TYPE_CHECKING:
    from laboneq.data.artifacts_qccs import ArtifactsCodegen


def calc_ct_replacement(
    artifacts: ArtifactsCodegen,
    parameter_uid: str,
    new_value: int | float,
    in_place=False,
) -> list[dict[str, Any]]:
    try:
        phase_increment_map = artifacts.parameter_phase_increment_map[parameter_uid]
    except KeyError as e:
        raise LabOneQException(
            f"Cannot replace parameter '{parameter_uid}'. Check that the parameter was"
            f" used for phase increments of a HW oscillator."
        ) from e

    old_tables_by_key = {ct["seqc"]: ct for ct in artifacts.command_tables}
    if in_place:
        new_tables_by_key = old_tables_by_key
    else:
        new_tables_by_key = {}

    def get_or_copy(key):
        default = old_tables_by_key[key]
        if not in_place:
            default = deepcopy(default)
        return new_tables_by_key.setdefault(key, default)

    for entry in phase_increment_map.entries:
        if entry == COMPLEX_USAGE:
            raise LabOneQException(
                f"Cannot replace phase increment driven by parameter '{parameter_uid}'."
                f" The phase increment is tied to the waveform of other pulses."
                f" Recompile the experiment instead."
            )
        assert isinstance(entry, CommandTableMapEntry)

        table = get_or_copy(entry.ct_ref)

        ct_entry = next(
            ct_entry
            for ct_entry in table["ct"]["table"]
            if ct_entry["index"] == entry.ct_index
        )

        # depending on whether the instrument is HDAWG or SHFSG, the table is slightly different
        if "phase" in ct_entry:  # SHFSG
            ct_entry["phase"]["value"] = new_value * 180 / np.pi
            assert ct_entry["phase"]["increment"] is True
        else:  # HDAWG
            assert "phase0" in ct_entry and "phase1" in ct_entry
            ct_entry["phase0"]["value"] = new_value * 180 / np.pi
            ct_entry["phase1"]["value"] = new_value * 180 / np.pi
            assert ct_entry["phase0"]["increment"] is True
            assert ct_entry["phase1"]["increment"] is True

    return list(new_tables_by_key.values())
