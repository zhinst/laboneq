# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Dict, List, Optional

from laboneq.compiler.code_generator.signatures import PlaybackSignature
from laboneq.compiler.common.device_type import DeviceType


class CommandTableTracker:
    def __init__(self, device_type: DeviceType):
        self._command_table: Dict[PlaybackSignature, Dict] = {}
        self._device_type = device_type

    def lookup_index_by_signature(self, signature: PlaybackSignature) -> Optional[int]:
        table_entry = self._command_table.get(signature)
        if table_entry is None:
            return
        return table_entry["index"]

    def __getitem__(self, index: int):
        return next(
            (
                (sig, ct_entry)
                for sig, ct_entry in self._command_table.items()
                if ct_entry["index"] == index
            ),
            None,
        )

    def __len__(self):
        return len(self._command_table)

    def create_entry(
        self, signature: PlaybackSignature, wave_index: Optional[int]
    ) -> int:
        assert signature not in self._command_table
        index = len(self._command_table)
        ct_entry = {"index": index}
        if wave_index is None:
            if signature.waveform is not None:
                length = signature.waveform.length
                ct_entry["waveform"] = {"playZero": True, "length": length}

        else:
            ct_entry["waveform"] = {"index": wave_index}
            if signature.clear_precompensation:
                ct_entry["waveform"]["precompClear"] = True
        ct_entry.update(self._oscillator_config(signature))
        ct_entry.update(self._amplitude_config(signature))
        self._command_table[signature] = ct_entry
        return index

    def _oscillator_config(self, signature: PlaybackSignature):
        d = {}
        oscillator = signature.hw_oscillator
        if oscillator is not None:
            d["oscillatorSelect"] = {"value": {"$ref": oscillator}}

        ct_phase = None
        do_incr = None
        set_phase = signature.set_phase
        if set_phase is not None:
            ct_phase = set_phase
            do_incr = False
        incr_phase = signature.increment_phase
        if incr_phase is not None:
            assert set_phase is None, "Cannot set and increment phase at the same time"
            ct_phase = incr_phase
            do_incr = True

        if ct_phase is not None:
            assert do_incr is not None
            ct_phase *= 180 / math.pi
            ct_phase %= 360
            if self._device_type == DeviceType.HDAWG:
                if do_incr:
                    d["phase0"] = {"value": ct_phase % 360, "increment": True}
                    d["phase1"] = {"value": ct_phase}
                else:
                    d["phase0"] = {"value": (ct_phase + 90) % 360}
                    d["phase1"] = {"value": ct_phase}
                if do_incr:
                    d["phase1"]["increment"] = True
            elif self._device_type == DeviceType.SHFSG:
                d["phase"] = {"value": ct_phase}
                if do_incr:
                    d["phase"]["increment"] = True
            else:
                raise ValueError(f"Unsupported device type: {self._device_type}")

        return d

    def _amplitude_config(self, signature: PlaybackSignature):
        d = {}
        ct_amplitude = signature.set_amplitude
        if ct_amplitude is not None:
            if self._device_type == DeviceType.HDAWG:
                d["amplitude0"] = d["amplitude1"] = {"value": ct_amplitude}
            elif self._device_type == DeviceType.SHFSG:
                d["amplitude00"] = d["amplitude10"] = d["amplitude11"] = {
                    "value": ct_amplitude
                }
                d["amplitude01"] = {"value": -ct_amplitude}
        return d

    def command_table(self) -> List[Dict]:
        return list(self._command_table.values())
