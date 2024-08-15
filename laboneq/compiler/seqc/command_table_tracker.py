# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Dict, Optional, Literal

from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.seqc.signatures import PlaybackSignature
from laboneq.data.scheduled_experiment import COMPLEX_USAGE


class InvalidCommandTableError(Exception):
    """Base class for invalid command table errors."""

    pass


class EntryLimitExceededError(InvalidCommandTableError):
    """Too many entries are being set to the command table."""

    pass


class CommandTableTracker:
    def __init__(self, device_type: DeviceType):
        self._command_table: list[Dict] = []
        self._table_index_by_signature: Dict[PlaybackSignature, int] = {}
        self._device_type = device_type
        self._parameter_phase_increment_map: dict[
            str, list[int | Literal[COMPLEX_USAGE]]
        ] = {}

    def lookup_index_by_signature(self, signature: PlaybackSignature) -> int | None:
        return self._table_index_by_signature.get(signature)

    def __getitem__(self, index: int) -> tuple[PlaybackSignature, dict]:
        entry = self._command_table[index]

        for sig, index in self._table_index_by_signature.items():
            if index == entry["index"]:
                return sig, entry

        raise AssertionError("no signature associated with index {}".format(index))

    def __len__(self):
        return len(self._command_table)

    def create_entry(
        self,
        signature: PlaybackSignature,
        wave_index: Optional[int],
        ignore_already_in_table=False,
    ) -> int:
        """Create command table entry.

        Args:
            signature: Playback signature.
            wave_index: Wave index.
            ignore_already_in_table: Do not error if the signature is already in the table

        Raises:
            AssertionError: Signature already exists in the command table.
            EntryLimitExceededError: If the entries exceed the command table entry limit.
        """
        if not ignore_already_in_table:
            assert signature not in self._table_index_by_signature
        index = len(self._command_table)
        if index > self._device_type.max_ct_entries:
            raise EntryLimitExceededError(
                f"Invalid command table index: '{index}' for device {self._device_type}."
            )

        if signature.increment_phase_params:
            [_first, *rest] = signature.increment_phase_params
            complex_phase_increment = any(r is not None for r in rest)
            for param in signature.increment_phase_params:
                if param is None:
                    continue
                self._parameter_phase_increment_map.setdefault(param, []).append(
                    index if not complex_phase_increment else COMPLEX_USAGE
                )

        ct_entry: dict[str, bool | int | dict] = {"index": index}
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
        self._command_table.append(ct_entry)
        if signature not in self._table_index_by_signature:
            self._table_index_by_signature[signature] = index
        return index

    def _oscillator_config(self, signature: PlaybackSignature) -> dict:
        d: dict[str, int | bool | dict] = {}
        oscillator = signature.hw_oscillator
        if oscillator is not None:
            d["oscillatorSelect"] = {"value": {"$ref": oscillator}}

        ct_phase = None
        do_incr = None
        set_phase = signature.set_phase
        if set_phase is not None:
            assert signature.set_phase == 0.0
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

    def _amplitude_config(self, signature: PlaybackSignature) -> dict:
        d = {}

        assert signature.set_amplitude is None or signature.increment_amplitude is None
        if signature.set_amplitude is not None:
            increment = False
            amplitude = signature.set_amplitude
        elif signature.increment_amplitude is not None:
            increment = True
            amplitude = signature.increment_amplitude
        else:
            return d
        if self._device_type == DeviceType.HDAWG:
            dd = {"value": amplitude}
            if increment:
                dd["increment"] = True
            if signature.amplitude_register is not None:
                dd["register"] = signature.amplitude_register
            d["amplitude0"] = d["amplitude1"] = dd

        elif self._device_type == DeviceType.SHFSG:
            dd = {"value": amplitude}
            if increment:
                dd["increment"] = True
            d["amplitude00"] = d["amplitude10"] = d["amplitude11"] = dd
            d["amplitude01"] = {"value": -amplitude}
            if increment:
                d["amplitude01"]["increment"] = True

        return d

    def command_table(self) -> list[dict]:
        return self._command_table

    def get_or_create_entry(
        self,
        signature: PlaybackSignature,
        wave_index: int | None,
    ) -> int:
        if (idx := self.lookup_index_by_signature(signature)) is not None:
            return idx
        return self.create_entry(signature, wave_index)

    def parameter_phase_increment_map(
        self,
    ) -> dict[str, list[int | Literal[COMPLEX_USAGE]]]:
        return self._parameter_phase_increment_map
