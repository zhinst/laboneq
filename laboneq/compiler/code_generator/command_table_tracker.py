# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, List, Optional

from laboneq.compiler.code_generator.signatures import PlaybackSignature


class CommandTableTracker:
    def __init__(self):
        self._command_table: Dict[PlaybackSignature, Dict] = {}

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
        if wave_index is None:
            self._command_table[signature] = self._create_playzero_command_table_entry(
                index, signature.waveform.length
            )
        else:
            self._command_table[signature] = self._create_command_table_entry(
                index, wave_index, signature.hw_oscillator
            )
        return index

    @staticmethod
    def _create_command_table_entry(
        ct_index: int, wave_index: int, oscillator: Optional[str]
    ):
        d = {
            "index": ct_index,
            "waveform": {"index": wave_index},
        }
        if oscillator is not None:
            d["oscillatorSelect"] = {"value": {"$ref": oscillator}}
        return d

    @staticmethod
    def _create_playzero_command_table_entry(ct_index: int, length: int):
        d = {
            "index": ct_index,
            "waveform": {"playZero": True, "length": length},
        }
        return d

    def command_table(self) -> List[Dict]:
        return list(self._command_table.values())
