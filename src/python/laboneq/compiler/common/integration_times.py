# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(init=True, repr=True, order=True)
class SignalIntegrationInfo:
    is_play: bool = False
    length_in_samples: int = None


@dataclass(init=True, repr=True, order=True)
class IntegrationTimes:
    signal_infos: dict[str, SignalIntegrationInfo] = field(default_factory=dict)

    def signal_info(self, signal_id: str) -> SignalIntegrationInfo | None:
        return self.signal_infos.get(signal_id)
