# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class CompressableLogEntry:
    header: str
    messages: list[str]
    max_messages: int = 3

    def generate_log_messages(self, compress: bool) -> list[str]:
        compress = compress and len(self.messages) > self.max_messages
        messages = [self.header] if self.header else []
        if compress:
            messages += self.messages[: self.max_messages - 1] + [
                f"  [{len(self.messages) - self.max_messages + 1} similar messages suppressed, see log file for full set of messages.]"
            ]
        else:
            messages += self.messages
        return messages

    def __str__(self) -> str:
        """Return a string representation, mainly for unit testing"""
        return "\n".join(self.generate_log_messages(False))


class CompressedFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style="%",
        validate: bool = True,
        compress: bool = False,
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(fmt, datefmt, style, validate)
        self.compress = compress

    def format(self, record):
        if isinstance(record.msg, CompressableLogEntry):
            messages = []
            record_copy = copy(record)
            for m in record.msg.generate_log_messages(self.compress):
                record_copy.msg = m
                messages.append(super().format(record_copy))
            return "\n".join(messages)
        return super().format(record)
