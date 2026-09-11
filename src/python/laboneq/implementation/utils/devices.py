# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def parse_device_options(device_options: str | None) -> tuple[str | None, list[str]]:
    if device_options is None:
        return None, []
    opts = device_options.upper().split("/")
    if len(opts) > 0 and opts[0] == "":
        opts.pop(0)
    dev_type: str | None = None
    if len(opts) > 0:
        dev_type = opts.pop(0)
    dev_opts = opts
    return dev_type, dev_opts
