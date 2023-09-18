# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import hashlib
import re


@functools.lru_cache()
def string_sanitize(input):
    """Sanitize the input string, so it can be safely used as (part of) an identifier
    in SeqC."""

    # strip non-ascii characters
    s = input.encode("ascii", "ignore").decode()

    if s == "":
        s = "_"

    # only allowed characters are alphanumeric and underscore
    s = re.sub(r"\W", "_", s)

    # names must not start with a digit
    if s[0].isdigit():
        s = "_" + s

    if s != input:
        s = f"{s}_{hashlib.md5(input.encode()).hexdigest()[:4]}"

    return s
