# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from itertools import count

_iid_map = defaultdict(count)


def id_generator(cat: str = "") -> str:
    """Incremental IDs for each category."""
    global _iid_map
    return f"_{cat}_{next(_iid_map[cat])}"
