# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, List, Tuple

_logger = logging.getLogger(__name__)


@dataclass
class Run:
    word: Tuple[Hashable] | str
    count: int

    @property
    def span(self):
        return self.count * len(self.word)


def default_cost_function(r: Run):
    """Some function to grade the fitness of this roll-up.

    This implementation is quite simple: if `r.word` is a string, it will return
    the number of characters saved (minus a fixed penalty for the overhead of the loop
    itself).

    This mostly serves for testing. For the real application (seqc compression) not all
    statements are equally expensive. Some might be loops themselves.
    Custom cost functions will be helpful.
    """
    return -((r.count - 1) * len(r.word) - 2)


def compressor_core(
    plaintext: List[Hashable] | str,
    cost_function: Callable = default_cost_function,
    recurse=False,
):
    if len(plaintext) <= 1:
        return plaintext
    next_seen_map = {}
    offsets = []
    for i, c in list(enumerate(plaintext))[::-1]:
        next_seen_index = next_seen_map.get(c)
        if next_seen_index is None:
            offsets.append(None)
        else:
            offsets.append(next_seen_index - i)
        next_seen_map[c] = i

    offsets = offsets[::-1]

    runs = defaultdict(list)  # word -> List[(start, Run)]
    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        word = tuple(plaintext[index : index + offset])
        if any(start < index <= start + r.span for start, r in runs[word]):
            continue

        run_length = 1
        while index + (run_length + 1) * offset <= len(plaintext) and word == tuple(
            plaintext[index + run_length * offset : index + (run_length + 1) * offset]
        ):
            run_length += 1
        runs[word].append((index, Run(word, run_length)))

    runs_flat = [(start, run) for run_list in runs.values() for start, run in run_list]

    if len(runs_flat) == 0:
        return plaintext

    runs_flat.sort(key=lambda x: cost_function(x[1]))
    best_run_start, best_run = runs_flat[0]
    best_run_end = best_run_start + best_run.span

    if cost_function(best_run) > 0:
        return plaintext

    if recurse:
        best_run.word = compressor_core(best_run.word, cost_function, recurse)

    return [
        *compressor_core(plaintext[:best_run_start], cost_function, recurse),
        best_run,
        *compressor_core(plaintext[best_run_end:], cost_function, recurse),
    ]
