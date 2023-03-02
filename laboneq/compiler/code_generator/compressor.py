# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import List

from laboneq.compiler.code_generator.seq_c_generator import SeqCGenerator

_logger = logging.getLogger(__name__)


def compress_generators(
    generators: List[SeqCGenerator], declarations_generator: SeqCGenerator
):
    retval = SeqCGenerator()
    last_hash = None
    compressed_generators = []
    _logger.debug("Compressing generators %s", generators)
    for generator in generators:
        if generator is None:
            raise Exception("None of the generators to be compressed may be None")
        _logger.debug(
            "Generator with %d statements, hash %s",
            generator.num_statements(),
            hash(generator),
        )
        for statement in generator._statements:
            _logger.debug("Statement: %s", statement)
        current_hash = hash(generator)
        if last_hash == current_hash:
            compressed_generators[-1]["count"] += 1
        else:
            compressed_generators.append({"generator": generator, "count": 1})
        last_hash = current_hash
    variable_counter = 0
    for generator_spec in compressed_generators:
        count = generator_spec["count"]
        if count > 1 and variable_counter < 14:
            variable_name = "repeat_count_" + str(variable_counter)
            declarations_generator.add_variable_declaration(variable_name, count)
            retval.add_countdown_loop(variable_name, count, generator_spec["generator"])
            variable_counter += 1
        else:
            retval.append_statements_from(generator_spec["generator"])

    return retval


def compress_generators_rle(
    seq_c_generators: List[SeqCGenerator], declarations_generator: SeqCGenerator
):
    for generator in seq_c_generators:
        if generator is None:
            raise Exception("None of the generators to be compressed may be None")
    hash_pairs = [(hash(generator), generator) for generator in seq_c_generators]

    hashes_back_converter = {hash_pair[0]: hash_pair[1] for hash_pair in hash_pairs}
    hashes = [hash_pair[0] for hash_pair in hash_pairs]

    _logger.debug("Trying to compress generator hashes %s", hashes)

    hashes_dict = list(set(hashes))

    last_hash = None
    for item_hash in hashes:
        if last_hash != item_hash:
            generator = hashes_back_converter[item_hash]
            _logger.debug(
                "-- START %s hash %s", hashes_dict.index(item_hash), item_hash
            )
            for line in generator.generate_seq_c().splitlines():
                _logger.debug("  %s", line)
            _logger.debug(" -- END %s hash %s", hashes_dict.index(item_hash), item_hash)
        last_hash = item_hash

    if len(hashes_dict) >= 400:
        _logger.info(
            "The number of hashes %d is too large for compression, using fallback",
            len(hashes_dict),
        )
        return SeqCGenerator.compress_generators(
            seq_c_generators, declarations_generator
        )
    as_index_sequence = list(map(lambda x: hashes_dict.index(x), hashes))

    _logger.debug(
        "The form of the generators is %s", " ".join(map(str, as_index_sequence))
    )

    mtf_encoded = mtf_encode(as_index_sequence)
    _logger.debug("mtf encoded %s", " ".join(map(str, mtf_encoded)))
    flat_runs = find_runs(mtf_encoded)
    _logger.debug("Found generator runs %s", flat_runs)

    retval = SeqCGenerator()
    good_runs = []

    for top_run in flat_runs:
        if top_run[2] >= 3:
            run_end = top_run[1] + top_run[2]
            run_length = top_run[0] + 1
            num_repeats = round((top_run[2] + 1) / run_length)
            run_start_index = run_end - num_repeats * run_length
            runs_string = as_index_sequence[run_start_index:run_end]
            the_run = as_index_sequence[run_end - run_length : run_end]
            _logger.debug(
                "The runs section from %d to %d contains %d runs of %s  and looks as follows %s",
                run_start_index,
                run_end,
                num_repeats,
                the_run,
                runs_string,
            )
            good_runs.append((run_start_index, run_length, num_repeats))

    index = 0
    variable_added = False
    while True:
        if len(good_runs) == 0 or good_runs[0][0] > index:
            item_hash = hashes[index]
            dict_index = hashes_dict.index(item_hash)
            _logger.debug(
                "Emitting %d at position %d -  generator with hash %s",
                dict_index,
                index,
                item_hash,
            )
            generator = hashes_back_converter[item_hash]
            retval.append_statements_from(generator)
            index += 1
        else:
            run = good_runs.pop(0)

            variable_name = "repeat_count_comp"
            if not variable_added:
                declarations_generator.add_variable_declaration(
                    variable_name, num_repeats
                )
                variable_added = True

            loop_body = SeqCGenerator()
            num_repeats = run[2]
            run_length = run[1]
            the_run_as_hashes = [hashes[index + offset] for offset in range(run_length)]
            _logger.debug(
                "Emitting run with %d of hashes %s at position %d",
                num_repeats,
                the_run_as_hashes,
                index,
            )
            for generator_hash in the_run_as_hashes:
                generator = hashes_back_converter[generator_hash]
                loop_body.append_statements_from(generator)

            retval.add_countdown_loop(variable_name, num_repeats, loop_body)
            index += num_repeats * run_length

        if index >= len(hashes):
            break

    return retval


def mtf_encode(index_sequence: str) -> List[int]:
    dict_set = set()
    dictionary = []
    for c in index_sequence:
        if c not in dict_set:
            dict_set.add(c)
            dictionary.append(c)
    compressed_text = []
    rank = 0
    for c in index_sequence:
        rank = dictionary.index(c)
        compressed_text.append(rank)

        dictionary.pop(rank)
        dictionary.insert(0, c)

    return compressed_text


def find_runs(data):
    last_d = None
    current_run_length = 1
    current_run_start = 0
    runs = []
    for i, d in enumerate(data):
        if last_d is None:
            last_d = d
        elif last_d == d:
            current_run_length += 1
        else:
            current_run = (last_d, current_run_start, current_run_length)
            runs.append(current_run)
            current_run_length = 1
            last_d = d
            current_run_start = i
    current_run = (d, current_run_start, current_run_length)
    runs.append(current_run)
    return runs
