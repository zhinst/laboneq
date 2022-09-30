# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from contextlib import contextmanager
from pathlib import Path
import sys


@contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def get_es_version():
    with open(
        os.path.join(Path(__file__).parent.absolute(), "VERSION.txt")
    ) as version_file:
        return version_file.read().rstrip()


def use_es_compiler():
    print(f"use_es_compiler is no longer necessary, ES compiler is used by default")


def use_cpp_compiler():
    raise Exception("cpp compiler is no longer supported")


def init_logging(level=logging.INFO):
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-30s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging.getLogger().info("loging initialized")
    logging.getLogger().debug("debug enabled")
