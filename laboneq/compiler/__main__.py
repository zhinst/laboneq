# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from .compiler import Compiler
import sys
import json

import os
from pathlib import Path
import logging


def main():

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # logging.Formatter(fmt="%(asctime)s.%(msecs)03d", datefmt="%Y-%m-%d,%H:%M:%S")

    logging.basicConfig(level=logging.INFO)
    # add formatter to ch
    logging.getLogger().handlers[0].setFormatter(formatter)

    filename = sys.argv[1]
    if sys.argv[1] == "-d":
        logging.getLogger().setLevel("DEBUG")
        filename = sys.argv[2]
    else:
        logging.getLogger().setLevel("INFO")

    with open(filename) as f:
        data = json.load(f)

        compiler = Compiler()
        compiler.process_experiment(data)
        Path("awg").mkdir(parents=True, exist_ok=True)
        Path("awg/src").mkdir(parents=True, exist_ok=True)
        Path("awg/waves").mkdir(parents=True, exist_ok=True)

        compiler.generate_code()
        compiler.generate_recipe()
        compiler.write_files()


if __name__ == "__main__":
    main()
