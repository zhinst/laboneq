# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import logging.config
import os

import yaml

from laboneq.core.utilities.compressed_formatter import CompressedFormatter

_logger = logging.getLogger(__name__)
_log_dir = os.path.join("laboneq_output", "log")
_logging_initialized = False


def set_log_dir(dir):
    global _log_dir
    _log_dir = os.path.join(dir, "log")


def get_log_dir():
    global _log_dir
    return _log_dir


DEFAULT_CONFIG_YML = """
    version: 1
    disable_existing_loggers: false

    formatters:
        console_formatter:
            format: \"[%(asctime)s.%(msecs)03d] %(levelname)-7s %(message)s\"
            datefmt: \"%Y.%m.%d %H:%M:%S\"

        file_formatter:
            format: \"%(asctime)s.%(msecs)03d | %(name)-45s | %(levelname)-6s %(message)s\"
            datefmt: \"%Y.%m.%d %H:%M:%S\"

        node_log_formatter:
            format: \"%(asctime)s.%(msecs)03d %(message)s\"
            datefmt: \"%Y.%m.%d %H:%M:%S\"

    handlers:
        console_handler:
            class: logging.StreamHandler
            level: NOTSET
            formatter: console_formatter
            stream: ext://sys.stdout

        error_handler:
            class: logging.StreamHandler
            level: ERROR
            formatter: console_formatter
            stream: ext://sys.stderr

        file_handler:
            class: logging.FileHandler
            level: NOTSET
            formatter: file_formatter
            filename: \"controller.log\"
            encoding: utf-8

        node_log_handler:
            class: logging.FileHandler
            level: NOTSET
            formatter: node_log_formatter
            filename: \"node.log\"
            encoding: utf-8

    root:
        level: NOTSET
        handlers: [error_handler]

    loggers:
        laboneq:
            level: INFO
            handlers: [console_handler, file_handler]

        node.log:
            level: DEBUG
            handlers: [node_log_handler]
"""


def initialize_logging(
    performance_log: bool = False,
    logging_config_dict: dict | None = None,
    log_level: int | str | None = None,
):
    """Configure logging.

    If logging has already been configured, it is configured again.

    The logging configuration is read from one of the following, in order
    of preference:

    - The default log configuration file, `./controller_log_config.yml`.
    - The configuration dictionary passed via the `logging_config_dict`
      parameter.
    - The hardcoded default configuration.

    In the hardcoded default configuration:

    - `laboneq` messages are logged at the `INFO` level to `stdout` and
      to the file `laboneq_output/log/controller.log`.
    - `node.log` messages are logged at the `DEBUG` level to the file
      `laboneq_output/log/node.log`.
    - Other messages are logged at the `ERROR` level to `stderr`.

    Args:
        performance_log:
            If true, timestamped logs are written to
            `laboneq_output/log/controller_perf.log` at the `DEBUG` level.
        logging_config_dict:
            The logging configuration to use. Only used if the
            default log configuration file, `./controller_log_config.yml`
            does not exist.
        log_level:
            If specified, sets the log level for the `laboneq` logger.
    """
    global _logging_initialized
    logdir = get_log_dir()
    if _logging_initialized:
        _logger.debug(
            "Logging has already been initialized - initializing again with logdir %s and console log level %d.",
            logdir,
            log_level,
        )

    _logging_initialized = True
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config = None
    config_source = None
    LOG_CONFIG_FILE = "controller_log_config.yml"
    if os.path.exists(LOG_CONFIG_FILE):
        print("Loading log config from %s", LOG_CONFIG_FILE)
        with open(LOG_CONFIG_FILE, "r") as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        config_source = LOG_CONFIG_FILE
    elif logging_config_dict is not None:
        config = logging_config_dict
        config_source = "Argument logging_config_dict of initialize_logging"
    else:
        config = yaml.safe_load(DEFAULT_CONFIG_YML)
        if "handlers" in config:
            for handler in config["handlers"].values():
                if "filename" in handler:
                    default_filename = handler["filename"]
                    if not os.path.isabs(default_filename):
                        handler["filename"] = os.path.join(logdir, default_filename)
        config_source = f"Default inline config in {__name__}"
        for name, formatter in config["formatters"].items():
            formatter["()"] = functools.partial(
                CompressedFormatter, compress=name == "console_formatter"
            )

    logging.config.dictConfig(config)

    performance_log_file = None
    if performance_log:
        performance_log_file = os.path.abspath(
            os.path.join(get_log_dir(), "controller_perf.log")
        )
        performance_handler = logging.FileHandler(
            performance_log_file, mode="a", encoding="utf-8"
        )
        performance_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d\t%(levelname)s\t%(name)-30s\t%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        performance_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(performance_handler)

    if log_level is not None:
        logging.getLogger("laboneq").setLevel(log_level)

    _logger.info(
        "Logging initialized from [%s] logdir is %s",
        config_source,
        os.path.abspath(logdir),
    )

    if performance_log:
        _logger.info("Performance logging into %s", performance_log_file)


def set_level(log_level: int | str = logging.INFO):
    """Set the logging level for the root and `laboneq` loggers.

    Args:
        log_level:
            The logging level to set.
    """
    initialize_logging(log_level=log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("laboneq").setLevel(log_level)
