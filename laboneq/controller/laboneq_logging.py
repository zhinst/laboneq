# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import logging.config
import os

import yaml

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
            format: \"%(asctime)s.%(msecs)03d %(name)-30s %(levelname)-6s %(message)s\"
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

        node_log_handler:
            class: logging.FileHandler
            level: NOTSET
            formatter: node_log_formatter
            filename: \"node.log\"

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


def initialize_logging(performance_log=False, logging_config_dict=None, log_level=None):
    global _logging_initialized
    logdir = get_log_dir()
    if _logging_initialized:
        logging.getLogger(__name__).debug(
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

    logging.config.dictConfig(config)

    performance_log_file = None
    if performance_log:
        performance_log_file = os.path.abspath(
            os.path.join(get_log_dir(), "controller_perf.log")
        )
        performance_handler = logging.FileHandler(performance_log_file, mode="a")
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

    logging.getLogger(__name__).info(
        "Logging initialized from [%s] logdir is %s",
        config_source,
        os.path.abspath(logdir),
    )

    if performance_log:
        logging.getLogger(__name__).info(
            "Performance logging into %s", performance_log_file
        )


def set_level(log_level=logging.INFO):
    initialize_logging(log_level=log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("laboneq").setLevel(log_level)
