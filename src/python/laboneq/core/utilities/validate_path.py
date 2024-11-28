# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import re

controlled_nodes = [
    (r"/dev[^/]+/qachannels/\d+/oscs/0/gain", "amplitude"),
    (r"/dev[^/]+/sigouts/\d+/delay", "port_delay"),
    (r"/dev[^/]+/qachannels/\d+/spectroscopy/envelope/delay", "port_delay"),
    (r"/dev[^/]+/qachannels/\d+/generator/delay", "port_delay"),
    (r"/dev[^/]+/qachannels/\d+/spectroscopy/delay", "port_delay"),
    (r"/dev[^/]+/qachannels/\d+/readout/integration/delay", "port_delay"),
    (r"/dev[^/]+/qas/0/delay", "port_delay"),
    (r"/dev[^/]+/sigouts/\d+/range", "range"),
    (r"/dev[^/]+/sigins/\d+/range", "range"),
    (r"/dev[^/]+/sgchannels/\d+/output/range", "range"),
    (r"/dev[^/]+/qachannels/\d+/output/range", "range"),
    (r"/dev[^/]+/qachannels/\d+/input/range", "range"),
    (r"/dev[^/]+/qachannels/\d+/oscs/\d+/freq", "oscillator"),
    (r"/dev[^/]+/sgchannels/\d+/oscs/\d+/freq", "oscillator"),
    (r"/dev[^/]+/oscs/\d+/freq", "oscillator"),
    (r"/dev[^/]+/sigouts/\d+/offset", "voltage_offset"),
    (r"/dev[^/]+/sgchannels/\d+/output/rflfpath", "port_mode"),
    (r"/dev[^/]+/awgs/\d+/outputs/\d+/modulation/mode", "mixer_calibration"),
    (r"/dev[^/]+/awgs/\d+/outputs/\d+/gains/\d+", "mixer_calibration"),
    (r"/dev[^/]+/sigouts/\d+/precompensation/.*", "precompensation"),
    (r"/dev[^/]+/synthesizers/\d+/centerfreq", "local_oscillator"),
    (r"/dev[^/]+/sgchannels/\d+/digitalmixer/centerfreq", "local_oscillator"),
]

_logger = logging.getLogger(__name__)


def validate_path(path: str):
    for pattern, calib_param in controlled_nodes:
        if re.match(pattern, path.lower()):
            _logger.warning(
                "The instrument node '%s' you are trying to access is also controlled "
                "through a calibration setting in LabOne Q - to avoid conflicts, it is "
                "recommended to use the '%s' calibration property instead of a direct "
                "node setting.",
                path,
                calib_param,
            )
            break
