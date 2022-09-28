# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# import logging


class NullLogger:
    def __init__(self):
        pass

    def debug(self, *args, **kwargs):
        pass
