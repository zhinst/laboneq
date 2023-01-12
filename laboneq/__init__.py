# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=wrong-import-order


"""Main functionality of the LabOne Q Software."""

import pkgutil

from laboneq._version import get_version

__path__ = pkgutil.extend_path(__path__, __name__)
__version__ = get_version()
