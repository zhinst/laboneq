# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from laboneq.core.types.enums import AcquisitionType


class LoaderBase:
    def __init__(self):
        self.root_section_ids: List[str] = []
        self.data = {}
        self.acquisition_type: Optional[AcquisitionType] = None
