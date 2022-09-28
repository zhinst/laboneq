# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.controller.util import LabOneQControllerException, SimpleProxy


class ProtectedSession(SimpleProxy):
    def disconnect(self):
        raise LabOneQControllerException(
            "'disconnect' is not allowed from the user function."
        )
