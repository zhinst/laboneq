# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from laboneq.data.experiment_description import Parameter


class ExperimentHelper:
    @classmethod
    def get_parameter_values(cls, parameter: Parameter):
        if (
            hasattr(parameter, "count")
            and hasattr(parameter, "start")
            and hasattr(parameter, "stop")
        ):
            return np.linspace(
                start=parameter.start, stop=parameter.stop, num=parameter.count
            )
        else:
            return parameter.values
