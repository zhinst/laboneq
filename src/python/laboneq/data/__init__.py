# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


class EnumReprMixin:
    name: str

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
