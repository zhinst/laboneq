# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.implementation.data_storage.services.sqlite_dict import (
    DataStorageServiceSqliteDict,
)


class DataStore(DataStorageServiceSqliteDict):
    """Default implementation of a data store.

    Forwards to `DataStorageServiceSqliteDict`.
    """

    pass
