# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.implementation.data_storage.services.sqlite_dict import (
    DataStorageServiceSqliteDict,
)


class DataStore(DataStorageServiceSqliteDict):
    """Default implementation of a data store.

    Forwards to `DataStorageServiceSqliteDict`.

    !!! version-changed "Deprecated in version 2.43.0"
        The integrated sqlite database component is now deprecated and will be removed in a future version.
    """

    pass
