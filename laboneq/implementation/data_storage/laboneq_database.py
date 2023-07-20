# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.implementation.data_storage.services.sqlite_dict import (
    DataStorageServiceSqliteDict,
)
from laboneq.interfaces.data_storage.data_storage_api import DataStorageAPI


class DataStore(DataStorageAPI):
    """Proxy object to access the data base.

    Defaults to `DataStorageServiceSqliteDict`.
    """

    def __init__(self, file_path=None):
        self._data_storage_service = DataStorageServiceSqliteDict(file_path=file_path)

    def __getattribute__(self, attr):
        if hasattr(DataStorageAPI, attr):
            return getattr(self._data_storage_service, attr)
        else:
            return super().__getattribute__(attr)
