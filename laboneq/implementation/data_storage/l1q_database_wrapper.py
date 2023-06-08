# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.implementation.data_storage_service.data_storage_service_sqlite_dict import (
    DataStorageServiceSqliteDict,
)
from laboneq.interfaces.data_storage.data_storage_api import DataStorageAPI


class L1QDatabase(DataStorageAPI):
    """This is proxy object to access the data DataStorageServiceSqliteDict. It is used to give the user simple access to the data storage api.
    This class is included in the simple.py so that the user can access it using 'from laboneq.simple import *'
    """

    def __init__(self, file_path=None):
        self._data_storage_service = DataStorageServiceSqliteDict(file_path=file_path)

    def __getattribute__(self, attr):
        if hasattr(DataStorageAPI, attr):
            return getattr(self._data_storage_service, attr)
        else:
            return super().__getattribute__(attr)
