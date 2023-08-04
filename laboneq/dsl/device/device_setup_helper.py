# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import requests
import yaml

_logger = logging.getLogger(__name__)


class DeviceSetupHelper:
    @staticmethod
    def upload_wiring(api_url, wiring_text):
        """Upload wiring information to the LabOne Q monitoring server.

        Args:
            api_url (str): URL of the monitoring server. http://localhost:9005/slugname/wiring
            wiring_text (str): Json-like string contains wiring information.

        Returns:
            status_code (int): 200 if succeeded.
        """

        with requests.Session() as session:
            response = session.post(api_url, data=wiring_text, timeout=5)
            response.raise_for_status()
            _logger.info("Wiring successfully uploaded at %s", api_url)
            return response.status_code

    @staticmethod
    def upload_wiring_from_descriptor(api_url, descriptor):
        """Upload wiring information to the LabOne Q monitoring server using yaml descriptor.

        Args:
            api_url (str): URL of the monitoring server. http://localhost:9005/slugname/wiring
            descriptor (str): yaml-like text contains wiring information.

        Returns:
            status_code (int): 200 if succeeded.
        """
        res = yaml.safe_load(descriptor)
        return DeviceSetupHelper.upload_wiring(api_url, json.dumps(res, indent=4))

    @staticmethod
    def delete_wiring(api_url):
        """Delete wiring information to the LabOne Q monitoring server.

        Args:
            api_url (str): URL of the monitoring server. http://localhost:9005/slugname/wiring

        Returns:
            status_code (int): 200 if succeeded.
        """
        with requests.Session() as session:
            response = session.delete(api_url, timeout=5)
            response.raise_for_status()
            _logger.info("Wiring successfully deleted at %s", api_url)
            return response.status_code

    @staticmethod
    def download_wiring(api_url):
        """Download wiring information from the LabOne Q monitoring server.

        Args:
            api_url (str): URL of the monitoring server. http://localhost:9005/slugname/wiring

        Returns:
            wiring (str):
                the GET content if succeeded.
        """

        with requests.Session() as session:
            response = session.get(api_url, timeout=5)
            response.raise_for_status()
            _logger.info("Successfully downloaded wiring information from %s", api_url)
            return response.text
