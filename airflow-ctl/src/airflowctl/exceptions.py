#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# Note: Any AirflowException raised is expected to cause the TaskInstance
#       to be marked in an ERROR state
"""Exceptions used by AirflowCtl."""

from __future__ import annotations


class AirflowCtlException(Exception):
    """
    Base class for all AirflowCTL's errors.

    Each custom exception should be derived from this class.
    """


class AirflowCtlNotFoundException(AirflowCtlException):
    """Raise when the requested object/resource is not available in the system."""


class AirflowCtlCredentialNotFoundException(AirflowCtlNotFoundException):
    """Raise when a credential couldn't be found while performing an operation."""


class AirflowCtlConnectionException(AirflowCtlException):
    """Raise when a connection error occurs while performing an operation."""
