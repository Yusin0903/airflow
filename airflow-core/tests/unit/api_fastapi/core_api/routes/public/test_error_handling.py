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
from __future__ import annotations

from unittest import mock

import pytest
from fastapi import status

from tests_common.test_utils.db import clear_db_dags, clear_db_runs, clear_db_serialized_dags
import json
import logging
import zlib  # Needed for manual compression if updating DB directly
from sqlalchemy import delete, func, select, update  # Import update

# Assuming Airflow imports are correctly handled by the test environment setup
from airflow.models.dag import DAG, DagModel
from airflow.models.serialized_dag import SerializedDagModel
from airflow.operators.empty import EmptyOperator
from airflow.utils.session import create_session  # Use create_session for safety
from airflow.utils.timezone import datetime

logger = logging.getLogger(__name__)


# Consider making test_dag_id unique per test run if needed, e.g., using fixtures or uuid
TEST_DAG_ID = "test_dag_missing_dag_id_in_db"
pytestmark = pytest.mark.db_test


class TestDagBagErrorHandling:
    """Unit tests for error handling when using dag_bag.get_dag."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        clear_db_dags()
        clear_db_runs()
        clear_db_serialized_dags()
        yield
        clear_db_dags()
        clear_db_runs()
        clear_db_serialized_dags()

    def test_get_dag_generic_exception(self, test_client):
        """Test error handling when dag_bag.get_dag raises a generic Exception."""
        # Mock the dag_bag.get_dag method to raise a generic Exception
        with mock.patch.object(
            test_client.app.state.dag_bag, "get_dag", side_effect=Exception("Unexpected error")
        ):
            response = test_client.get("/dags/test_dag")
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_dag_not_found(self, test_client):
        """Test error handling when dag_bag.get_dag returns None."""
        # Mock the dag_bag.get_dag method to return None
        with mock.patch.object(test_client.app.state.dag_bag, "get_dag", return_value=None):
            response = test_client.get("/dags/test_dag")
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "was not found" in response.json()["detail"]

    def test_get_dag_missing_dag_id_in_serialized_data(self, test_client):
        test_dag = DAG(dag_id=TEST_DAG_ID, start_date=datetime(2025, 4, 15), schedule="@once")
        EmptyOperator(task_id="test_task", dag=test_dag)
        test_dag.sync_to_db()
        SerializedDagModel.write_dag(test_dag, bundle_name=TEST_DAG_ID)

        with create_session() as session:
            dag_model = session.scalar(
                select(SerializedDagModel).where(SerializedDagModel.dag_id == TEST_DAG_ID)
            )
            if not dag_model:
                pytest.fail("Failed to find serialized DAG in database")

            data = dag_model.data
            del data["dag"]["dag_id"]
            session.execute(
                update(SerializedDagModel).where(SerializedDagModel.dag_id == TEST_DAG_ID).values(_data=data)
            )
            session.commit()

        response = test_client.get(f"/dags/{TEST_DAG_ID}")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "An unexpected error occurred" in response.json()["detail"]
