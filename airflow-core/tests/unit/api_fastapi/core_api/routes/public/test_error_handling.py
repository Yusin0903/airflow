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

import logging

import pytest
from fastapi import status
from sqlalchemy import select, update

from airflow.models.dag import DAG, DagRun
from airflow.models.serialized_dag import SerializedDagModel
from airflow.operators.empty import EmptyOperator
from airflow.utils.session import create_session
from airflow.utils.timezone import datetime
from airflow.utils.types import DagRunType

from tests_common.test_utils.db import clear_db_dags, clear_db_runs, clear_db_serialized_dags

logger = logging.getLogger(__name__)


# Consider making test_dag_id unique per test run if needed, e.g., using fixtures or uuid
TEST_DAG_ID = "test_dag_missing_dag_id_in_db"
TEST_DAG_RUN_ID = "test_dag_run_missing_dag_id_in_db"
TEST_DAG_ID = "test_dag_missing_dag_id"
TEST_TASK_ID = "test_task"
TEST_RUN_ID = "test_run"
TEST_XCOM_KEY = "test_xcom_key"
TEST_XCOM_VALUE = {"foo": "bar"}

pytestmark = pytest.mark.db_test


class TestDagBagErrorHandling:
    """Unit tests for error handling when using dag_bag.get_dag."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        clear_db_dags()
        clear_db_runs()
        clear_db_serialized_dags()

    def test_get_dag_from_dag_bag_missing_dag_id_in_serialized_data(self, test_client):
        """
        Unit test for get_dag_from_dag_bag: should raise HTTP 400 when serialized DAG is missing dag_id.
        """
        from airflow.api.common.utils import get_dag_from_dag_bag

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

        with pytest.raises(Exception) as excinfo:
            get_dag_from_dag_bag(test_client.app.state.dag_bag, TEST_DAG_ID)
        assert hasattr(excinfo.value, "status_code")
        assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in str(excinfo.value.detail)

    @pytest.mark.parametrize(
        "endpoint,method,body",
        [
            ("/dags/{}/dagRuns/test_run/taskInstances/test_task/listMapped", "get", None),
            ("/dags/{}/dagRuns/test_run/taskInstances", "get", None),
            ("/dags/{}/clearTaskInstances", "post", {"dry_run": True, "dag_run_id": "test_run"}),
        ],
    )
    def test_task_instances_endpoints_missing_dag_id_in_serialized_data(
        self, test_client, endpoint, method, body
    ):
        """
        Test task_instances endpoints when serialized DAG is missing dag_id.
        """
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

        url = endpoint.format(TEST_DAG_ID)
        if method == "get":
            response = test_client.get(url)
        elif method == "post":
            response = test_client.post(url, json=body)
        else:
            pytest.fail(f"Unsupported method: {method}")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_create_xcom_entry_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries endpoint
        when serialized DAG is missing dag_id (should return 400 error).
        """

        test_dag = DAG(dag_id=TEST_DAG_ID, start_date=datetime(2025, 4, 15), schedule="@once")
        EmptyOperator(task_id=TEST_TASK_ID, dag=test_dag)
        test_dag.sync_to_db()
        SerializedDagModel.write_dag(test_dag, bundle_name=TEST_DAG_ID)
        with create_session() as session2:
            dag_model = session2.scalar(
                select(SerializedDagModel).where(SerializedDagModel.dag_id == TEST_DAG_ID)
            )
            if not dag_model:
                pytest.fail("Failed to find serialized DAG in database")
            dag_run = DagRun(
                dag_id=TEST_DAG_ID,
                run_id=TEST_RUN_ID,
                state="queued",
                run_type=DagRunType.MANUAL,
            )
            session2.add(dag_run)
            session2.commit()
            data = dag_model.data
            del data["dag"]["dag_id"]
            session2.execute(
                update(SerializedDagModel).where(SerializedDagModel.dag_id == TEST_DAG_ID).values(_data=data)
            )
            session2.commit()
        request_body = {"key": TEST_XCOM_KEY, "value": TEST_XCOM_VALUE}
        response = test_client.post(
            f"/dags/{TEST_DAG_ID}/dagRuns/{TEST_RUN_ID}/taskInstances/{TEST_TASK_ID}/xcomEntries",
            json=request_body,
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]
