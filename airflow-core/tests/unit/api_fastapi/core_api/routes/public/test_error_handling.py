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
from unittest import mock

import pytest
from fastapi import status
from sqlalchemy import select, update  # Import update

# Assuming Airflow imports are correctly handled by the test environment setup
from airflow.models.dag import DAG
from airflow.models.serialized_dag import SerializedDagModel
from airflow.models.xcom import XCom
from airflow.models.dagrun import DagRun
from airflow.operators.empty import EmptyOperator
from airflow.utils.session import create_session  # Use create_session for safety
from airflow.utils.timezone import datetime

from tests_common.test_utils.db import clear_db_dags, clear_db_runs, clear_db_serialized_dags

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
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

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
        # Should raise HTTPException with status 400
        assert hasattr(excinfo.value, "status_code")
        assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in str(excinfo.value.detail)

    def test_get_dag_run_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/dagRuns endpoint when serialized DAG is missing dag_id.
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

        response = test_client.get(f"/dags/{TEST_DAG_ID}/dagRuns")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_dag_versions_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/dagVersions endpoint when serialized DAG is missing dag_id.
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

        response = test_client.get(f"/dags/{TEST_DAG_ID}/dagVersions")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_extra_links_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/links endpoint when serialized DAG is missing dag_id.
        """
        test_dag = DAG(dag_id=TEST_DAG_ID, start_date=datetime(2025, 4, 15), schedule="@once")
        EmptyOperator(task_id="test_task", dag=test_dag)
        test_dag.sync_to_db()
        SerializedDagModel.write_dag(test_dag, bundle_name=TEST_DAG_ID)
        run_id = "test_run"
        task_id = "test_task"
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
        response = test_client.get(f"/dags/{TEST_DAG_ID}/dagRuns/{run_id}/taskInstances/{task_id}/links")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_tasks_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/tasks endpoint when serialized DAG is missing dag_id.
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
        response = test_client.get(f"/dags/{TEST_DAG_ID}/tasks")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_task_instances_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/taskInstances endpoint when serialized DAG is missing dag_id.
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
        response = test_client.get(f"/dags/{TEST_DAG_ID}/taskInstances")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_xcom_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/xcom/{xcom_key} endpoint when serialized DAG is missing dag_id.
        """
        test_dag = DAG(dag_id=TEST_DAG_ID, start_date=datetime(2025, 4, 15), schedule="@once")
        EmptyOperator(task_id="test_task", dag=test_dag)
        test_dag.sync_to_db()
        SerializedDagModel.write_dag(test_dag, bundle_name=TEST_DAG_ID)
        run_id = "test_run"
        task_id = "test_task"
        xcom_key = "test_key"
        # 建立對應的 DagRun 和 XCom entry
        dagrun = test_dag.create_dagrun(
            run_id=run_id,
            start_date=datetime(2025, 4, 15),
            execution_date=datetime(2025, 4, 15),
            state="success",
        )
        XCom.set(
            key=xcom_key,
            value="test_value",
            execution_date=datetime(2025, 4, 15),
            task_id=task_id,
            dag_id=TEST_DAG_ID,
            run_id=run_id,
        )
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
        response = test_client.get(
            f"/dags/{TEST_DAG_ID}/dagRuns/{run_id}/taskInstances/{task_id}/xcom/{xcom_key}"
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]

    def test_get_log_missing_dag_id_in_serialized_data(self, test_client):
        """
        Test /dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs endpoint when serialized DAG is missing dag_id.
        """
        test_dag = DAG(dag_id=TEST_DAG_ID, start_date=datetime(2025, 4, 15), schedule="@once")
        EmptyOperator(task_id="test_task", dag=test_dag)
        test_dag.sync_to_db()
        SerializedDagModel.write_dag(test_dag, bundle_name=TEST_DAG_ID)
        run_id = "test_run"
        task_id = "test_task"
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
        response = test_client.get(f"/dags/{TEST_DAG_ID}/dagRuns/{run_id}/taskInstances/{task_id}/logs")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "An unexpected error occurred" in response.json()["detail"]
