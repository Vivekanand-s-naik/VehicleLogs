"""
REST API Endpoint Tests
=======================
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.connection import init_db

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_api_db():
    init_db()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert "database_ok" in data


def test_system_status_endpoint():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "health" in data
    assert "camera" in data
    assert "pipeline_metrics" in data


def test_dashboard_summary_endpoint():
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_registered_vehicles" in data
    assert "vehicles_currently_inside" in data


def test_vehicle_endpoints():
    response = client.get("/api/vehicles")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data

    # 404 on non-existent vehicle
    res_404 = client.get("/api/vehicles/NONEXISTENT999")
    assert res_404.status_code == 404


def test_events_endpoint():
    response = client.get("/api/events")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data


def test_camera_status_endpoint():
    response = client.get("/api/camera/status")
    assert response.status_code == 200
    data = response.json()
    assert "source" in data


def test_logs_endpoint():
    response = client.get("/api/system/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_invalid_capture_folder():
    response = client.get("/api/captures/invalid_folder/test.jpg")
    assert response.status_code == 400
