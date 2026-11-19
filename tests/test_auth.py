"""
RBAC & Authentication Test Suite
================================
Validates JWT login, role permissions, and route access restrictions.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.connection import init_db

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_auth_db():
    init_db()


def test_login_success_and_roles():
    # 1. Admin login
    res_admin = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res_admin.status_code == 200
    data_admin = res_admin.json()
    assert data_admin["role"] == "ADMIN"
    assert "access_token" in data_admin

    # 2. Guard login
    res_guard = client.post("/api/auth/login", json={"username": "guard", "password": "guard123"})
    assert res_guard.status_code == 200
    assert res_guard.json()["role"] == "OPERATOR"

    # 3. Auditor login
    res_auditor = client.post("/api/auth/login", json={"username": "auditor", "password": "auditor123"})
    assert res_auditor.status_code == 200
    assert res_auditor.json()["role"] == "AUDITOR"


def test_login_invalid_credentials():
    res = client.post("/api/auth/login", json={"username": "admin", "password": "wrongpassword"})
    assert res.status_code == 401


def test_auth_me_endpoint():
    res_login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    token = res_login.json()["access_token"]

    res_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    assert res_me.json()["username"] == "admin"
    assert res_me.json()["role"] == "ADMIN"


def test_rbac_user_management_access():
    # Login as admin and auditor
    admin_token = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).json()["access_token"]
    auditor_token = client.post("/api/auth/login", json={"username": "auditor", "password": "auditor123"}).json()["access_token"]

    # Admin CAN list users
    res_admin = client.get("/api/auth/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin.status_code == 200
    assert len(res_admin.json()) >= 3

    # Auditor CANNOT list users (403 Forbidden)
    res_auditor = client.get("/api/auth/users", headers={"Authorization": f"Bearer {auditor_token}"})
    assert res_auditor.status_code == 403


def test_rbac_manual_entry_access():
    guard_token = client.post("/api/auth/login", json={"username": "guard", "password": "guard123"}).json()["access_token"]
    auditor_token = client.post("/api/auth/login", json={"username": "auditor", "password": "auditor123"}).json()["access_token"]

    # Guard (Operator) CAN perform manual entry
    res_guard = client.post(
        "/api/vehicles/KA01TEST99/manual-entry",
        json={"event_type": "IN", "notes": "Guard test"},
        headers={"Authorization": f"Bearer {guard_token}"}
    )
    assert res_guard.status_code in (200, 400)  # Accepted or processed

    # Auditor CANNOT perform manual entry (403 Forbidden)
    res_auditor = client.post(
        "/api/vehicles/KA01TEST99/manual-entry",
        json={"event_type": "IN", "notes": "Auditor attempt"},
        headers={"Authorization": f"Bearer {auditor_token}"}
    )
    assert res_auditor.status_code == 403


def test_serve_production_ui_at_root():
    # Root '/' should serve HTML production UI
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "ANPR Gate Security" in res.text

