"""
Tests for Disclosed License Check & Periodic Validation (`license_check.py`)
============================================================================
Covers:
  - Active license acceptance, minimal payload verification, TLS verify=True,
    local audit logging (with masked key), and offline grace cache creation.
  - Revoked and expired license blocking, immediate grace cache invalidation,
    and clear visible user messaging.
  - Missing key and non-HTTPS URL rejection without transmitting secrets.
  - Unreachable server handling: allowed within grace period, blocked after
    grace expiry, blocked when no prior active verification exists.
  - Transient 5xx retry behavior.
  - Startup enforcement exiting cleanly with code 1 when blocked.
  - Presence and content of `LICENSE_NOTICE.txt`.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from license_check import (
    LicenseCheckResult,
    enforce_license_or_exit,
    mask_license_key,
    verify_license,
)


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def test_mask_license_key():
    assert mask_license_key("") == "<missing>"
    assert mask_license_key("SHORT") == "SH****"
    assert mask_license_key("ANPR-LIVE-2026-9988") == "ANPR...9988"


def test_license_active_minimal_payload_tls_and_audit_log(tmp_path):
    log_path = tmp_path / "license.log"
    cache_path = tmp_path / ".license_state.json"
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, verify=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        captured["verify"] = verify
        return _mock_response(200, {"status": "active", "message": "OK"})

    res = verify_license(
        license_key="ANPR-TEST-KEY-1234",
        server_url="https://my-license-server.example.com/api/v1/validate",
        support_contact="support@example.com",
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=fake_post,
    )

    assert res.allowed is True
    assert res.status == "active"

    # Verify strict minimal payload — NO extra/exfiltrated fields!
    assert set(captured["json"].keys()) == {"license_key", "device_id", "app_version"}
    assert captured["json"]["license_key"] == "ANPR-TEST-KEY-1234"
    assert len(captured["json"]["device_id"]) == 32
    assert captured["verify"] is True

    # Verify transparent local audit log exists and masks the full key
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "status=active" in log_text
    assert "allowed=True" in log_text
    assert "ANPR...1234" in log_text
    assert "ANPR-TEST-KEY-1234" not in log_text

    # Verify grace cache created
    assert cache_path.exists()


@pytest.mark.parametrize("bad_status", ["revoked", "expired"])
def test_license_revoked_or_expired_blocks_and_clears_grace_cache(tmp_path, bad_status):
    log_path = tmp_path / "license.log"
    cache_path = tmp_path / ".license_state.json"

    # First, simulate a prior active check to populate grace cache
    verify_license(
        license_key="ANPR-TEST-KEY-5678",
        server_url="https://my-license-server.example.com/api/v1/validate",
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=lambda *a, **kw: _mock_response(200, {"status": "active"}),
    )
    assert cache_path.exists()

    # Now server reports revoked/expired
    res = verify_license(
        license_key="ANPR-TEST-KEY-5678",
        server_url="https://my-license-server.example.com/api/v1/validate",
        support_contact="support@example.com",
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=lambda *a, **kw: _mock_response(200, {"status": bad_status}),
    )

    assert res.allowed is False
    assert res.status == bad_status
    assert "no longer active" in res.user_message.lower()
    assert "support@example.com" in res.user_message
    # Grace cache MUST be wiped immediately upon authoritative revocation/expiry
    assert not cache_path.exists()
    assert f"status={bad_status}" in log_path.read_text(encoding="utf-8")


def test_license_missing_key_and_non_https_rejected_without_network(tmp_path):
    called = {"count": 0}

    def fail_if_called(*a, **kw):
        called["count"] += 1
        raise AssertionError("Network should not be called")

    res_missing = verify_license(
        license_key="",
        server_url="https://my-license-server.example.com/api/v1/validate",
        log_path=tmp_path / "license.log",
        cache_path=tmp_path / ".cache.json",
        http_post=fail_if_called,
    )
    assert res_missing.allowed is False
    assert res_missing.status == "missing_key"

    res_http = verify_license(
        license_key="ANPR-TEST-KEY-0001",
        server_url="http://insecure.example.com/api/v1/validate",
        log_path=tmp_path / "license.log",
        cache_path=tmp_path / ".cache.json",
        http_post=fail_if_called,
    )
    assert res_http.allowed is False
    assert res_http.status == "invalid_endpoint"
    assert called["count"] == 0


def test_license_offline_grace_period_and_expiry(tmp_path):
    log_path = tmp_path / "license.log"
    cache_path = tmp_path / ".license_state.json"
    t0 = 1_750_000_000.0

    # 1. Fresh install, server unreachable -> blocked (no grace cache yet)
    def raise_conn_err(*a, **kw):
        raise requests.ConnectionError("DNS unreachable")

    res_fresh = verify_license(
        license_key="ANPR-GRACE-KEY-1111",
        server_url="https://my-license-server.example.com/api/v1/validate",
        grace_hours=72.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=raise_conn_err,
        now_epoch=t0,
    )
    assert res_fresh.allowed is False
    assert res_fresh.status == "unreachable"

    # 2. Successful online validation at t0
    res_ok = verify_license(
        license_key="ANPR-GRACE-KEY-1111",
        server_url="https://my-license-server.example.com/api/v1/validate",
        grace_hours=72.0,
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=lambda *a, **kw: _mock_response(200, {"status": "active"}),
        now_epoch=t0,
    )
    assert res_ok.allowed is True

    # 3. Unreachable 12 hours later -> allowed within 72h grace period (~60h left)
    res_grace = verify_license(
        license_key="ANPR-GRACE-KEY-1111",
        server_url="https://my-license-server.example.com/api/v1/validate",
        grace_hours=72.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=raise_conn_err,
        now_epoch=t0 + 12 * 3600,
    )
    assert res_grace.allowed is True
    assert res_grace.status == "offline_grace"
    assert res_grace.grace_remaining_hours is not None
    assert 59.5 < res_grace.grace_remaining_hours < 60.5

    # 4. Unreachable 73 hours later -> grace period expired -> blocked
    res_expired_grace = verify_license(
        license_key="ANPR-GRACE-KEY-1111",
        server_url="https://my-license-server.example.com/api/v1/validate",
        grace_hours=72.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
        log_path=log_path,
        cache_path=cache_path,
        http_post=raise_conn_err,
        now_epoch=t0 + 73 * 3600,
    )
    assert res_expired_grace.allowed is False
    assert res_expired_grace.status == "unreachable"


def test_license_transient_5xx_retries_and_succeeds(tmp_path):
    attempts = {"n": 0}

    def flaky_post(*a, **kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _mock_response(503, {"error": "temporary overload"})
        return _mock_response(200, {"status": "active"})

    res = verify_license(
        license_key="ANPR-RETRY-KEY-2222",
        server_url="https://my-license-server.example.com/api/v1/validate",
        max_retries=3,
        retry_backoff_seconds=0.0,
        log_path=tmp_path / "license.log",
        cache_path=tmp_path / ".cache.json",
        http_post=flaky_post,
    )
    assert res.allowed is True
    assert attempts["n"] == 3


def test_enforce_license_exits_with_code_1_when_revoked():
    denied = LicenseCheckResult(
        allowed=False,
        status="revoked",
        detail="Server denied license",
        user_message="This license is no longer active. Contact support@example.com for support.",
        checked_at_utc="2026-11-19T00:00:00Z",
        endpoint="https://my-license-server.example.com/api/v1/validate",
        device_id="abcdef1234567890abcdef1234567890",
    )
    with patch("app.core.license_check.verify_license", return_value=denied), \
         patch("app.core.license_check.notify_license_blocked") as mock_notify:
        with pytest.raises(SystemExit) as exc_info:
            enforce_license_or_exit(force=True)
        assert exc_info.value.code == 1
        mock_notify.assert_called_once_with(denied)


def test_license_notice_bundled_and_transparent():
    notice_path = BASE_DIR / "LICENSE_NOTICE.txt"
    assert notice_path.exists(), "LICENSE_NOTICE.txt must exist for transparent user disclosure"
    text = notice_path.read_text(encoding="utf-8").lower()
    assert "periodic" in text
    assert "revoked" in text or "disabled" in text
    assert "logs/license.log" in text
    assert "does not" in text
