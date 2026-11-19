"""
Disclosed Application License Check & Periodic Validation
=========================================================
Transparent, locally logged HTTPS license verification called at startup
(and optionally on a periodic interval while running).

What this module DOES:
  - Sends an HTTPS POST to LICENSE_SERVER_URL containing ONLY:
      1. `license_key` (configured in .env)
      2. `device_id`   (one-way SHA-256 hash of local machine identifier)
      3. `app_version` (application version string)
  - Proceeds normally when the server returns status "active".
  - Allows a bounded offline grace period (default: 72 hours) after the last
    successful "active" verification if the license server is temporarily unreachable.
  - Displays a clear, visible message (console + Windows dialog when running
    without a console) and exits if the license is "revoked", "expired",
    "suspended", "invalid", or unreachable past the grace period.
  - Records every check in a human-readable local audit log (`logs/license.log`).

What this module explicitly DOES NOT do:
  - No silent data exfiltration, telemetry, file access, keylogging, screen
    capture, camera access, or remote command execution of any kind.
  - Its sole capability is allowing or stopping application execution.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
import requests

from app.core.config import settings

logger = logging.getLogger("anpr.license")


@dataclass
class LicenseCheckResult:
    allowed: bool
    status: str  # "active", "offline_grace", "revoked", "expired", "suspended", "invalid", "unreachable", "missing_key", "invalid_endpoint", "disabled"
    detail: str
    user_message: str
    checked_at_utc: str
    endpoint: str
    device_id: str
    grace_remaining_hours: Optional[float] = None


def mask_license_key(key: str) -> str:
    """Mask license key for transparent local audit logs without exposing the full secret."""
    clean = (key or "").strip()
    if not clean:
        return "<missing>"
    if len(clean) <= 8:
        return clean[:2] + "****"
    return f"{clean[:4]}...{clean[-4:]}"


def get_device_id() -> str:
    """
    Compute a deterministic, non-PII 32-char SHA-256 device fingerprint.
    Combines Windows MachineGuid (when on Windows) with hardware node UUID.
    No usernames, hostnames, IP addresses, or files are collected.
    """
    parts = [platform.system(), str(uuid.getnode())]
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as reg_key:
                machine_guid, _ = winreg.QueryValueEx(reg_key, "MachineGuid")
                if machine_guid:
                    parts.append(str(machine_guid))
        except Exception:
            pass
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _key_hash(license_key: str) -> str:
    return hashlib.sha256(license_key.strip().encode("utf-8")).hexdigest()


def _write_audit_log(
    log_path: Path,
    endpoint: str,
    license_key: str,
    device_id: str,
    status: str,
    allowed: bool,
    detail: str,
    timestamp_utc: str,
) -> None:
    """Append a transparent, human-readable audit record to logs/license.log."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = (
            f"{timestamp_utc} | endpoint={endpoint} | key={mask_license_key(license_key)} | "
            f"device={device_id[:12]} | status={status} | allowed={allowed} | detail={detail}\n"
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.warning(f"Failed to write local license audit log ({log_path}): {exc}")


def _save_grace_cache(cache_path: Path, license_key: str, device_id: str, now_ts: float) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "active",
            "last_active_epoch": now_ts,
            "last_active_utc": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
            "license_key_sha256": _key_hash(license_key),
            "device_id": device_id,
        }
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        logger.warning(f"Could not update license grace cache ({cache_path}): {exc}")


def _clear_grace_cache(cache_path: Path) -> None:
    try:
        if cache_path.exists():
            cache_path.unlink()
    except Exception:
        pass


def _check_offline_grace(
    cache_path: Path,
    license_key: str,
    device_id: str,
    grace_hours: float,
    now_ts: float,
) -> Optional[float]:
    """
    Return remaining grace hours if a valid recent 'active' verification exists
    for this exact license key and device within `grace_hours`; else None.
    """
    if grace_hours <= 0 or not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("status") != "active":
            return None
        if data.get("license_key_sha256") != _key_hash(license_key):
            return None
        if data.get("device_id") != device_id:
            return None
        last_epoch = float(data["last_active_epoch"])
        elapsed_sec = now_ts - last_epoch
        # Reject large negative clock rollbacks (> 5 minutes)
        if elapsed_sec < -300.0:
            return None
        remaining_hours = grace_hours - (max(0.0, elapsed_sec) / 3600.0)
        return remaining_hours if remaining_hours > 0.0 else None
    except Exception:
        return None


def verify_license(
    license_key: Optional[str] = None,
    server_url: Optional[str] = None,
    support_contact: Optional[str] = None,
    grace_hours: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_backoff_seconds: float = 1.0,
    log_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    app_version: Optional[str] = None,
    http_post: Optional[Callable[..., requests.Response]] = None,
    now_epoch: Optional[float] = None,
) -> LicenseCheckResult:
    """
    Validate the application license against the configured HTTPS license server.
    Always logs the check transparently to `logs/license.log`.
    """
    key = (settings.LICENSE_KEY if license_key is None else license_key).strip()
    url = (settings.LICENSE_SERVER_URL if server_url is None else server_url).strip()
    contact = (settings.LICENSE_SUPPORT_CONTACT if support_contact is None else support_contact).strip()
    grace = float(settings.LICENSE_GRACE_HOURS if grace_hours is None else grace_hours)
    timeout = float(settings.LICENSE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds)
    retries = max(1, int(settings.LICENSE_MAX_RETRIES if max_retries is None else max_retries))
    audit_log = Path(log_path) if log_path is not None else Path(settings.LICENSE_LOG_PATH)
    state_cache = Path(cache_path) if cache_path is not None else Path(settings.LICENSE_CACHE_PATH)
    version = app_version or settings.APP_VERSION
    post_fn = http_post or requests.post

    now_ts = time.time() if now_epoch is None else float(now_epoch)
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    device_id = get_device_id()

    # 1. Require a configured license key
    if not key:
        msg = (
            f"No license key is configured for this installation. "
            f"Set LICENSE_KEY in .env or contact {contact} for support."
        )
        _write_audit_log(audit_log, url, key, device_id, "missing_key", False, "LICENSE_KEY is empty", now_iso)
        return LicenseCheckResult(
            allowed=False,
            status="missing_key",
            detail="LICENSE_KEY is empty",
            user_message=msg,
            checked_at_utc=now_iso,
            endpoint=url,
            device_id=device_id,
        )

    # 2. Enforce HTTPS strictly so credentials are never sent in cleartext
    if not url.lower().startswith("https://"):
        msg = (
            f"License server configuration error: endpoint must use HTTPS. "
            f"Contact {contact} for support."
        )
        _write_audit_log(audit_log, url, key, device_id, "invalid_endpoint", False, "Non-HTTPS URL rejected", now_iso)
        return LicenseCheckResult(
            allowed=False,
            status="invalid_endpoint",
            detail="Non-HTTPS license server URL rejected",
            user_message=msg,
            checked_at_utc=now_iso,
            endpoint=url,
            device_id=device_id,
        )

    # 3. Perform bounded HTTPS validation request (minimal payload only)
    payload = {
        "license_key": key,
        "device_id": device_id,
        "app_version": version,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": f"ANPR-LicenseCheck/{version}",
    }

    last_error = ""
    response = None
    for attempt in range(1, retries + 1):
        try:
            resp = post_fn(url, json=payload, headers=headers, timeout=timeout, verify=True)
            # Retry only on transient 5xx server errors; 2xx/4xx are authoritative answers
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                if attempt < retries and retry_backoff_seconds > 0:
                    time.sleep(retry_backoff_seconds * attempt)
                continue
            response = resp
            break
        except requests.RequestException as exc:
            last_error = exc.__class__.__name__
            if attempt < retries and retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * attempt)

    # 4. Handle authoritative server response
    if response is not None:
        body = {}
        try:
            body = response.json() or {}
        except ValueError:
            body = {}

        raw_status = str(body.get("status", "")).strip().lower()
        server_msg = str(body.get("message", "")).strip()

        if response.status_code == 200 and (raw_status == "active" or body.get("active") is True):
            _save_grace_cache(state_cache, key, device_id, now_ts)
            detail = server_msg or "License verified active by server"
            _write_audit_log(audit_log, url, key, device_id, "active", True, detail, now_iso)
            return LicenseCheckResult(
                allowed=True,
                status="active",
                detail=detail,
                user_message="License active.",
                checked_at_utc=now_iso,
                endpoint=url,
                device_id=device_id,
            )

        # Authoritative denial (revoked / expired / suspended / invalid / 401 / 403)
        if raw_status in {"revoked", "expired", "suspended", "invalid", "disabled"}:
            denied_status = raw_status
        elif response.status_code in {401, 403}:
            denied_status = "revoked"
        else:
            denied_status = raw_status or f"denied_http_{response.status_code}"

        # Immediately invalidate any cached grace token on authoritative denial
        _clear_grace_cache(state_cache)

        user_msg = (
            f"This license is no longer active (status: {denied_status}). "
            f"Contact {contact} for support."
        )
        if server_msg:
            user_msg = f"{user_msg}\nDetails: {server_msg}"

        detail = f"Server denied license (HTTP {response.status_code}, status={denied_status})"
        _write_audit_log(audit_log, url, key, device_id, denied_status, False, detail, now_iso)
        return LicenseCheckResult(
            allowed=False,
            status=denied_status,
            detail=detail,
            user_message=user_msg,
            checked_at_utc=now_iso,
            endpoint=url,
            device_id=device_id,
        )

    # 5. Server unreachable after retries -> evaluate bounded offline grace period
    remaining = _check_offline_grace(state_cache, key, device_id, grace, now_ts)
    if remaining is not None:
        detail = (
            f"License server unreachable ({last_error}) after {retries} attempt(s); "
            f"allowing offline grace ({remaining:.1f}h remaining of {grace:.1f}h)"
        )
        _write_audit_log(audit_log, url, key, device_id, "offline_grace", True, detail, now_iso)
        return LicenseCheckResult(
            allowed=True,
            status="offline_grace",
            detail=detail,
            user_message=f"License server unreachable; running within offline grace period ({remaining:.1f} hours remaining).",
            checked_at_utc=now_iso,
            endpoint=url,
            device_id=device_id,
            grace_remaining_hours=remaining,
        )

    # Unreachable and no valid grace period
    user_msg = (
        f"This license could not be verified because the license server is unreachable "
        f"and no valid offline grace period remains. "
        f"Please check your internet connection or contact {contact} for support."
    )
    detail = f"License server unreachable ({last_error}) after {retries} attempt(s); grace period unavailable/expired"
    _write_audit_log(audit_log, url, key, device_id, "unreachable", False, detail, now_iso)
    return LicenseCheckResult(
        allowed=False,
        status="unreachable",
        detail=detail,
        user_message=user_msg,
        checked_at_utc=now_iso,
        endpoint=url,
        device_id=device_id,
    )


def notify_license_blocked(result: LicenseCheckResult, show_dialog: Optional[bool] = None) -> None:
    """
    Display a clear, visible license notice to the user:
      - Writes a formatted notice to stderr when a console is attached.
      - Displays a native Windows MessageBox when running without a console
        (e.g., Nuitka `--windows-console-mode=disable` builds) or when requested.
    """
    banner = (
        "\n"
        + "=" * 72 + "\n"
        + "  APPLICATION LICENSE CHECK FAILED\n"
        + "=" * 72 + "\n"
        + f"  {result.user_message}\n\n"
        + f"  Status:    {result.status}\n"
        + f"  Device ID: {result.device_id[:12]}...\n"
        + f"  Audit Log: {settings.LICENSE_LOG_PATH}\n"
        + "=" * 72 + "\n"
    )
    try:
        if sys.stderr is not None:
            sys.stderr.write(banner)
            sys.stderr.flush()
    except Exception:
        pass

    is_windowed = (
        sys.stdout is None
        or "__compiled__" in globals()
        or getattr(sys, "frozen", False)
    )
    should_show_dialog = is_windowed if show_dialog is None else bool(show_dialog)

    if should_show_dialog and sys.platform == "win32":
        try:
            MB_OK = 0x00000000
            MB_ICONERROR = 0x00000010
            dialog_text = (
                f"{result.user_message}\n\n"
                f"Status: {result.status}\n"
                f"Device ID: {result.device_id[:12]}\n"
                f"Log file: {settings.LICENSE_LOG_PATH}"
            )
            ctypes.windll.user32.MessageBoxW(
                0,
                dialog_text,
                f"{settings.APP_NAME} — License Inactive",
                MB_OK | MB_ICONERROR,
            )
        except Exception:
            pass


def enforce_license_or_exit(force: bool = False) -> LicenseCheckResult:
    """
    Startup entry point called by the application.
    If LICENSE_CHECK_ENABLED is False (and force=False), records a skipped result
    and permits local development execution. Otherwise validates via HTTPS and
    exits with code 1 after displaying a clear visible message if not allowed.
    """
    if not force and not settings.LICENSE_CHECK_ENABLED:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("License check disabled (LICENSE_CHECK_ENABLED=false). Skipping remote check.")
        return LicenseCheckResult(
            allowed=True,
            status="disabled",
            detail="LICENSE_CHECK_ENABLED is false",
            user_message="License check disabled in local configuration.",
            checked_at_utc=now_iso,
            endpoint=settings.LICENSE_SERVER_URL,
            device_id=get_device_id(),
        )

    result = verify_license()
    if result.allowed:
        logger.info(f"License check passed (status={result.status}): {result.detail}")
        return result

    logger.error(f"License check blocked startup (status={result.status}): {result.detail}")
    notify_license_blocked(result)
    sys.exit(1)


class LicenseMonitor:
    """
    Optional periodic background re-check disclosed in LICENSE_NOTICE.txt.
    Re-validates at `interval_seconds` while the server stays running and
    triggers a clean visible shutdown if the license is revoked mid-session.
    """

    def __init__(
        self,
        interval_seconds: Optional[int] = None,
        on_revoked: Optional[Callable[[LicenseCheckResult], None]] = None,
    ):
        self.interval_seconds = max(60, int(interval_seconds or settings.LICENSE_RECHECK_INTERVAL_SECONDS))
        self.on_revoked = on_revoked
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not settings.LICENSE_CHECK_ENABLED:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="LicenseCheckMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Periodic license check monitor started (interval={self.interval_seconds}s).")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            result = verify_license()
            if not result.allowed:
                logger.error(f"Periodic license re-check failed (status={result.status}). Stopping application.")
                notify_license_blocked(result)
                if self.on_revoked is not None:
                    self.on_revoked(result)
                else:
                    os._exit(1)
                return
