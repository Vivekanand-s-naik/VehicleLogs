"""
Top-level Disclosed License Check Facade
========================================
Re-exports the transparent HTTPS license validation module so both
`import license_check` and `from app.core.license_check import ...` work
seamlessly, and supports running a standalone check from the CLI:

    python license_check.py
"""

from app.core.license_check import (
    LicenseCheckResult,
    LicenseMonitor,
    enforce_license_or_exit,
    get_device_id,
    mask_license_key,
    notify_license_blocked,
    verify_license,
)

__all__ = [
    "LicenseCheckResult",
    "LicenseMonitor",
    "enforce_license_or_exit",
    "get_device_id",
    "mask_license_key",
    "notify_license_blocked",
    "verify_license",
]


if __name__ == "__main__":
    res = enforce_license_or_exit(force=True)
    print(f"License verification PASSED (status={res.status}, device={res.device_id[:12]}...)")
