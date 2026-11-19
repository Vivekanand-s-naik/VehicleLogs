"""
SQLite Database Backup and Restore Utility
==========================================
Uses SQLite's online backup API for hot, lock-free database backups and restores.
"""

import os
import sys
import sqlite3
import argparse
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings

BACKUPS_DIR = BASE_DIR / "backups"


def get_sqlite_path() -> Path:
    url = settings.DATABASE_URL
    if not url.startswith("sqlite"):
        raise ValueError(f"Backup utility only supports SQLite, got: {url}")
    raw_path = url.replace("sqlite:///", "").replace("sqlite://", "")
    p = Path(raw_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def backup_database() -> Path:
    """Create a hot backup of the active SQLite database."""
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    src_path = get_sqlite_path()
    if not src_path.exists():
        raise FileNotFoundError(f"Database file not found at: {src_path}")

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest_path = BACKUPS_DIR / f"anpr_backup_{timestamp_str}.db"

    print(f"Backing up database: {src_path} -> {dest_path}...")
    src_conn = sqlite3.connect(str(src_path))
    dest_conn = sqlite3.connect(str(dest_path))

    with dest_conn:
        src_conn.backup(dest_conn, pages=100, sleep=0.01)

    src_conn.close()
    dest_conn.close()

    print(f"[OK] Backup successfully created at: {dest_path} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def restore_database(backup_path: Path):
    """Restore SQLite database from a backup file."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found at: {backup_path}")

    target_path = get_sqlite_path()
    print(f"Restoring database from: {backup_path} -> {target_path}...")

    src_conn = sqlite3.connect(str(backup_path))
    dest_conn = sqlite3.connect(str(target_path))

    with dest_conn:
        src_conn.backup(dest_conn, pages=100, sleep=0.01)

    src_conn.close()
    dest_conn.close()

    print(f"[OK] Database successfully restored to: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="ANPR Database Backup and Restore Utility")
    parser.add_argument("--backup", action="store_true", help="Perform database backup")
    parser.add_argument("--restore", type=str, help="Path to backup file to restore")

    args = parser.parse_args()

    if args.backup:
        backup_database()
    elif args.restore:
        restore_database(Path(args.restore))
    else:
        # Default action: backup
        backup_database()


if __name__ == "__main__":
    main()
