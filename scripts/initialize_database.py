"""
Database Initialization Script
==============================
Initializes tables, sets SQLite WAL pragma, and verifies database integrity.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.database.connection import init_db, engine
from app.database.models import Base
from sqlalchemy import inspect


def main():
    print("=" * 60)
    print("ANPR System: Database Initializer")
    print(f"Target Database: {engine.url}")
    print("=" * 60)

    try:
        init_db()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\n[OK] Successfully initialized database tables:")
        for tbl in tables:
            cols = inspector.get_columns(tbl)
            print(f"  - Table '{tbl}' ({len(cols)} columns)")
        print("\nDatabase initialization complete.")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
