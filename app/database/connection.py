"""
Database Connection & Engine Management
=======================================
Configures SQLite with WAL mode, busy timeout, connection pooling,
session management, and initial RBAC seed data.
"""

import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.database.models import Base, Camera, User
from app.core.security import get_password_hash

logger = logging.getLogger("anpr.database")

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "timeout": settings.DATABASE_TIMEOUT,
        "check_same_thread": False
    }

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.DEBUG
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        try:
            if settings.DATABASE_WAL_MODE:
                cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute(f"PRAGMA busy_timeout={int(settings.DATABASE_TIMEOUT * 1000)};")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to set SQLite PRAGMA: {e}")


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def seed_rbac_users(db: Session):
    """Seed initial role-based users if none exist."""
    default_users = [
        {
            "username": "admin",
            "email": "admin@facility.local",
            "password": "admin123",
            "role": "ADMIN",
            "full_name": "Chief Security Admin"
        },
        {
            "username": "guard",
            "email": "guard@facility.local",
            "password": "guard123",
            "role": "OPERATOR",
            "full_name": "Gate 1 Security Officer"
        },
        {
            "username": "auditor",
            "email": "auditor@facility.local",
            "password": "auditor123",
            "role": "AUDITOR",
            "full_name": "Compliance Auditor"
        }
    ]

    for u_data in default_users:
        existing = db.query(User).filter(User.username == u_data["username"]).first()
        if not existing:
            user = User(
                username=u_data["username"],
                email=u_data["email"],
                hashed_password=get_password_hash(u_data["password"]),
                role=u_data["role"],
                full_name=u_data["full_name"],
                is_active=True
            )
            db.add(user)
    db.commit()


def _ensure_schema_columns():
    """Idempotently add newly introduced columns to existing SQLite tables."""
    try:
        inspector = inspect(engine)
        if "vehicle_events" in inspector.get_table_names():
            cols = {c["name"] for c in inspector.get_columns("vehicle_events")}
            if "video_clip_path" not in cols:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE vehicle_events ADD COLUMN video_clip_path VARCHAR(255)"))
                logger.info("Migrated vehicle_events: added video_clip_path column.")
    except Exception as e:
        logger.warning(f"Schema migration check warning: {e}")


def init_db():
    """Create all tables and ensure default camera and RBAC records exist."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns()
    
    with SessionLocal() as db:
        # Seed default camera
        cam = db.query(Camera).first()
        if not cam:
            cam = Camera(
                camera_name="Main Entry/Exit Camera",
                camera_index=settings.CAMERA_INDEX,
                location="Main Gate",
                status="ACTIVE"
            )
            db.add(cam)
            db.commit()
            logger.info("Seeded default entrance camera record (id=1).")

        # Seed RBAC users
        seed_rbac_users(db)
        logger.info("Seeded default RBAC users: admin, guard, auditor.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency to yield database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def transactional_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction rolled back due to error: {e}", exc_info=True)
        raise
    finally:
        session.close()
