"""
FastAPI Main Application
========================
Production REST backend and Web Application Server.
Serves:
- REST APIs (/api/vehicles, /api/events, /api/camera, /api/system, /api/auth)
- Production Responsive Web Application (Desktop & Mobile) at root '/'
- OpenAPI / Swagger interactive documentation at '/docs'
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, BASE_DIR
from app.core.license_check import enforce_license_or_exit, LicenseMonitor
from app.core.logging_config import setup_logging
from app.core.exceptions import ANPRError
from app.database.connection import init_db
from app.services.event_service import EventService
from app.api.routes import vehicles, events, cameras, system, auth

logger = setup_logging()
STATIC_DIR = BASE_DIR / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycles."""
    # Ensure license check runs even when launched directly via `uvicorn app.main:app`
    license_monitor = None
    if settings.LICENSE_CHECK_ENABLED and os.environ.get("_ANPR_LICENSE_VERIFIED_PID") != str(os.getpid()):
        enforce_license_or_exit(force=True)
        license_monitor = LicenseMonitor()
        license_monitor.start()

    logger.info("Initializing database and seeding RBAC users...")
    init_db()

    logger.info("Starting ANPR Vision & Event pipeline...")
    event_service = EventService.get_instance()
    event_service.start()

    yield

    logger.info("Shutting down ANPR Vision & Event pipeline...")
    event_service.stop()
    if license_monitor is not None:
        license_monitor.stop()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-Ready ANPR & Intelligent Vehicle Entry/Exit Management System with RBAC",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Centralized exception handler
@app.exception_handler(ANPRError)
async def anpr_exception_handler(request: Request, exc: ANPRError):
    logger.error(f"Handled ANPR exception: {exc.message} ({exc.details})")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details}
    )


# Mount API Routers
app.include_router(auth.router, prefix="/api")
app.include_router(system.router)
app.include_router(vehicles.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(cameras.router, prefix="/api")


# Mount static directory and root dashboard route
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    def serve_dashboard():
        """Serve production responsive web application."""
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
