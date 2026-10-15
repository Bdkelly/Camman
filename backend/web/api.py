"""Authenticated same-origin API; the caller supplies frontend static assets."""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from backend.web.service import ControlError


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["start", "stop", "left", "right", "manual-stop"]


def create_app(service, token, static_dir):
    if len(token) < 24 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("Pairing key must contain at least 24 non-whitespace ASCII characters")

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="Camman control", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed"}, status_code=403
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    def authenticate(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "Enter the pairing key shown by the Camman server")

    @app.exception_handler(ControlError)
    async def control_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(OSError)
    async def transport_error(request, exc):
        service._fail(exc)
        return JSONResponse(
            {"detail": "Controller communication failed; motion disabled"}, status_code=503
        )

    auth = [Depends(authenticate)]

    @app.get("/api/status", dependencies=auth)
    def status():
        return service.status()

    @app.get("/api/preview.jpg", dependencies=auth)
    def preview():
        jpeg, age = service.preview()
        if jpeg is None or age is None or age > 2:
            raise HTTPException(503, "No fresh camera frame")
        return Response(jpeg, media_type="image/jpeg", headers={"X-Frame-Age": f"{age:.3f}"})

    @app.post("/api/control/acquire", dependencies=auth)
    def acquire():
        return service.acquire()

    @app.post("/api/control/heartbeat", dependencies=auth)
    def heartbeat(x_control_lease: str = Header(default="")):
        service.heartbeat(x_control_lease)
        return {"ok": True}

    @app.post("/api/control/release", dependencies=auth)
    def release(x_control_lease: str = Header(default="")):
        service.release(x_control_lease)
        return {"ok": True}

    @app.post("/api/control/command", dependencies=auth)
    def command(body: Command, x_control_lease: str = Header(default="")):
        service.command(body.action, x_control_lease)
        return {"ok": True}

    static_dir = Path(static_dir)

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
