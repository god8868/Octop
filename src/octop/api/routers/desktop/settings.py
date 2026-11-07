"""Remote desktop settings (geometry)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from octop.api.deps import require_permission
from octop.infra.control.desktop.setup import (
    apply_geometry,
    desktop_status,
    parse_geometry,
    read_geometry,
)
from octop.infra.users.identity import User
from octop.infra.utils.offload import Pool, run_blocking

router = APIRouter()


class DesktopGeometryRequest(BaseModel):
    geometry: str = Field(..., description="Width x height, e.g. 1920x1080")

    @field_validator("geometry")
    @classmethod
    def validate_geometry(cls, value: str) -> str:
        parse_geometry(value)
        return value


@router.post("/desktop/geometry")
async def set_desktop_geometry(
    body: DesktopGeometryRequest,
    _user: User = Depends(require_permission("desktop")),
) -> dict[str, object]:
    """Resize the Linux virtual desktop (restarts Xvnc). Admin only."""
    if desktop_status().platform != "linux":
        raise HTTPException(status_code=400, detail="geometry change is only supported on Linux")
    parse_geometry(body.geometry)
    # The resize script restarts Xvnc and is allowed 120 s, so this is not the
    # millisecond-scale class: keep it off the pool that serves DB reads.
    await run_blocking(apply_geometry, body.geometry, pool=Pool.SUBPROCESS, label="apply_geometry")
    w, h = parse_geometry(body.geometry)
    return {"ok": True, "geometry": body.geometry, "width": w, "height": h}


@router.get("/desktop/geometry")
async def get_desktop_geometry(
    _user: User = Depends(require_permission("desktop")),
) -> dict[str, object]:
    geometry = read_geometry()
    w, h = parse_geometry(geometry)
    return {"geometry": geometry, "width": w, "height": h}
