"""Localhost server for the atlas viewer, and the static export."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

WEB_DIR = Path(__file__).parent / "web"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LocalOnly:
    """Reject requests whose Host or Origin is not this machine.

    Binding to 127.0.0.1 alone does not stop a web page in the user's browser from
    calling a local port (cross-site requests, DNS rebinding). Checking both headers does.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            host = urlsplit("//" + headers.get("host", "")).hostname
            origin = headers.get("origin")
            if host not in LOCAL_HOSTS or (origin and urlsplit(origin).hostname not in LOCAL_HOSTS):
                await PlainTextResponse("forbidden: local requests only", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(atlas_dir: str | Path) -> LocalOnly:
    atlas_dir = Path(atlas_dir)

    async def atlas_json(_: Request) -> FileResponse:
        return FileResponse(atlas_dir / "atlas.json", media_type="application/json", headers={"Cache-Control": "no-store"})

    async def search_api(request: Request) -> JSONResponse:
        from .atlas import search

        query = request.query_params.get("q", "")[:500]
        return JSONResponse({"query": query, "hits": search(atlas_dir, query) if query.strip() else []})

    async def loop_json(_: Request) -> FileResponse | JSONResponse:
        path = atlas_dir / "loop.json"
        if not path.exists():
            return JSONResponse({"error": "no loop.json; run `nyxcore loop-train` then `nyxcore loop-export`"}, status_code=404)
        return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})

    app = Starlette(routes=[
        Route("/atlas.json", atlas_json),
        Route("/loop.json", loop_json),
        Route("/api/search", search_api),
        Mount("/", StaticFiles(directory=WEB_DIR, html=True)),
    ])
    return LocalOnly(app)


def serve(atlas_dir: str | Path, port: int = 4318) -> None:
    import uvicorn

    if not (Path(atlas_dir) / "atlas.json").exists():
        raise SystemExit(f"no atlas.json in {atlas_dir}; run `nyxcore atlas` first")
    print(f"NYXCORE atlas: http://127.0.0.1:{port}")
    uvicorn.run(build_app(atlas_dir), host="127.0.0.1", port=port, log_level="warning")


def export_static(atlas_dir: str | Path, out: str | Path) -> None:
    """Copy the viewer and its data to a folder any static host can serve (search falls back to text matching)."""
    out = Path(out)
    shutil.copytree(WEB_DIR, out, dirs_exist_ok=True)
    shutil.copy2(Path(atlas_dir) / "atlas.json", out / "atlas.json")
    if (Path(atlas_dir) / "loop.json").exists():
        shutil.copy2(Path(atlas_dir) / "loop.json", out / "loop.json")
