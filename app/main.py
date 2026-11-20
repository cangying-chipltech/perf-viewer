from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config
from .api import catalog, read, skills, write
from .db import init_db
from .entities import seed_metric_defs
from .db import get_db


def create_app() -> FastAPI:
    init_db()
    with get_db() as conn:
        seed_metric_defs(conn)
        from .importer import maybe_refresh_signatures
        maybe_refresh_signatures(conn)
    app = FastAPI(title="perf-viewer", version="0.1.0")
    app.include_router(read.router)
    app.include_router(write.router)
    app.include_router(catalog.router)
    app.include_router(skills.router)

    from . import web
    app.include_router(web.router)

    from . import cache

    @app.middleware("http")
    async def invalidate_cache_on_write(request, call_next):
        response = await call_next(request)
        if (request.method != "GET" and request.url.path.startswith("/api")
                and response.status_code < 400):
            cache.bump_generation()
        return response

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/healthz")
    def healthz():
        with get_db() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        return {"ok": True, "schema_version": version,
                "writes_enabled": bool(config.TOKEN)}

    return app


app = create_app()