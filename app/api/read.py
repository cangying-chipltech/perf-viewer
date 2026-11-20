from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

import json

from .. import config
from ..db import get_db
from ..queries import (comparison_view, model_detail, model_summaries, run_detail,
                       scenario_detail)

router = APIRouter(prefix="/api")


@router.get("/models")
def api_models():
    with get_db() as conn:
        return {"models": model_summaries(conn)}


@router.get("/models/{name:path}/scenarios")
def api_model_scenarios(name: str):
    with get_db() as conn:
        detail = model_detail(conn, name)
    if not detail:
        raise HTTPException(404, "模型不存在")
    return {"scenarios": detail["scenarios"]}


@router.get("/models/{name:path}")
def api_model(name: str):
    with get_db() as conn:
        detail = model_detail(conn, name)
    if not detail:
        raise HTTPException(404, "模型不存在")
    return detail


@router.get("/scenarios/{scenario_id}")
def api_scenario(scenario_id: int):
    with get_db() as conn:
        detail = scenario_detail(conn, scenario_id)
    if not detail:
        raise HTTPException(404, "场景不存在")
    return detail


@router.get("/scenarios/{scenario_id}/comparison")
def api_comparison(scenario_id: int, suite: int | None = Query(default=None),
                   mode: str = Query(default="same-node")):
    if mode not in ("same-node", "adjusted", "best"):
        raise HTTPException(400, "mode 必须是 same-node、adjusted 或 best")
    with get_db() as conn:
        try:
            return comparison_view(conn, scenario_id, suite, mode)
        except KeyError as exc:
            raise HTTPException(404, str(exc))


@router.get("/runs/{run_id}")
def api_run(run_id: int):
    with get_db() as conn:
        detail = run_detail(conn, run_id)
    if not detail:
        raise HTTPException(404, "run 不存在")
    return detail


@router.get("/artifacts/{artifact_id}")
def api_artifact(artifact_id: int, download: bool = Query(default=False)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM artifact WHERE id = ?", (artifact_id,)).fetchone()
    if not row:
        raise HTTPException(404, "产物不存在")
    base = config.ARTIFACTS_DIR.resolve()
    path = (base / row["rel_path"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise HTTPException(404, "产物文件缺失或路径非法")
    media = {
        "md": "text/markdown; charset=utf-8",
        "png": "image/png",
        "csv": "text/csv; charset=utf-8",
        "log": "text/plain; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }.get(row["kind"], "application/octet-stream")
    if row["kind"] == "png" and path.suffix.lower() in (".jpg", ".jpeg"):
        media = "image/jpeg"
    is_svg = row["kind"] == "png" and path.suffix.lower() == ".svg"
    if is_svg:
        media = "image/svg+xml"
    filename = path.name if (download or is_svg) else None
    return FileResponse(path, media_type=media, filename=filename)


@router.get("/catalog")
def api_catalog():
    with get_db() as conn:
        models = [dict(r) for r in conn.execute("SELECT * FROM model ORDER BY name").fetchall()]
        suites = [dict(r) for r in conn.execute("SELECT * FROM suite ORDER BY key").fetchall()]
        for s in suites:
            s["cases"] = [dict(r) for r in conn.execute(
                "SELECT * FROM test_case WHERE suite_id = ? ORDER BY sort_order, id",
                (s["id"],)).fetchall()]
        metrics = [dict(r) for r in conn.execute(
            "SELECT * FROM metric_def ORDER BY rowid").fetchall()]
        nodes = [dict(r) for r in conn.execute("SELECT * FROM node ORDER BY label").fetchall()]
        scenarios = [dict(r) for r in conn.execute(
            "SELECT * FROM scenario ORDER BY model_id, id").fetchall()]
        for s in scenarios:
            s["suite_ids"] = [r["suite_id"] for r in conn.execute(
                "SELECT suite_id FROM scenario_suite WHERE scenario_id = ?", (s["id"],)).fetchall()]
        configs = [dict(r) for r in conn.execute(
            "SELECT * FROM config_set ORDER BY model_id, id").fetchall()]
        for c in configs:
            c["params"] = json.loads(c["params"])
            c["deploy"] = json.loads(c.get("deploy") or "{}")
    return {"models": models, "suites": suites, "metric_defs": metrics,
            "nodes": nodes, "scenarios": scenarios, "configs": configs}


@router.get("/batches/{batch_id}")
def api_batch(batch_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM import_batch WHERE id = ?", (batch_id,)).fetchone()
    if not row:
        raise HTTPException(404, "batch 不存在")
    out = dict(row)
    out["manifest"] = json.loads(out["manifest"])
    out["report"] = json.loads(out["report"])
    return out