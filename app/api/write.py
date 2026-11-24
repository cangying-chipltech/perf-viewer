import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .. import config
from ..db import get_db
from ..importer import ImportError_, extract_zip, import_runs, now_iso
from ..security import check_token

router = APIRouter(prefix="/api", dependencies=[Depends(check_token)])

_CHUNK = 1024 * 1024


@router.post("/import")
def api_import(files: list[UploadFile] = File(...),
               manifest: UploadFile | None = File(default=None),
               manifest_json: str | None = Form(default=None),
               dry_run: bool = Form(default=False)):
    if manifest is not None:
        raw = manifest.file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(raw) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "manifest 超限")
        try:
            manifest_data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(400, f"manifest.json 解析失败: {exc}")
    elif manifest_json:
        try:
            manifest_data = json.loads(manifest_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"manifest_json 解析失败: {exc}")
    else:
        raise HTTPException(400, "缺少 manifest（multipart 文件 manifest.json 或表单字段 manifest_json）")

    staging = Path(tempfile.mkdtemp(prefix="pv-upload-", dir=str(config.DATA_DIR)))
    try:
        root = staging / "root"
        root.mkdir()
        total = 0
        zip_path = None
        for upload in files:
            name = Path(upload.filename or "file").name
            if not name:
                raise HTTPException(400, "上传文件缺少文件名")
            is_zip = name.lower().endswith(".zip") and len(files) == 1
            dest = (staging / name) if is_zip else (root / name)
            if is_zip:
                zip_path = dest
            with dest.open("wb") as out:
                while True:
                    chunk = upload.file.read(_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > config.MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "上传总量超限")
                    out.write(chunk)
        if zip_path is not None:
            try:
                extract_zip(zip_path, root)
            except ImportError_ as exc:
                raise HTTPException(400, str(exc))
        try:
            report = import_runs(root, manifest_data, source="http", dry_run=dry_run)
        except ImportError_ as exc:
            raise HTTPException(400, str(exc))
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class BestDecision(BaseModel):
    config_id: int
    status: str = "confirmed"
    reason: str = ""


@router.post("/scenarios/{scenario_id}/best")
def api_set_best(scenario_id: int, body: BestDecision):
    if body.status not in ("confirmed", "overridden"):
        raise HTTPException(400, "status 必须是 confirmed 或 overridden")
    if body.status == "overridden" and not body.reason.strip():
        raise HTTPException(400, "改判必填理由")
    with get_db() as conn:
        scenario = conn.execute("SELECT * FROM scenario WHERE id = ?",
                                (scenario_id,)).fetchone()
        if not scenario:
            raise HTTPException(404, "场景不存在")
        cfg = conn.execute("SELECT * FROM config_set WHERE id = ? AND model_id = ?",
                           (body.config_id, scenario["model_id"])).fetchone()
        if not cfg:
            raise HTTPException(400, "config_id 不属于该场景的模型")
        conn.execute(
            "UPDATE scenario SET best_config_id = ?, best_status = ?, best_reason = ?,"
            " best_updated_at = ? WHERE id = ?",
            (body.config_id, body.status, body.reason.strip(), now_iso(), scenario_id),
        )
        conn.execute(
            "INSERT INTO best_history(scenario_id, prev_config_id, new_config_id, status,"
            " reason, created_at) VALUES(?,?,?,?,?,?)",
            (scenario_id, scenario["best_config_id"], body.config_id, body.status,
             body.reason.strip(), now_iso()),
        )
    return {"ok": True, "scenario_id": scenario_id, "best_config_id": body.config_id,
            "status": body.status}


class ExcludeBody(BaseModel):
    excluded: bool = True
    reason: str = ""


@router.post("/runs/{run_id}/exclude")
def api_exclude_run(run_id: int, body: ExcludeBody):
    with get_db() as conn:
        run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "run 不存在")
        manifest = json.loads(run["manifest"])
        if body.reason:
            manifest["exclude_reason"] = body.reason
        conn.execute("UPDATE run SET excluded = ?, manifest = ? WHERE id = ?",
                     (int(body.excluded), json.dumps(manifest, ensure_ascii=False), run_id))
    return {"ok": True, "run_id": run_id, "excluded": body.excluded}


class ScenarioRunsBody(BaseModel):
    run_ids: list[int]


def _require_scenario(conn, scenario_id: int):
    scenario = conn.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
    if not scenario:
        raise HTTPException(404, "场景不存在")
    return scenario


def _validate_run_scope(conn, scenario, run_id: int) -> None:
    run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, f"run {run_id} 不存在")
    if run["model_id"] != scenario["model_id"]:
        raise HTTPException(400, f"run {run_id} 不属于场景所在模型")
    linked = conn.execute(
        "SELECT 1 FROM scenario_suite WHERE scenario_id = ? AND suite_id = ?",
        (scenario["id"], run["suite_id"])).fetchone()
    if not linked:
        raise HTTPException(400, f"run {run_id} 的 suite 未关联到该场景")


@router.post("/scenarios/{scenario_id}/runs")
def api_assign_scenario_runs(scenario_id: int, body: ScenarioRunsBody):
    """把 run 归属到指定场景（覆盖原归属），用于修正无 scenario_id 的历史数据。"""
    run_ids = sorted({int(r) for r in body.run_ids})
    with get_db() as conn:
        scenario = _require_scenario(conn, scenario_id)
        for run_id in run_ids:
            _validate_run_scope(conn, scenario, run_id)
        ph = ",".join("?" * len(run_ids))
        if run_ids:
            conn.execute(f"UPDATE run SET scenario_id = ? WHERE id IN ({ph})",
                         (scenario_id, *run_ids))
    return {"ok": True, "scenario_id": scenario_id, "assigned": len(run_ids)}


class RunScenarioBody(BaseModel):
    scenario_id: int | None = None


@router.post("/runs/{run_id}/scenario")
def api_set_run_scenario(run_id: int, body: RunScenarioBody):
    """设置或清除单个 run 的场景归属；scenario_id=null 表示取消归属。"""
    with get_db() as conn:
        run = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "run 不存在")
        if body.scenario_id is not None:
            scenario = _require_scenario(conn, body.scenario_id)
            _validate_run_scope(conn, scenario, run_id)
        conn.execute("UPDATE run SET scenario_id = ? WHERE id = ?",
                     (body.scenario_id, run_id))
    return {"ok": True, "run_id": run_id, "scenario_id": body.scenario_id}


class MergeBody(BaseModel):
    source_id: int
    target_id: int


@router.post("/configs/merge")
def api_merge_configs(body: MergeBody):
    if body.source_id == body.target_id:
        raise HTTPException(400, "source 与 target 不能相同")
    with get_db() as conn:
        source = conn.execute("SELECT * FROM config_set WHERE id = ?",
                              (body.source_id,)).fetchone()
        target = conn.execute("SELECT * FROM config_set WHERE id = ?",
                              (body.target_id,)).fetchone()
        if not source or not target:
            raise HTTPException(404, "配置不存在")
        if source["model_id"] != target["model_id"]:
            raise HTTPException(400, "只能合并同一模型的配置")
        moved = conn.execute("UPDATE run SET config_id = ? WHERE config_id = ?",
                             (body.target_id, body.source_id)).rowcount
        conn.execute("UPDATE scenario SET best_config_id = ? WHERE best_config_id = ?",
                     (body.target_id, body.source_id))
        conn.execute("DELETE FROM config_set WHERE id = ?", (body.source_id,))
    return {"ok": True, "moved_runs": moved, "target_id": body.target_id}