"""导入流程：detect/parse → 实体登记 → run/case_result/artifact 落盘 → batch 报告。"""

import hashlib
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .adapters import AdapterError, detect_adapter
from .adapters.base import as_runs
from .db import get_db, get_db_dryrun
from .entities import (ensure_case, ensure_suite, normalize_deploy, register_config,
                       register_node, resolve_model, seed_metric_defs)
from .normalize import NormalizedRun, kind_of_suffix
from .security import redact

MAX_ZIP_MEMBERS = 20000
MAX_ZIP_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

SIGNATURE_VERSION = 2


class ImportError_(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_rel_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ImportError_(f"非法产物路径（路径穿越）: {rel}")
    return p


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ImportError_("zip 成员数超限")
        total = sum(m.file_size for m in members)
        if total > MAX_ZIP_TOTAL_BYTES:
            raise ImportError_("zip 解压总量超限")
        for member in members:
            rel = safe_rel_path(member.filename)
            target = (dest / rel).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ImportError_(f"zip 路径穿越: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _case_payload(case_key: str, metrics: dict, n, err, in_tok, out_tok) -> dict:
    return {
        "case_key": case_key,
        "metrics": {k: v for k, v in sorted(metrics.items())},
        "n": n,
        "err": err,
        "in": in_tok,
        "out": out_tok,
    }


def _signature_payload(model_name: str, fingerprint: str, node: str, suite_key: str,
                       backend: str, cases: list[dict]) -> str:
    payload = {
        "model": model_name,
        "config_fingerprint": fingerprint,
        "node": node,
        "suite_key": suite_key,
        "backend": backend,
        "cases": sorted(cases, key=lambda c: c["case_key"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def run_signature(run: NormalizedRun, fingerprint: str, model_name: str) -> str:
    """run 特征指纹：规范模型名+配置指纹+节点+suite+backend+case 数据。

    使用解析后的规范模型名（而非 manifest 别名），与 refresh_run_signatures 一致；
    不含 manifest 注入的时间戳（CSV 等产物本身无时间），保证同一产物重复导入幂等。
    """
    cases = [_case_payload(c.case_key, c.metrics, c.n_measured, c.err_count, c.in_tok, c.out_tok)
             for c in run.cases]
    return _signature_payload(model_name, fingerprint, run.node_label,
                              run.suite_key, run.backend, cases)


def refresh_run_signatures(conn) -> int:
    """批量重算存量 run 的 signature 列（算法演进后保持幂等判定一致）。

    两次批量查询 + 仅更新变化行；单行 JSON 损坏跳过而非中断启动。
    """
    runs = conn.execute(
        "SELECT r.id, r.backend, r.signature, m.name AS model_name, c.fingerprint AS fp,"
        " n.label AS node_label, s.key AS suite_key"
        " FROM run r JOIN model m ON m.id = r.model_id"
        " JOIN config_set c ON c.id = r.config_id"
        " JOIN node n ON n.id = r.node_id JOIN suite s ON s.id = r.suite_id"
    ).fetchall()
    results = conn.execute(
        "SELECT cr.run_id, tc.case_key, cr.n_measured, cr.err_count, cr.metrics,"
        " cr.in_tok, cr.out_tok FROM case_result cr JOIN test_case tc ON tc.id = cr.case_id"
    ).fetchall()
    by_run: dict = {}
    for cr in results:
        by_run.setdefault(cr["run_id"], []).append(cr)
    updated = 0
    for r in runs:
        try:
            cases = [
                _case_payload(cr["case_key"], json.loads(cr["metrics"]), cr["n_measured"],
                              cr["err_count"], cr["in_tok"], cr["out_tok"])
                for cr in by_run.get(r["id"], [])
            ]
            sig = _signature_payload(r["model_name"], r["fp"], r["node_label"],
                                     r["suite_key"], r["backend"], cases)
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        if r["signature"] != sig:
            conn.execute("UPDATE run SET signature = ? WHERE id = ?", (sig, r["id"]))
            updated += 1
    return updated


def maybe_refresh_signatures(conn) -> int:
    """仅在签名算法版本变化时重算，避免每次启动全量重写历史。"""
    row = conn.execute("SELECT value FROM meta WHERE key = 'signature_version'").fetchone()
    if row and row["value"] == str(SIGNATURE_VERSION):
        return 0
    updated = refresh_run_signatures(conn)
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('signature_version', ?)",
                 (str(SIGNATURE_VERSION),))
    return updated


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ImportError_("manifest 必须是 JSON 对象")
    manifest = redact(manifest)
    model = manifest.get("model")
    name = model.get("name") if isinstance(model, dict) else model
    if not name:
        raise ImportError_("manifest 缺少 model")
    cfg = manifest.get("config")
    if not isinstance(cfg, dict) or not cfg.get("name") or cfg.get("params") is None:
        raise ImportError_("manifest 缺少 config{name,params}：未声明配置的导入拒绝入库")
    if not manifest.get("suite_key"):
        raise ImportError_("manifest 缺少 suite_key")
    return manifest


def _kind_of(path: Path) -> str:
    return kind_of_suffix(path.suffix)


def _store_artifact(conn, batch_dir: Path, root: Path, rel: str, kind: str, title: str,
                    run_id: int | None, scenario_id: int | None, seen: set) -> dict | None:
    rel_path = safe_rel_path(rel)
    src = (root / rel_path).resolve()
    if not src.is_relative_to(root.resolve()):
        raise ImportError_(f"产物路径越界: {rel}")
    if not src.is_file():
        return {"path": rel, "status": "missing"}
    dest = (batch_dir / rel_path).resolve()
    if not dest.is_relative_to(batch_dir.resolve()):
        raise ImportError_(f"产物路径越界: {rel}")
    key = str(rel_path)
    if key not in seen:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        seen.add(key)
    cur = conn.execute(
        "INSERT INTO artifact(run_id, scenario_id, kind, rel_path, title) VALUES(?,?,?,?,?)",
        (run_id, scenario_id, kind, str(batch_dir.name / rel_path), title or rel_path.name),
    )
    return {"path": rel, "status": "stored", "artifact_id": cur.lastrowid}


def import_runs(root: Path, manifest: dict, source: str = "", dry_run: bool = False) -> dict:
    """root 为已就位的产物目录（上传暂存或解压结果）。返回 batch 报告。"""
    manifest = validate_manifest(manifest)
    cfg_deploy_raw = (manifest.get("config") or {}).get("deploy")
    try:
        cfg_deploy = normalize_deploy(cfg_deploy_raw) if cfg_deploy_raw is not None else None
    except ValueError as exc:
        raise ImportError_(str(exc))
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    adapter = detect_adapter(files, manifest)
    runs = as_runs(adapter.parse(root, manifest))
    for run in runs:
        if not run.cases:
            raise ImportError_(f"{adapter.name}: 解析结果不含任何 case")
        run.clean_metrics()

    report: dict = {
        "backend": adapter.name,
        "dry_run": dry_run,
        "runs": [],
        "entities": {"model": None, "configs": [], "nodes": [], "suite": None, "cases_created": []},
        "artifacts": [],
        "errors": [],
    }

    db_ctx = get_db_dryrun() if dry_run else get_db()
    with db_ctx as conn:
        seed_metric_defs(conn)
        batch_id = None
        batch_dir = None
        if not dry_run:
            batch_cur = conn.execute(
                "INSERT INTO import_batch(created_at, source, manifest, status, report) VALUES(?,?,?,?,?)",
                (now_iso(), source, json.dumps(manifest, ensure_ascii=False), "running", "{}"),
            )
            batch_id = batch_cur.lastrowid
            batch_dir = config.ARTIFACTS_DIR / str(batch_id)

        scenario_id = manifest.get("scenario_id")
        if scenario_id is not None:
            row = conn.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
            if not row:
                raise ImportError_(f"scenario_id={scenario_id} 不存在")
            scenario_id = row["id"]

        seen_files: set[str] = set()
        consumed: set[str] = set()
        status = "ok"

        for run in runs:
            entry: dict = {"node": run.node_label, "cases": len(run.cases), "valid": run.valid}
            conn.execute("SAVEPOINT run_sp")
            try:
                model = resolve_model(conn, run.model_name, run.model_aliases)
                cfg_notes = (manifest.get("config") or {}).get("notes")
                cfg, cfg_created = register_config(conn, model["id"], run.config_name,
                                                   redact(run.config_params), notes=cfg_notes,
                                                   deploy=cfg_deploy)
                node = register_node(conn, run.node_label)
                suite = ensure_suite(conn, run.suite_key, manifest.get("suite_name"))
                report["entities"]["model"] = model["name"]
                if cfg_created:
                    report["entities"]["configs"].append(cfg["name"])
                if node["label"] not in report["entities"]["nodes"]:
                    report["entities"]["nodes"].append(node["label"])
                report["entities"]["suite"] = suite["key"]

                sig = run_signature(run, cfg["fingerprint"], model["name"])
                dup = conn.execute(
                    "SELECT id FROM run WHERE model_id = ? AND suite_id = ? AND signature = ?",
                    (model["id"], suite["id"], sig),
                ).fetchone()
                if dup is not None:
                    conn.execute("RELEASE SAVEPOINT run_sp")
                    entry.update({"status": "duplicate", "existing_run_id": dup["id"]})
                    report["runs"].append(entry)
                    status = "partial" if status == "ok" else status
                    continue

                case_manifest = manifest.get("cases") or {}
                case_ids = []
                seen_case_keys: set[str] = set()
                for idx, case in enumerate(run.cases):
                    if case.case_key in seen_case_keys:
                        raise AdapterError(
                            f"重复的 node×case 行: {run.node_label}/{case.case_key}")
                    seen_case_keys.add(case.case_key)
                    decl = case_manifest.get(case.case_key) or {}
                    params = redact(decl.get("params") or case.params)
                    existed = conn.execute(
                        "SELECT id FROM test_case WHERE suite_id = ? AND case_key = ?",
                        (suite["id"], case.case_key),
                    ).fetchone()
                    row = ensure_case(conn, suite["id"], case.case_key, params,
                                      decl.get("description") or case.description, idx)
                    if existed is None:
                        report["entities"]["cases_created"].append(
                            {"case_key": row["case_key"], "suite": suite["key"]})
                    case_ids.append((row, case))

                if dry_run:
                    conn.execute("RELEASE SAVEPOINT run_sp")
                    entry.update({"status": "would-import", "config": cfg["name"],
                                  "fingerprint": cfg["fingerprint"][:12]})
                    report["runs"].append(entry)
                    continue

                run_meta = redact(dict(run.meta))
                run_meta["case_meta"] = {c.case_key: redact(c.meta) for c in run.cases if c.meta}
                run_meta["import_manifest"] = manifest
                batch_dir.mkdir(parents=True, exist_ok=True)
                cur = conn.execute(
                    "INSERT INTO run(model_id, config_id, node_id, suite_id, backend, started_at,"
                    " ended_at, worker_label, valid, excluded, manifest, artifacts_dir, batch_id,"
                    " signature) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (model["id"], cfg["id"], node["id"], suite["id"], run.backend,
                     run.started_at, run.ended_at, run.worker_label, int(run.valid), 0,
                     json.dumps(run_meta, ensure_ascii=False), str(batch_dir), batch_id, sig),
                )
                run_id = cur.lastrowid
                for case_row, case in case_ids:
                    conn.execute(
                        "INSERT INTO case_result(run_id, case_id, n_measured, err_count, metrics,"
                        " in_tok, out_tok) VALUES(?,?,?,?,?,?,?)",
                        (run_id, case_row["id"], case.n_measured, case.err_count,
                         json.dumps(case.metrics, ensure_ascii=False), case.in_tok, case.out_tok),
                    )
                run_arts = []
                for art in run.artifacts:
                    res = _store_artifact(conn, batch_dir, root, art.path, art.kind, art.title,
                                          run_id, None, seen_files)
                    if res:
                        run_arts.append((str(safe_rel_path(art.path)), res))
                conn.execute("RELEASE SAVEPOINT run_sp")
                for rel, res in run_arts:
                    consumed.add(rel)
                    report["artifacts"].append(res)
                entry.update({"status": "imported", "run_id": run_id, "config": cfg["name"],
                              "fingerprint": cfg["fingerprint"][:12]})
            except (AdapterError, ImportError_, sqlite3.IntegrityError) as exc:
                conn.execute("ROLLBACK TO SAVEPOINT run_sp")
                conn.execute("RELEASE SAVEPOINT run_sp")
                entry.update({"status": "failed", "error": str(exc)})
                report["errors"].append(str(exc))
                status = "partial"
            report["runs"].append(entry)

        if not dry_run and any(r.get("run_id") for r in report["runs"]):
            first_run_id = next((r.get("run_id") for r in report["runs"] if r.get("run_id")), None)
            for path in files:
                rel = str(path.relative_to(root))
                if rel in consumed or rel in seen_files:
                    continue
                scope_run = None if scenario_id is not None else first_run_id
                res = _store_artifact(conn, batch_dir, root, rel, _kind_of(path),
                                      path.name, scope_run, scenario_id, seen_files)
                if res:
                    report["artifacts"].append(res)

        if any(r["status"] == "failed" for r in report["runs"]):
            status = "partial" if any(r["status"] in ("imported", "duplicate") for r in report["runs"]) else "failed"
        if not report["runs"]:
            status = "failed"
        report["status"] = status
        if not dry_run:
            conn.execute(
                "UPDATE import_batch SET status = ?, report = ? WHERE id = ?",
                (status, json.dumps(report, ensure_ascii=False), batch_id),
            )
        report["batch_id"] = batch_id
        return report