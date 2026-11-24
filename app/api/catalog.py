import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator

from ..db import assign_legacy_scenarios, get_db
from ..entities import ensure_case, ensure_suite, normalize_deploy, resolve_model
from ..scope import parse_config_ids, parse_source_scenario_ids
from ..security import check_token, redact

router = APIRouter(prefix="/api/catalog", dependencies=[Depends(check_token)])


class ModelBody(BaseModel):
    name: str
    aliases: list[str] = []
    notes: str = ""


@router.post("/models")
def create_model(body: ModelBody):
    with get_db() as conn:
        candidates = {body.name, *body.aliases}
        for row in conn.execute("SELECT name, aliases FROM model").fetchall():
            existing = {row["name"], *json.loads(row["aliases"])}
            if existing & candidates:
                raise HTTPException(409, "模型已存在（名称或别名冲突）")
        model = resolve_model(conn, body.name, body.aliases)
        conn.execute("UPDATE model SET notes = ? WHERE id = ?", (body.notes, model["id"]))
    return {"ok": True, "id": model["id"]}


@router.patch("/models/{model_id}")
def update_model(model_id: int, body: ModelBody):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM model WHERE id = ?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(404, "模型不存在")
        aliases = json.dumps(sorted(set(body.aliases) - {body.name}), ensure_ascii=False)
        conn.execute("UPDATE model SET name = ?, aliases = ?, notes = ? WHERE id = ?",
                     (body.name, aliases, body.notes, model_id))
    return {"ok": True}


class ScenarioBody(BaseModel):
    model_id: int | None = None
    model_name: str | None = None
    name: str
    description: str = ""
    suite_keys: list[str] | None = None
    scoring: dict | None = None
    source_scenario_ids: list[int] | None = None
    config_ids: list[int] | None = None

    @field_validator("source_scenario_ids", "config_ids", mode="before")
    @classmethod
    def _coerce_id_list(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = [v for v in value.split(",") if v.strip()]
        elif not isinstance(value, (list, tuple, set)):
            value = [value]
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out


def _validate_sources(conn, model_id: int, source_ids: list[int],
                      self_id: int | None = None) -> list[int]:
    clean: list[int] = []
    for raw in source_ids:
        sid = int(raw)
        if self_id is not None and sid == self_id:
            raise HTTPException(400, "场景不能把自身作为来源场景")
        row = conn.execute("SELECT * FROM scenario WHERE id = ?", (sid,)).fetchone()
        if not row:
            raise HTTPException(400, f"来源场景不存在: {sid}")
        if row["model_id"] != model_id:
            raise HTTPException(400, f"来源场景 {sid} 不属于同一模型")
        if sid not in clean:
            clean.append(sid)
    return clean


def _validate_configs(conn, model_id: int, config_ids: list[int]) -> list[int]:
    clean: list[int] = []
    for raw in config_ids:
        cid = int(raw)
        row = conn.execute("SELECT model_id FROM config_set WHERE id = ?", (cid,)).fetchone()
        if not row:
            raise HTTPException(400, f"配置不存在: {cid}")
        if row["model_id"] != model_id:
            raise HTTPException(400, f"配置 {cid} 不属于同一模型")
        if cid not in clean:
            clean.append(cid)
    return clean


def _link_suites(conn, scenario_id: int, suite_keys: list[str],
                 source_ids: list[int] = ()):
    conn.execute("DELETE FROM scenario_suite WHERE scenario_id = ?", (scenario_id,))
    suite_ids: list[int] = []
    for key in suite_keys:
        suite = conn.execute("SELECT * FROM suite WHERE key = ?", (key,)).fetchone()
        if not suite:
            raise HTTPException(400, f"suite 不存在: {key}")
        suite_ids.append(suite["id"])
    for source_id in source_ids:
        suite_ids.extend(r["suite_id"] for r in conn.execute(
            "SELECT suite_id FROM scenario_suite WHERE scenario_id = ?", (source_id,)))
    for suite_id in dict.fromkeys(suite_ids):
        conn.execute("INSERT OR IGNORE INTO scenario_suite(scenario_id, suite_id)"
                     " VALUES(?,?)", (scenario_id, suite_id))


@router.post("/scenarios")
def create_scenario(body: ScenarioBody):
    with get_db() as conn:
        model_id = body.model_id
        if model_id is None and body.model_name:
            row = conn.execute("SELECT * FROM model WHERE name = ?",
                               (body.model_name,)).fetchone()
            if not row:
                raise HTTPException(400, f"模型不存在: {body.model_name}")
            model_id = row["id"]
        if model_id is None:
            raise HTTPException(400, "需要 model_id 或 model_name")
        source_ids = _validate_sources(conn, model_id, body.source_scenario_ids or [])
        config_ids = _validate_configs(conn, model_id, body.config_ids or [])
        scoring = json.dumps(redact(body.scoring or {}), ensure_ascii=False)
        try:
            cur = conn.execute(
                "INSERT INTO scenario(model_id, name, description, scoring,"
                " source_scenario_ids, config_ids) VALUES(?,?,?,?,?,?)",
                (model_id, body.name, body.description, scoring,
                 json.dumps(source_ids), json.dumps(config_ids)))
        except Exception:
            raise HTTPException(409, "同名场景已存在")
        _link_suites(conn, cur.lastrowid, body.suite_keys or [], source_ids)
        assign_legacy_scenarios(conn)
    return {"ok": True, "id": cur.lastrowid}


@router.patch("/scenarios/{scenario_id}")
def update_scenario(scenario_id: int, body: ScenarioBody):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
        if not row:
            raise HTTPException(404, "场景不存在")
        scoring = json.dumps(redact(body.scoring), ensure_ascii=False) \
            if body.scoring is not None else row["scoring"]
        if body.source_scenario_ids is None:
            source_ids = parse_source_scenario_ids(row["source_scenario_ids"])
        else:
            source_ids = _validate_sources(
                conn, row["model_id"], body.source_scenario_ids, self_id=scenario_id)
        if body.config_ids is None:
            config_ids = parse_config_ids(row["config_ids"])
        else:
            config_ids = _validate_configs(conn, row["model_id"], body.config_ids)
        conn.execute(
            "UPDATE scenario SET name = ?, description = ?, scoring = ?,"
            " source_scenario_ids = ?, config_ids = ? WHERE id = ?",
            (body.name, body.description, scoring, json.dumps(source_ids),
             json.dumps(config_ids), scenario_id))
        if body.suite_keys is None:
            existing = [r["key"] for r in conn.execute(
                "SELECT su.key FROM suite su JOIN scenario_suite ss ON ss.suite_id = su.id"
                " WHERE ss.scenario_id = ?", (scenario_id,)).fetchall()]
            _link_suites(conn, scenario_id, existing, source_ids)
        else:
            _link_suites(conn, scenario_id, body.suite_keys, source_ids)
        assign_legacy_scenarios(conn)
    return {"ok": True}


@router.delete("/scenarios/{scenario_id}")
def delete_scenario(scenario_id: int):
    with get_db() as conn:
        cur = conn.execute("DELETE FROM scenario WHERE id = ?", (scenario_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "场景不存在")
    return {"ok": True}


class CaseBody(BaseModel):
    case_key: str
    description: str = ""
    params: dict = {}
    weight: float = 1.0
    sort_order: int = 0


class SuiteBody(BaseModel):
    key: str
    name: str = ""
    description: str = ""
    cases: list[CaseBody] = []


@router.post("/suites")
def create_suite(body: SuiteBody):
    with get_db() as conn:
        existing = conn.execute("SELECT * FROM suite WHERE key = ?", (body.key,)).fetchone()
        if existing:
            raise HTTPException(409, "suite 已存在")
        suite = ensure_suite(conn, body.key, body.name or body.key, body.description)
        for idx, case in enumerate(body.cases):
            ensure_case(conn, suite["id"], case.case_key, redact(case.params),
                        case.description, case.sort_order or idx)
            if case.weight != 1.0:
                conn.execute("UPDATE test_case SET weight = ? WHERE suite_id = ? AND case_key = ?",
                             (case.weight, suite["id"], case.case_key))
    return {"ok": True, "id": suite["id"]}


@router.patch("/suites/{suite_id}")
def update_suite(suite_id: int, body: SuiteBody):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM suite WHERE id = ?", (suite_id,)).fetchone()
        if not row:
            raise HTTPException(404, "suite 不存在")
        conn.execute("UPDATE suite SET key = ?, name = ?, description = ? WHERE id = ?",
                     (body.key, body.name or body.key, body.description, suite_id))
    return {"ok": True}


@router.post("/suites/{suite_id}/cases")
def create_case(suite_id: int, body: CaseBody):
    with get_db() as conn:
        suite = conn.execute("SELECT * FROM suite WHERE id = ?", (suite_id,)).fetchone()
        if not suite:
            raise HTTPException(404, "suite 不存在")
        existing = conn.execute("SELECT * FROM test_case WHERE suite_id = ? AND case_key = ?",
                                (suite_id, body.case_key)).fetchone()
        if existing:
            raise HTTPException(409, "case 已存在")
        case = ensure_case(conn, suite_id, body.case_key, redact(body.params),
                           body.description, body.sort_order)
        conn.execute("UPDATE test_case SET weight = ? WHERE id = ?", (body.weight, case["id"]))
    return {"ok": True, "id": case["id"]}


@router.patch("/cases/{case_id}")
def update_case(case_id: int, body: CaseBody):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM test_case WHERE id = ?", (case_id,)).fetchone()
        if not row:
            raise HTTPException(404, "case 不存在")
        conn.execute(
            "UPDATE test_case SET case_key = ?, description = ?, params = ?, weight = ?,"
            " sort_order = ? WHERE id = ?",
            (body.case_key, body.description, json.dumps(redact(body.params), ensure_ascii=False),
             body.weight, body.sort_order, case_id))
    return {"ok": True}


@router.delete("/cases/{case_id}")
def delete_case(case_id: int):
    with get_db() as conn:
        used = conn.execute("SELECT COUNT(*) AS n FROM case_result WHERE case_id = ?",
                            (case_id,)).fetchone()["n"]
        if used:
            raise HTTPException(409, f"case 已有 {used} 条结果，不能删除")
        cur = conn.execute("DELETE FROM test_case WHERE id = ?", (case_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "case 不存在")
    return {"ok": True}


class NodeBody(BaseModel):
    label: str
    description: str = ""


@router.post("/nodes")
def create_node(body: NodeBody):
    with get_db() as conn:
        existing = conn.execute("SELECT * FROM node WHERE label = ?", (body.label,)).fetchone()
        if existing:
            raise HTTPException(409, "节点已存在")
        cur = conn.execute("INSERT INTO node(label, description) VALUES(?,?)",
                           (body.label, body.description))
    return {"ok": True, "id": cur.lastrowid}


class ConfigRenameBody(BaseModel):
    name: str
    notes: str = ""
    deploy: dict | None = None


@router.patch("/configs/{config_id}")
def rename_config(config_id: int, body: ConfigRenameBody):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM config_set WHERE id = ?", (config_id,)).fetchone()
        if not row:
            raise HTTPException(404, "配置不存在")
        conn.execute("UPDATE config_set SET name = ?, notes = ? WHERE id = ?",
                     (body.name, body.notes, config_id))
        if body.deploy is not None:
            try:
                deploy = normalize_deploy(body.deploy)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            conn.execute("UPDATE config_set SET deploy = ? WHERE id = ?",
                         (json.dumps(deploy, ensure_ascii=False), config_id))
    return {"ok": True}