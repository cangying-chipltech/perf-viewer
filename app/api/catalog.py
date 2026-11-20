import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..db import get_db
from ..entities import ensure_case, ensure_suite, normalize_deploy, resolve_model
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
    suite_keys: list[str] = []
    scoring: dict | None = None


def _link_suites(conn, scenario_id: int, suite_keys: list[str]):
    conn.execute("DELETE FROM scenario_suite WHERE scenario_id = ?", (scenario_id,))
    for key in suite_keys:
        suite = conn.execute("SELECT * FROM suite WHERE key = ?", (key,)).fetchone()
        if not suite:
            raise HTTPException(400, f"suite 不存在: {key}")
        conn.execute("INSERT INTO scenario_suite(scenario_id, suite_id) VALUES(?,?)",
                     (scenario_id, suite["id"]))


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
        scoring = json.dumps(redact(body.scoring or {}), ensure_ascii=False)
        try:
            cur = conn.execute(
                "INSERT INTO scenario(model_id, name, description, scoring) VALUES(?,?,?,?)",
                (model_id, body.name, body.description, scoring))
        except Exception:
            raise HTTPException(409, "同名场景已存在")
        _link_suites(conn, cur.lastrowid, body.suite_keys)
    return {"ok": True, "id": cur.lastrowid}


@router.patch("/scenarios/{scenario_id}")
def update_scenario(scenario_id: int, body: ScenarioBody):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
        if not row:
            raise HTTPException(404, "场景不存在")
        scoring = json.dumps(redact(body.scoring), ensure_ascii=False) \
            if body.scoring is not None else row["scoring"]
        conn.execute("UPDATE scenario SET name = ?, description = ?, scoring = ? WHERE id = ?",
                     (body.name, body.description, scoring, scenario_id))
        if body.suite_keys is not None:
            _link_suites(conn, scenario_id, body.suite_keys)
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