import os
import tempfile
from pathlib import Path

os.environ.setdefault("PERF_VIEWER_DATA_DIR", tempfile.mkdtemp(prefix="pv-test-session-"))
os.environ.setdefault("PERF_VIEWER_TOKEN", "test-token")

import pytest

from app import config
from app.db import get_db, init_db

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "perf_viewer.sqlite3")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "data" / "artifacts")
    monkeypatch.setattr(config, "TOKEN", "test-token")
    from app import cache
    cache.bump_generation()
    init_db()
    yield


@pytest.fixture
def conn():
    with get_db() as c:
        yield c


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def make_catalog(conn, model_name="test/model", suite_key="s1", case_keys=("c1", "c2")):
    from app.entities import ensure_case, ensure_suite, register_config, register_node, resolve_model
    model = resolve_model(conn, model_name)
    suite = ensure_suite(conn, suite_key)
    cases = [ensure_case(conn, suite["id"], k, sort_order=i) for i, k in enumerate(case_keys)]
    return model, suite, cases


def make_run(conn, model, suite, cases, config_name, params, node_label,
             metrics_by_case, valid=True, excluded=False, started_at=None, backend="test",
             scenario_id=None):
    import json
    from app.entities import register_config, register_node
    if scenario_id is None:
        linked = {r["scenario_id"] for r in conn.execute(
            "SELECT DISTINCT scenario_id FROM scenario_suite WHERE suite_id = ?",
            (suite["id"],)).fetchall()}
        if len(linked) == 1:
            scenario_id = linked.pop()
    cfg, _ = register_config(conn, model["id"], config_name, params)
    node = register_node(conn, node_label)
    cur = conn.execute(
        "INSERT INTO run(model_id, config_id, node_id, suite_id, backend, started_at,"
        " valid, excluded, manifest, scenario_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (model["id"], cfg["id"], node["id"], suite["id"], backend, started_at,
         int(valid), int(excluded), "{}", scenario_id),
    )
    run_id = cur.lastrowid
    for case, metrics in zip(cases, metrics_by_case):
        conn.execute(
            "INSERT INTO case_result(run_id, case_id, n_measured, err_count, metrics)"
            " VALUES(?,?,?,?,?)",
            (run_id, case["id"], 1, 0, json.dumps(metrics)),
        )
    return run_id, cfg


def make_scenario(conn, model, suite, name="场景"):
    import json
    cur = conn.execute(
        "INSERT INTO scenario(model_id, name, scoring) VALUES(?,?,?)",
        (model["id"], name, json.dumps({})),
    )
    sid = cur.lastrowid
    conn.execute("INSERT INTO scenario_suite(scenario_id, suite_id) VALUES(?,?)",
                 (sid, suite["id"]))
    linked = conn.execute("SELECT COUNT(DISTINCT scenario_id) n FROM scenario_suite"
                          " WHERE suite_id = ?", (suite["id"],)).fetchone()["n"]
    if linked == 1:
        conn.execute("UPDATE run SET scenario_id = ? WHERE scenario_id IS NULL"
                     " AND suite_id = ?", (sid, suite["id"]))
    return sid