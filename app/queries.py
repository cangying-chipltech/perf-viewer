"""读取侧聚合查询，供 API 与页面共用。"""

import json
from datetime import datetime, timezone

from .groups import group_configs
from .scope import effective_scenario_ids, parse_config_ids, parse_source_scenario_ids
from .scoring import cached_comparison


def _parse_ts(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _latest_ts(rows):
    latest, latest_dt = None, None
    for r in rows:
        dt = _parse_ts(r["ended_at"] or r["started_at"])
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt, latest = dt, r["ended_at"] or r["started_at"]
    return latest


def _in_clause(ids):
    ids = list(ids)
    if not ids:
        return None, []
    return ",".join("?" * len(ids)), ids


def _parse_config(row: dict) -> dict:
    row["params"] = json.loads(row["params"])
    row["deploy"] = json.loads(row.get("deploy") or "{}")
    return row


def pk_gap_configs(conn, scenario_id: int, config_id: int | None) -> list[dict]:
    """返回场景内有数据、但与给定配置所在聚合组没有共同 case 的聚合配置。"""
    if not config_id:
        return []
    scenario = conn.execute("SELECT config_ids FROM scenario WHERE id = ?",
                            (scenario_id,)).fetchone()
    selected_config_ids = parse_config_ids(scenario["config_ids"] if scenario else None)
    config_ph = ",".join("?" * len(selected_config_ids))
    config_clause = (f" AND r.config_id IN ({config_ph})"
                     if selected_config_ids else "")
    scope_ids = effective_scenario_ids(conn, scenario_id)
    scope_ph = ",".join("?" * len(scope_ids))
    suite_ids = [r["suite_id"] for r in conn.execute(
        f"SELECT DISTINCT suite_id FROM scenario_suite"
        f" WHERE scenario_id IN ({scope_ph})", scope_ids).fetchall()]
    if not suite_ids:
        return []
    ph, params = _in_clause(suite_ids)
    rows = conn.execute(
        f"SELECT DISTINCT r.config_id AS id, c.name, c.params FROM run r"
        f" JOIN config_set c ON c.id = r.config_id"
        f" WHERE r.suite_id IN ({ph}) AND r.valid = 1 AND r.excluded = 0"
        f" AND r.scenario_id IN ({scope_ph}){config_clause}"
        f" ORDER BY r.config_id",
        (*params, *scope_ids, *selected_config_ids)).fetchall()
    groups = group_configs([dict(r) for r in rows])
    candidate = next((g for g in groups
                      if any(c["id"] == config_id for c in g["configs"])), None)
    if candidate is None:
        return []
    gaps = []
    for group in groups:
        if group is candidate:
            continue
        common = None
        for ca in candidate["configs"]:
            for cb in group["configs"]:
                common = conn.execute(
                    f"SELECT 1 FROM case_result a"
                    f" JOIN run ra ON ra.id = a.run_id"
                    f" JOIN case_result b ON b.case_id = a.case_id"
                    f" JOIN run rb ON rb.id = b.run_id"
                    f" WHERE ra.config_id = ? AND rb.config_id = ?"
                    f" AND ra.suite_id IN ({ph}) AND rb.suite_id IN ({ph})"
                    f" AND ra.valid = 1 AND ra.excluded = 0"
                    f" AND rb.valid = 1 AND rb.excluded = 0"
                    f" AND ra.scenario_id IN ({scope_ph})"
                    f" AND rb.scenario_id IN ({scope_ph}) LIMIT 1",
                    (ca["id"], cb["id"], *params, *params,
                     *scope_ids, *scope_ids)).fetchone()
                if common:
                    break
            if common:
                break
        if not common:
            gaps.append({"id": group["id"], "name": group["label"]})
    return gaps


def model_summaries(conn) -> list[dict]:
    models = [dict(r) for r in conn.execute("SELECT * FROM model ORDER BY name").fetchall()]
    if not models:
        return []
    ph, params = _in_clause(m["id"] for m in models)
    scenarios = [dict(r) for r in conn.execute(
        f"SELECT * FROM scenario WHERE model_id IN ({ph}) ORDER BY id", params).fetchall()]
    scen_by_model: dict = {}
    for s in scenarios:
        scen_by_model.setdefault(s["model_id"], []).append(s)
    counts = {r["model_id"]: r["n"] for r in conn.execute(
        f"SELECT model_id, COUNT(*) AS n FROM run WHERE model_id IN ({ph}) GROUP BY model_id",
        params).fetchall()}
    ts_by_model: dict = {}
    for r in conn.execute(
            f"SELECT model_id, ended_at, started_at FROM run WHERE model_id IN ({ph})",
            params).fetchall():
        ts_by_model.setdefault(r["model_id"], []).append(r)
    best_ids = {s["best_config_id"] for s in scenarios if s["best_config_id"]}
    best_map: dict = {}
    if best_ids:
        bph, bparams = _in_clause(best_ids)
        best_map = {r["id"]: _parse_config(dict(r)) for r in conn.execute(
            f"SELECT * FROM config_set WHERE id IN ({bph})", bparams).fetchall()}
    out = []
    for m in models:
        scen_summaries = []
        for s in scen_by_model.get(m["id"], []):
            gaps = pk_gap_configs(conn, s["id"], s["best_config_id"])
            scen_summaries.append({
                "id": s["id"], "name": s["name"], "best_status": s["best_status"],
                "best_config": best_map.get(s["best_config_id"]), "best_reason": s["best_reason"],
                "best_pk_gaps": gaps,
                "best_status_effective": "undetermined" if gaps else s["best_status"],
            })
        out.append({
            "id": m["id"], "name": m["name"], "aliases": json.loads(m["aliases"]),
            "notes": m["notes"], "scenario_count": len(scen_summaries),
            "run_count": counts.get(m["id"], 0),
            "last_updated": _latest_ts(ts_by_model.get(m["id"], [])),
            "scenarios": scen_summaries,
        })
    return out


def model_detail(conn, name: str) -> dict | None:
    row = conn.execute("SELECT * FROM model WHERE name = ?", (name,)).fetchone()
    if not row and name.isdigit():
        row = conn.execute("SELECT * FROM model WHERE id = ?", (int(name),)).fetchone()
    if not row:
        row = next((r for r in conn.execute("SELECT * FROM model").fetchall()
                    if name in json.loads(r["aliases"])), None)
    if not row:
        return None
    m = dict(row)
    scenarios = [dict(r) for r in conn.execute(
        "SELECT * FROM scenario WHERE model_id = ? ORDER BY id", (m["id"],)).fetchall()]
    suites_by_scen: dict = {}
    best_map: dict = {}
    if scenarios:
        sph, sparams = _in_clause(s["id"] for s in scenarios)
        for r in conn.execute(
                f"SELECT ss.scenario_id, su.* FROM suite su"
                f" JOIN scenario_suite ss ON ss.suite_id = su.id"
                f" WHERE ss.scenario_id IN ({sph}) ORDER BY su.id", sparams).fetchall():
            suites_by_scen.setdefault(r["scenario_id"], []).append(dict(r))
        best_ids = {s["best_config_id"] for s in scenarios if s["best_config_id"]}
        if best_ids:
            bph, bparams = _in_clause(best_ids)
            best_map = {r["id"]: _parse_config(dict(r)) for r in conn.execute(
                f"SELECT * FROM config_set WHERE id IN ({bph})", bparams).fetchall()}
    for s in scenarios:
        s["suites"] = suites_by_scen.get(s["id"], [])
        s["best_config"] = best_map.get(s["best_config_id"])
        s["best_pk_gaps"] = pk_gap_configs(conn, s["id"], s["best_config_id"])
        s["best_status_effective"] = ("undetermined" if s["best_pk_gaps"]
                                      else s["best_status"])
    configs = [dict(r) for r in conn.execute(
        "SELECT * FROM config_set WHERE model_id = ? ORDER BY id", (m["id"],)).fetchall()]
    run_counts: dict = {}
    if configs:
        cph, cparams = _in_clause(c["id"] for c in configs)
        run_counts = {r["config_id"]: r["n"] for r in conn.execute(
            f"SELECT config_id, COUNT(*) AS n FROM run WHERE config_id IN ({cph})"
            " GROUP BY config_id", cparams).fetchall()}
    for c in configs:
        _parse_config(c)
        c["run_count"] = run_counts.get(c["id"], 0)
    m["aliases"] = json.loads(m["aliases"])
    m["scenarios"] = scenarios
    m["configs"] = configs
    return m


def run_detail(conn, run_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    run = dict(row)
    run["manifest"] = json.loads(run["manifest"])
    for key, table in (("model", "model"), ("config", "config_set"),
                       ("node", "node"), ("suite", "suite")):
        fk = run[f"{key}_id"]
        r = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (fk,)).fetchone()
        run[key] = dict(r) if r else None
    if run["config"]:
        _parse_config(run["config"])
    results = []
    for cr in conn.execute(
            "SELECT cr.*, tc.case_key, tc.description, tc.params AS case_params, tc.weight"
            " FROM case_result cr JOIN test_case tc ON tc.id = cr.case_id"
            " WHERE cr.run_id = ? ORDER BY tc.sort_order, tc.id", (run_id,)).fetchall():
        item = dict(cr)
        item["metrics"] = json.loads(item["metrics"])
        item["case_params"] = json.loads(item["case_params"])
        results.append(item)
    run["case_results"] = results
    run["artifacts"] = [dict(r) for r in conn.execute(
        "SELECT * FROM artifact WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()]
    return run


def scenario_detail(conn, scenario_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
    if not row:
        return None
    s = dict(row)
    s["scoring"] = json.loads(s["scoring"] or "{}")
    scope_ids = effective_scenario_ids(conn, scenario_id)
    scope_ph = ",".join("?" * len(scope_ids))
    s["source_scenario_ids"] = parse_source_scenario_ids(s.get("source_scenario_ids"))
    s["config_ids"] = parse_config_ids(s.get("config_ids"))
    s["selected_configs"] = [dict(r) for r in conn.execute(
        f"SELECT id, name FROM config_set WHERE id IN ({','.join('?' * len(s['config_ids']))})"
        f" ORDER BY id", s["config_ids"]).fetchall()] if s["config_ids"] else []
    s["source_scenarios"] = [dict(r) for r in conn.execute(
        f"SELECT id, name FROM scenario WHERE id IN ({scope_ph})"
        f" AND id != ? ORDER BY id", (*scope_ids, scenario_id)).fetchall()]
    s["suites"] = [dict(r) for r in conn.execute(
        "SELECT DISTINCT su.* FROM suite su JOIN scenario_suite ss ON ss.suite_id = su.id"
        f" WHERE ss.scenario_id IN ({scope_ph}) ORDER BY su.id", scope_ids).fetchall()]
    if s["best_config_id"]:
        cfg = conn.execute("SELECT * FROM config_set WHERE id = ?",
                           (s["best_config_id"],)).fetchone()
        s["best_config"] = _parse_config(dict(cfg)) if cfg else None
    else:
        s["best_config"] = None
    s["best_pk_gaps"] = pk_gap_configs(conn, scenario_id, s["best_config_id"])
    s["best_status_effective"] = ("undetermined" if s["best_pk_gaps"]
                                  else s["best_status"])
    s["artifacts"] = [dict(r) for r in conn.execute(
        "SELECT * FROM artifact WHERE scenario_id = ? ORDER BY id", (scenario_id,)).fetchall()]
    s["history"] = [dict(r) for r in conn.execute(
        "SELECT * FROM best_history WHERE scenario_id = ? ORDER BY id DESC",
        (scenario_id,)).fetchall()]
    model = conn.execute("SELECT * FROM model WHERE id = ?", (s["model_id"],)).fetchone()
    s["model"] = dict(model) if model else None
    return s


def comparison_view(conn, scenario_id: int, suite_id: int | None, mode: str) -> dict:
    data = dict(cached_comparison(conn, scenario_id, suite_id))
    data["mode"] = mode
    return data