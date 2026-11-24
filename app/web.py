"""页面路由：Jinja2 + htmx + ECharts，浏览免认证。"""

import json
import re
import shlex
from pathlib import Path
from urllib.parse import quote

import markdown as md
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.templating import Jinja2Templates

from . import config
from .audit import deploy_audit
from .db import get_db
from .entities import normalize_params, parse_serve_flags
from .groups import group_configs
from .queries import model_detail, model_summaries, run_detail, scenario_detail
from .repro import resolve_repro
from .scope import effective_scenario_ids, parse_config_ids, parse_source_scenario_ids
from .scoring import _compare, cached_comparison
from .security import sanitize_html

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["model_url"] = lambda name: "/models/" + quote(name, safe="")
templates.env.globals["asset_version"] = str(int(
    (Path(__file__).parent / "static" / "app.js").stat().st_mtime))


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _short(value, limit=72):
    if value is None:
        return "—"
    if isinstance(value, list):
        text = ", ".join(str(v) for v in value)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit - 1] + "…"


templates.env.filters["fmt"] = _fmt
templates.env.filters["short"] = _short


def render_markdown(text: str, artifacts: list[dict]) -> str:
    html = md.markdown(text, extensions=["tables", "fenced_code"])
    for art in artifacts:
        basename = Path(art["rel_path"]).name
        url = f"/api/artifacts/{art['id']}"
        html = re.sub(
            r'(src|href)="([^"]*/)?' + re.escape(basename) + r'"',
            rf'\1="{url}"', html)
    return sanitize_html(html)


def artifact_path(rel_path: str) -> Path | None:
    base = config.ARTIFACTS_DIR.resolve()
    path = (base / rel_path).resolve()
    if not path.is_relative_to(base):
        return None
    return path


def _mark(score: float | None) -> str:
    if score is None:
        return ""
    if score > 0.5:
        return "win"
    if score < 0.5:
        return "loss"
    return "tie"


def _case_decision_notes(case_out: dict) -> list[str]:
    """单个测试项无法判定最优时的说明；不要求用例全覆盖。"""
    notes: list[str] = []
    cfg_ids = case_out.get("configs_with_data") or []
    pairwises = case_out.get("pairwise") or []
    if not cfg_ids:
        return ["暂无有效运行数据，本测试项不参与横向对比。"]
    if len(cfg_ids) == 1:
        return ["仅一套配置有该测试项数据，本测试项不参与横向对比。"]
    if not pairwises:
        return ["配置间无共同指标，本测试项不参与横向对比。"]
    scores = [pw.get("case_score") for pw in pairwises if pw.get("case_score") is not None]
    if scores and all(abs(score - 0.5) <= 1e-9 for score in scores):
        notes.append("各配置在该测试项打平，无法判定最优；建议补充重复运行或统一节点后复测。")
    details = [d for pw in pairwises for d in (pw.get("metrics") or {}).values()]
    if details and all(d.get("node_mode") is None for d in details):
        notes.append("无共同节点且缺少节点校正证据，仅按最优值比较；"
                     "建议补充同节点测试以提高结论可信度。")
    return notes


def build_table(comp: dict, suite_out: dict, mode: str) -> list[dict]:
    """case×metric 行 × 配置列 的展示视图模型。

    mode:
    - same-node：原始节点代表值均值；标记优先用同节点胜负，无共同节点时用节点校正胜负。
    - adjusted：把各节点代表值折算到同一基准后的值，用于跨节点配置的横向对比。
    - best：历史最好值。
    """
    epsilon = float(comp["scoring"]["epsilon"])
    metric_defs = comp["metric_defs"]
    configs = comp["configs"]
    rows = []
    for case_out in suite_out["cases"]:
        case = case_out["case"]
        cfg_ids = case_out["configs_with_data"]
        notes = _case_decision_notes(case_out)
        if not cfg_ids:
            rows.append({"case": case, "metric": None, "cells": [], "empty": True,
                         "notes": notes})
            continue
        for metric in case_out["metric_keys"]:
            direction = metric_defs[metric]["direction"]
            cells = []
            raw_values = []
            for cid in cfg_ids:
                cell = case_out["cells"].get(cid, {}).get(metric)
                if not cell:
                    cells.append({"config_id": cid, "name": configs[cid]["name"],
                                  "value": None, "node_mode": None})
                    raw_values.append(None)
                    continue
                if mode == "best":
                    rep = cell["best"]
                    value = rep["value"]
                    n_label = f"{rep['n_runs']} runs"
                    rng = (rep["min"], rep["max"])
                    run_ids = rep["run_ids"]
                    node_mode = None
                    adjusted_value = None
                elif mode == "adjusted":
                    adjusted_reps = [r for r in cell["same"].values()
                                     if r.get("adjusted") is not None]
                    if not adjusted_reps:
                        value = None
                        n_label = "无可校正节点"
                        rng = (None, None)
                        run_ids = []
                        node_mode = None
                        adjusted_value = None
                    else:
                        value = sum(r["adjusted"] for r in adjusted_reps) / len(adjusted_reps)
                        n_label = f"{len(adjusted_reps)} 节点校正"
                        rng = (min(r["adjusted"] for r in adjusted_reps),
                               max(r["adjusted"] for r in adjusted_reps))
                        run_ids = sorted({rid for r in adjusted_reps for rid in r["run_ids"]})
                        node_mode = "adjusted"
                        adjusted_value = value
                else:
                    node_reps = list(cell["same"].values())
                    value = sum(r["value"] for r in node_reps) / len(node_reps)
                    n_label = f"{len(node_reps)} 节点"
                    rng = (min(r["value"] for r in node_reps),
                           max(r["value"] for r in node_reps))
                    run_ids = sorted({rid for r in node_reps for rid in r["run_ids"]})
                    adjusted_reps = [r for r in node_reps if r.get("adjusted") is not None]
                    adjusted_value = (sum(r["adjusted"] for r in adjusted_reps)
                                      / len(adjusted_reps)) if adjusted_reps else None
                    node_mode = None
                cells.append({
                    "config_id": cid, "name": configs[cid]["name"],
                    "value": value, "n_label": n_label, "min": rng[0], "max": rng[1],
                    "run_ids": run_ids, "node_mode": node_mode,
                    "adjusted_value": adjusted_value,
                })
                raw_values.append(value)
            present = [v for v in raw_values if v is not None]
            row_best = (min(present) if direction == "lower" else max(present)) if present else None
            for cell, value in zip(cells, raw_values):
                if value is None or row_best is None:
                    cell["mark"] = ""
                    cell["score"] = None
                    continue
                if mode in ("best", "adjusted"):
                    score = _compare(value, row_best, direction, epsilon)
                else:
                    scores = []
                    for pw in case_out["pairwise"]:
                        other = pw["b"] if pw["a"] == cell["config_id"] else (
                            pw["a"] if pw["b"] == cell["config_id"] else None)
                        if other is None:
                            continue
                        detail = pw["metrics"].get(metric)
                        if not detail or detail["win_rate_node"] is None:
                            continue
                        s = detail["win_rate_node"]
                        if detail.get("node_mode") == "adjusted":
                            cell["node_mode"] = "adjusted"
                        scores.append(s if pw["a"] == cell["config_id"] else 1.0 - s)
                    score = sum(scores) / len(scores) if scores else None
                cell["score"] = score
                cell["mark"] = _mark(score)
            rows.append({"case": case, "metric": metric,
                         "metric_def": metric_defs[metric], "cells": cells, "empty": False,
                         "notes": notes, "repro": case_out.get("repro") or {}})
    return rows


def build_pk_view(comp: dict, a_id: int, b_id: int) -> dict:
    """两配置跨 suite 的 case 级 PK：只统计双方都有数据且有共同指标的测试项。"""
    configs = comp["configs"]
    if a_id not in configs or b_id not in configs:
        return {"a": None, "b": None, "comparisons": [], "compared_count": 0,
                "coverage_a": [], "coverage_b": [], "same": False,
                "wins": 0, "losses": 0, "ties": 0}
    coverage_a: dict[str, int] = {}
    coverage_b: dict[str, int] = {}
    comparisons = []
    no_common = 0
    for suite_out in comp["suites"]:
        skey = suite_out["suite"]["key"]
        for case_out in suite_out["cases"]:
            case = case_out["case"]
            in_a = bool(case_out["cells"].get(a_id))
            in_b = bool(case_out["cells"].get(b_id))
            if in_a:
                coverage_a[skey] = coverage_a.get(skey, 0) + 1
            if in_b:
                coverage_b[skey] = coverage_b.get(skey, 0) + 1
            if not (in_a and in_b):
                continue
            pw = next((p for p in case_out.get("pairwise") or []
                       if {p["a"], p["b"]} == {a_id, b_id}), None)
            if pw is None:
                no_common += 1
                continue
            score_a = (pw["case_score"] if pw["a"] == a_id else pw["case_score_b"])
            score_b = (pw["case_score_b"] if pw["a"] == a_id else pw["case_score"])
            score = score_a
            if score_a > score_b + 1e-9:
                status = "win"
            elif score_a < score_b - 1e-9:
                status = "loss"
            else:
                status = "tie"
            comparisons.append({
                "suite": skey, "case_key": case["case_key"],
                "label": f"{skey}/{case['case_key']}", "case_id": case["id"],
                "detail_a": a_id, "detail_b": b_id,
                "score": score, "score_a": score_a, "score_b": score_b,
                "status": status,
            })
    wins = sum(1 for c in comparisons if c["status"] == "win")
    losses = sum(1 for c in comparisons if c["status"] == "loss")
    ties = sum(1 for c in comparisons if c["status"] == "tie")
    return {
        "a": configs[a_id], "b": configs[b_id],
        "comparisons": comparisons, "compared_count": len(comparisons),
        "no_common": no_common, "wins": wins, "losses": losses, "ties": ties,
        "coverage_a": sorted(coverage_a.items()), "coverage_b": sorted(coverage_b.items()),
        "same": a_id == b_id,
    }


def build_config_groups(comp: dict) -> list[dict]:
    """核心参数相同的配置聚合为一组；secondary 参数差异保留为变体。"""
    ids: list[int] = []
    for suite_out in comp["suites"]:
        for cid in suite_out["config_ids"]:
            if cid not in ids:
                ids.append(cid)
    return group_configs([comp["configs"][cid] for cid in ids])


def _case_weights(comp: dict) -> dict[int, float]:
    weights: dict[int, float] = {}
    for suite_out in comp["suites"]:
        for case_out in suite_out["cases"]:
            weight = case_out["case"]["weight"]
            weights[case_out["case"]["id"]] = float(1.0 if weight is None else weight)
    return weights


def build_group_inner(comp: dict, group: dict) -> dict:
    """组内变体两两 PK，选出代表该聚合配置参与跨组对比的 Top 变体。"""
    variants = group["configs"]
    weights = _case_weights(comp)
    stats = {
        v["id"]: {"weighted_sum": 0.0, "weight_sum": 0.0,
                  "wins": 0, "losses": 0, "ties": 0, "compared": 0,
                  "coverage": sum(
                      1 for suite_out in comp["suites"]
                      for case_out in suite_out["cases"]
                      if case_out["cells"].get(v["id"]))}
        for v in variants
    }
    per_case: dict[int, dict[int, list[float]]] = {v["id"]: {} for v in variants}
    pair_pks: dict[tuple[int, int], dict] = {}
    matrix_rows = []
    for i, va in enumerate(variants):
        cells = []
        for j, vb in enumerate(variants):
            if j < i:
                cells.append({"lower": True})
            elif j == i:
                cells.append({"self": True})
            else:
                pk = build_pk_view(comp, va["id"], vb["id"])
                pair_pks[(va["id"], vb["id"])] = pk
                for item in pk["comparisons"]:
                    per_case[va["id"]].setdefault(item["case_id"], []).append(item["score_a"])
                    per_case[vb["id"]].setdefault(item["case_id"], []).append(item["score_b"])
                for vid, wins, losses in (
                        (va["id"], pk["wins"], pk["losses"]),
                        (vb["id"], pk["losses"], pk["wins"])):
                    stats[vid]["wins"] += wins
                    stats[vid]["losses"] += losses
                    stats[vid]["ties"] += pk["ties"]
                    stats[vid]["compared"] += pk["compared_count"]
                cells.append({
                    "b": vb, "compared_count": pk["compared_count"],
                    "wins": pk["wins"], "losses": pk["losses"], "ties": pk["ties"],
                    "has_pk": pk["compared_count"] > 0,
                })
        matrix_rows.append({"variant": va, "cells": cells})

    for vid, cases in per_case.items():
        for case_id, scores in cases.items():
            weight = weights.get(case_id, 1.0)
            stats[vid]["weighted_sum"] += weight * (sum(scores) / len(scores))
            stats[vid]["weight_sum"] += weight
    for item in stats.values():
        item["score"] = (item["weighted_sum"] / item["weight_sum"]
                         if item["weight_sum"] else None)

    scored = [v for v in variants if stats[v["id"]]["score"] is not None]
    pool = scored or variants
    top = min(pool, key=lambda v: (
        -(stats[v["id"]]["score"] if stats[v["id"]]["score"] is not None else 0.0),
        -stats[v["id"]]["compared"], -stats[v["id"]]["coverage"], v["name"]))
    top_stats = stats[top["id"]]

    default_pk = None
    if pair_pks:
        default_pk = max(pair_pks.values(), key=lambda pk: pk["compared_count"])
    return {
        "group": group, "variants": variants, "matrix_rows": matrix_rows,
        "stats": stats, "top": top, "top_score": top_stats["score"],
        "top_stats": top_stats, "variant_pk": default_pk,
        "variant_pk_data": [
            {"label": c["label"], "score": c["score"], "status": c["status"]}
            for c in (default_pk["comparisons"] if default_pk else [])
        ],
    }


def build_group_pk(comp: dict, ga: dict, gb: dict,
                   inner_a: dict | None = None,
                   inner_b: dict | None = None) -> dict:
    """聚合配置 A vs B：各组先按组内 PK 选出 Top 变体，再由 Top 变体 PK。"""
    inner_a = inner_a or build_group_inner(comp, ga)
    inner_b = inner_b or build_group_inner(comp, gb)
    top_a, top_b = inner_a["top"], inner_b["top"]
    pk = build_pk_view(comp, top_a["id"], top_b["id"])
    return {
        "a": {"id": ga["id"], "name": ga["label"]},
        "b": {"id": gb["id"], "name": gb["label"]},
        "top_a": top_a, "top_b": top_b,
        "comparisons": pk["comparisons"], "compared_count": pk["compared_count"],
        "wins": pk["wins"], "losses": pk["losses"], "ties": pk["ties"],
        "no_common": pk["no_common"],
        "coverage_a": pk["coverage_a"], "coverage_b": pk["coverage_b"],
        "same": False,
        "variants_a": ga["configs"], "variants_b": gb["configs"],
    }


def build_group_matrix(comp: dict, groups: list[dict],
                       inners: dict[int, dict] | None = None) -> dict:
    if not inners:
        inners = {g["id"]: build_group_inner(comp, g) for g in groups}
    rows = []
    for i, ga in enumerate(groups):
        cells = []
        for j, gb in enumerate(groups):
            if i == j:
                cells.append({"self": True})
            elif j < i:
                cells.append({"lower": True})
            else:
                pk = build_group_pk(comp, ga, gb,
                                    inners.get(ga["id"]), inners.get(gb["id"]))
                cells.append({
                    "group_id": gb["id"], "compared_count": pk["compared_count"],
                    "wins": pk["wins"], "losses": pk["losses"], "ties": pk["ties"],
                    "has_pk": pk["compared_count"] > 0,
                    "top_a": pk["top_a"]["name"], "top_b": pk["top_b"]["name"],
                })
        rows.append({"group": ga, "cells": cells})
    return {
        "groups": groups, "rows": rows,
        "top_names": {
            g["id"]: (inners.get(g["id"]) or build_group_inner(comp, g))["top"]["name"]
            for g in groups
        },
    }


def build_group_variant_rows(comp: dict, ga: dict, gb: dict) -> list[dict]:
    """跨组详情里的变体差异矩阵（仅参考，不参与跨组判定）。"""
    rows = []
    for va in ga["configs"]:
        cells = []
        for vb in gb["configs"]:
            pk = build_pk_view(comp, va["id"], vb["id"])
            cells.append({
                "b": vb, "compared_count": pk["compared_count"],
                "wins": pk["wins"], "losses": pk["losses"], "ties": pk["ties"],
                "has_pk": pk["compared_count"] > 0,
            })
        rows.append({"variant": va, "cells": cells})
    return rows


def build_pk_matrix(comp: dict) -> dict:
    """场景内全部配置两两 PK 概览（自动，按实际 pairwise case_score 聚合）。"""
    ids: list[int] = []
    for suite_out in comp["suites"]:
        for cid in suite_out["config_ids"]:
            if cid not in ids:
                ids.append(cid)
    configs = [comp["configs"][cid] for cid in ids]
    rows = []
    for a in ids:
        cells = []
        for b in ids:
            if a == b:
                cells.append({"self": True})
                continue
            pk = build_pk_view(comp, a, b)
            cells.append({
                "config_id": b, "compared_count": pk["compared_count"],
                "wins": pk["wins"], "losses": pk["losses"], "ties": pk["ties"],
                "has_pk": pk["compared_count"] > 0,
            })
        rows.append({"config": comp["configs"][a], "cells": cells})
    return {"configs": configs, "rows": rows}


def chart_payload(comp: dict, suite_out: dict) -> dict:
    """ECharts 数据：case → metric → 配置 → 值（best 与节点均值）。"""
    out = {}
    configs = comp["configs"]
    for case_out in suite_out["cases"]:
        case = case_out["case"]
        if not case_out["configs_with_data"]:
            continue
        metrics = {}
        for metric in case_out["metric_keys"]:
            series = []
            for cid in case_out["configs_with_data"]:
                cell = case_out["cells"].get(cid, {}).get(metric)
                if not cell:
                    continue
                node_reps = list(cell["same"].values())
                adjusted = [r["adjusted"] for r in node_reps
                            if r.get("adjusted") is not None]
                series.append({
                    "config_id": cid,
                    "name": configs[cid]["name"],
                    "best": cell["best"]["value"],
                    "node_mean": sum(r["value"] for r in node_reps) / len(node_reps),
                    "adjusted_mean": (sum(adjusted) / len(adjusted)) if adjusted else None,
                })
            metrics[metric] = series
        out[case["case_key"]] = {"case_id": case["id"], "metrics": metrics}
    return out


def _as_config_view(cfg: dict) -> dict:
    out = dict(cfg)
    if isinstance(out.get("params"), str):
        out["params"] = json.loads(out["params"])
    if isinstance(out.get("deploy"), str):
        out["deploy"] = json.loads(out["deploy"] or "{}")
    out.setdefault("deploy", {})
    return out


def best_config_view(scenario: dict, comp: dict, selected: dict | None,
                     recommended: dict | None) -> dict | None:
    """场景页展示的最佳配置：人工确认/改判优先，其次自动推荐，再次唯一配置。"""
    if (scenario.get("best_config")
            and scenario["best_config"].get("id") in comp["configs"]):
        status = scenario.get("best_status") or "confirmed"
        label = {"confirmed": "人工确认", "overridden": "人工改判",
                 "recommended": "推荐"}.get(status, status)
        return {"config": _as_config_view(scenario["best_config"]),
                "origin": status, "badge": status, "badge_text": label}
    if recommended:
        return {"config": _as_config_view(recommended), "origin": "recommended",
                "badge": "recommended", "badge_text": "自动推荐 · 未人工确认"}
    if selected and len(selected["config_ids"]) == 1:
        cfg = comp["configs"][selected["config_ids"][0]]
        return {"config": _as_config_view(cfg), "origin": "single",
                "badge": "recommended", "badge_text": "仅一套配置 · 待补充对比"}
    return None


def build_best_basis(comp: dict, focus_id: int) -> list[dict]:
    """为什么是最佳：focus 配置 vs 每个对手 × case × metric 的评分依据。"""
    w_same = float(comp["scoring"]["w_same"])
    w_adj = float(comp["scoring"]["w_adj"])
    w_best = float(comp["scoring"]["w_best"])
    out = []
    for suite_out in comp["suites"]:
        focus_rank = next((r for r in suite_out["ranking"]
                           if r["config_id"] == focus_id), None)
        opponents = []
        for other_id in suite_out["config_ids"]:
            if other_id == focus_id:
                continue
            cases = []
            weight_sum = 0.0
            score_sum = 0.0
            for case_out in suite_out["cases"]:
                pw = next((p for p in case_out["pairwise"]
                           if {p["a"], p["b"]} == {focus_id, other_id}), None)
                if pw is None:
                    continue
                focus_is_a = pw["a"] == focus_id
                metrics = []
                for mkey, detail in pw["metrics"].items():
                    wr_same = detail["win_rate_same"]
                    wr_adj = detail["win_rate_adj"]
                    wr_best = detail["win_rate_best"]
                    wr_best_raw = detail["win_rate_best_raw"]
                    node_mode = detail.get("node_mode")
                    node_conf = detail.get("node_confidence") or 0.0
                    if not focus_is_a:
                        wr_same = None if wr_same is None else 1.0 - wr_same
                        wr_adj = None if wr_adj is None else 1.0 - wr_adj
                        wr_best = 1.0 - wr_best
                        wr_best_raw = 1.0 - wr_best_raw
                    if node_mode == "same":
                        win_rate_node = wr_same
                        node_weight = w_same
                    elif node_mode == "adjusted":
                        win_rate_node = wr_adj
                        node_weight = w_adj * node_conf
                    else:
                        win_rate_node = None
                        node_weight = 0.0
                    metric_score = w_best * wr_best
                    if win_rate_node is not None:
                        metric_score += node_weight * win_rate_node
                    nodes = []
                    for nr in detail["node_results"]:
                        fv, ov = (nr["a"], nr["b"]) if focus_is_a else (nr["b"], nr["a"])
                        score = nr["score"] if focus_is_a else 1.0 - nr["score"]
                        nodes.append({"node": nr["node"], "focus": fv, "other": ov,
                                      "score": score, "mark": _mark(score)})
                    adj_nodes = []
                    for nr in detail.get("node_results_adj", []):
                        fv, ov = ((nr["a_adjusted"], nr["b_adjusted"]) if focus_is_a
                                  else (nr["b_adjusted"], nr["a_adjusted"]))
                        raw_f, raw_o = ((nr["a"], nr["b"]) if focus_is_a
                                        else (nr["b"], nr["a"]))
                        score = nr["score"] if focus_is_a else 1.0 - nr["score"]
                        adj_nodes.append({
                            "node_a": nr["node_a"], "node_b": nr["node_b"],
                            "focus": fv, "other": ov,
                            "raw_focus": raw_f, "raw_other": raw_o,
                            "confidence": nr["confidence"],
                            "score": score, "mark": _mark(score),
                        })
                    metrics.append({
                        "key": mkey, "def": comp["metric_defs"][mkey],
                        "win_rate_same": wr_same, "win_rate_adj": wr_adj,
                        "win_rate_node": win_rate_node,
                        "win_rate_best": wr_best, "win_rate_best_raw": wr_best_raw,
                        "node_mode": node_mode, "node_confidence": node_conf,
                        "node_weight": node_weight,
                        "metric_score": metric_score, "weight": detail["weight"],
                        "nodes": nodes, "adj_nodes": adj_nodes,
                        "best_mark": _mark(wr_best),
                        "focus_best": case_out["cells"][focus_id][mkey]["best"],
                        "other_best": case_out["cells"][other_id][mkey]["best"],
                    })
                case_score = pw["case_score"] if focus_is_a else pw["case_score_b"]
                case_weight = case_out["case"]["weight"]
                weight = float(1.0 if case_weight is None else case_weight)
                weight_sum += weight
                score_sum += weight * case_score
                cases.append({"case": case_out["case"], "case_score": case_score,
                              "mark": _mark(case_score), "metrics": metrics})
            if not cases:
                continue
            opponents.append({
                "config": comp["configs"][other_id],
                "avg_score": score_sum / weight_sum if weight_sum else None,
                "cases": cases,
            })
        out.append({"suite": suite_out["suite"], "focus_rank": focus_rank,
                    "opponents": opponents, "insufficient": suite_out["insufficient"],
                    "node_effects": suite_out.get("node_effects", {})})
    return out


@router.get("/")
def page_index(request: Request):
    with get_db() as conn:
        models = model_summaries(conn)
    return templates.TemplateResponse(request, "index.html", {"models": models})


def best_data_view(comp: dict, cfg_id: int) -> list[dict]:
    """最佳配置的当前最佳数据：逐 suite/case/metric 展示。

    case/指标缺失时回退到排名靠前的其他配置，并标明数据来源；不要求用例全覆盖。
    """
    out = []
    configs = comp["configs"]
    for suite_out in comp["suites"]:
        ranking = sorted(suite_out.get("ranking") or [], key=lambda r: r["rank"])
        order = [r["config_id"] for r in ranking]
        rank_by_id = {r["config_id"]: r["rank"] for r in ranking}
        has_focus = any(case_out["cells"].get(cfg_id) for case_out in suite_out["cases"])
        focus_id = cfg_id if has_focus else (
            suite_out.get("recommended_config_id") or (order[0] if order else cfg_id))
        cases = []
        for case_out in suite_out["cases"]:
            cells = case_out["cells"]
            own = cells.get(focus_id) or {}
            metrics = []
            for key in case_out["metric_keys"]:
                cell = own.get(key)
                source_id, fallback = focus_id, False
                if not cell:
                    for cid in order:
                        if cid == focus_id:
                            continue
                        candidate = cells.get(cid, {}).get(key)
                        if candidate:
                            cell = candidate
                            source_id = cid
                            fallback = True
                            break
                if not cell:
                    metrics.append({"key": key, "def": comp["metric_defs"][key],
                                    "missing": True})
                    continue
                metrics.append({
                    "key": key, "def": comp["metric_defs"][key],
                    "best": cell["best"], "same": cell["same"],
                    "missing": False, "fallback": fallback,
                    "source_config_id": source_id,
                    "source_config_name": configs[source_id]["name"],
                    "source_rank": rank_by_id.get(source_id),
                })
            sources = []
            seen_sources = set()
            source_config_ids: list[int] = []
            repro = case_out.get("repro") or {}
            seen_repro = set()
            for metric in metrics:
                if metric.get("missing"):
                    continue
                source_id = metric.get("source_config_id")
                if source_id is not None and source_id not in source_config_ids:
                    source_config_ids.append(source_id)
                metric["repro"] = repro.get(source_id)
                if metric["repro"] and source_id not in seen_repro:
                    seen_repro.add(source_id)
                    metric["repro_show"] = True
                if not metric.get("fallback"):
                    continue
                key = (metric.get("source_config_id"), metric.get("source_config_name"),
                       metric.get("source_rank"))
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append({"id": key[0], "name": key[1], "rank": key[2]})
            cases.append({"case": case_out["case"], "metrics": metrics,
                          "metric_map": {m["key"]: m for m in metrics},
                          "fallback_sources": sources,
                          "source_configs": [
                              {"id": cid, "name": configs[cid]["name"]}
                              for cid in source_config_ids if cid in configs
                          ],
                          "missing": not any(not m.get("missing") for m in metrics)})
        keys: list[str] = []
        for case_view in cases:
            for metric in case_view["metrics"]:
                if metric["key"] not in keys:
                    keys.append(metric["key"])
        metric_keys = [{"key": key, "def": comp["metric_defs"][key]}
                       for key in comp["metric_defs"] if key in keys]
        out.append({
            "suite": suite_out["suite"], "cases": cases, "metric_keys": metric_keys,
            "focus_config": configs.get(focus_id),
            "focus_rank": rank_by_id.get(focus_id),
            "focus_is_global": focus_id == cfg_id,
        })
    return out


_DOCKER_BOOL_FLAGS = {
    "privileged", "rm", "d", "i", "t", "it", "itd", "tty", "interactive", "init",
    "detach", "publish_all", "no_healthcheck", "read_only", "oom_kill_disable",
    "sig_proxy", "help",
}
_DOCKER_KEY_ALIASES = {
    "net": "network", "network": "network", "e": "container_env", "env": "container_env",
    "v": "mounts", "volume": "mounts", "p": "ports", "publish": "ports",
    "h": "hostname", "u": "user", "w": "workdir", "m": "memory",
    "cap_add": "cap_add", "device": "devices", "name": "container_name",
    "shm_size": "shm_size", "runtime": "runtime", "ipc": "ipc", "pid": "pid",
}
_DOCKER_KEY_ORDER = [
    "container_name", "image", "network", "runtime", "shm_size", "privileged",
    "ipc", "pid", "ports", "mounts", "container_env", "cap_add", "ulimit",
]


def _docker_canon(key: str) -> str:
    k = key.lstrip("-").lower().replace("-", "_")
    return _DOCKER_KEY_ALIASES.get(k, k)


def _image_identity(ref: str) -> str:
    """镜像 identity：去掉 registry/namespace 与 _with_ray 后缀，仅保留 name:tag。"""
    if not isinstance(ref, str) or not ref.strip():
        return ref
    return ref.strip().split("/")[-1].replace("_with_ray", "")


def _docker_args(cmd: str) -> dict:
    """解析 docker run 参数为比较用 dict（重复参数聚合为列表，末位位置参数为镜像）。

    仅用于展示/对比容器启动参数，不参与配置指纹。
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return {}
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return {}
    try:
        start = tokens.index("run") + 1
    except ValueError:
        return {}
    out: dict = {}
    positional: list[str] = []

    def _add(key: str, value):
        key = _docker_canon(key)
        if key in out:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(value)
        else:
            out[key] = value

    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            key, value = tok.split("=", 1)
            _add(key, value)
        elif tok.startswith("--"):
            key = _docker_canon(tok)
            if key in _DOCKER_BOOL_FLAGS:
                _add(key, True)
            elif i + 1 < len(tokens):
                _add(key, tokens[i + 1])
                i += 1
            else:
                _add(key, True)
        elif tok.startswith("-") and len(tok) > 1:
            if len(tok) > 2:
                i += 1
                continue
            key = _docker_canon(tok)
            if key in _DOCKER_BOOL_FLAGS or i + 1 >= len(tokens):
                _add(key, True)
            else:
                _add(key, tokens[i + 1])
                i += 1
        else:
            positional.append(tok)
        i += 1
    if positional:
        out["image"] = _image_identity(positional[-1])
    return out


def _param_map(cfg: dict) -> tuple[dict, list[str]]:
    """配置声明参数 ∪ serve 命令解析参数（声明优先），返回 (map, 键顺序)。"""
    declared = cfg["params"] or {}
    declared = normalize_params(declared) if isinstance(declared, dict) else {}
    parsed: dict = {}
    serve_cmd = cfg["deploy"].get("serve_cmd")
    if isinstance(serve_cmd, str) and serve_cmd.strip():
        try:
            parsed = parse_serve_flags(serve_cmd)
        except ValueError:
            parsed = {}
        parsed.pop("_args", None)
    merged = dict(parsed)
    merged.update(declared)
    return merged, list(declared) + [k for k in parsed if k not in declared]


def deploy_compare_view(comp: dict, config_ids: list[int], diff: bool = True) -> dict:
    """参数对比（配置声明 + serve 命令 + docker 容器参数 + 环境变量）+ 部署信息（折叠）。

    diff=True 时仅保留所选配置间取值不同的参数行。
    """
    configs = [_as_config_view(comp["configs"][cid]) for cid in config_ids
               if cid in comp["configs"]]
    groups: list[tuple[str, dict[int, dict], list[str]]] = []
    param_maps: dict[int, dict] = {}
    param_order: list[str] = []
    docker_maps: dict[int, dict] = {}
    docker_order: list[str] = []
    env_maps: dict[int, dict] = {}
    env_order: list[str] = []
    for cfg in configs:
        merged, order = _param_map(cfg)
        param_maps[cfg["id"]] = merged
        for key in order:
            if key not in param_order:
                param_order.append(key)
        docker = _docker_args(cfg["deploy"].get("docker_cmd") or "")
        docker.pop("label", None)
        if (cfg["params"] or {}).get("image"):
            docker.pop("image", None)
        docker_maps[cfg["id"]] = docker
        for key in _DOCKER_KEY_ORDER + sorted(k for k in docker if k not in _DOCKER_KEY_ORDER):
            if key in docker and key not in docker_order:
                docker_order.append(key)
        env = {k: v for k, v in (cfg["deploy"].get("env") or {}).items()
               if not str(k).upper().endswith("IFNAME")}
        env_maps[cfg["id"]] = env
        for key in env:
            if key not in env_order:
                env_order.append(key)
    groups = [("配置参数", param_maps, param_order),
              ("容器启动", docker_maps, docker_order),
              ("环境变量", env_maps, env_order)]

    def _sig(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)

    rows = []
    for source, maps, order in groups:
        for key in order:
            values = {c["id"]: maps[c["id"]].get(key) for c in configs}
            if diff and len(configs) > 1:
                if len({_sig(v) for v in values.values()}) <= 1:
                    continue
            rows.append({"key": key, "source": source, "val": values})
    return {
        "configs": configs,
        "params": rows,
        "diff": diff and len(configs) > 1,
        "has_extra_cmds": any(cfg["deploy"].get("extra_cmds") for cfg in configs),
    }


def _scenario_context(conn, scenario_id: int, suite: int | None,
                      mode: str) -> dict:
    scenario = scenario_detail(conn, scenario_id)
    if not scenario:
        raise HTTPException(404, "场景不存在")
    comp = cached_comparison(conn, scenario_id, None)
    suites = comp["suites"]
    selected = next((s for s in suites if suite is not None and s["suite"]["id"] == suite),
                    suites[0] if suites else None)
    recommended = None
    if selected and selected["recommended_config_id"]:
        recommended = comp["configs"][selected["recommended_config_id"]]
    best_view = best_config_view(scenario, comp, selected, recommended)
    scenario_config_ids: list[int] = []
    for suite_out in suites:
        for cid in suite_out["config_ids"]:
            if cid not in scenario_config_ids:
                scenario_config_ids.append(cid)
    if best_view:
        candidate_id = best_view["config"]["id"]
        groups = build_config_groups(comp)
        candidate_group = next(
            (g for g in groups if any(c["id"] == candidate_id for c in g["configs"])),
            None)
        gaps = []
        if candidate_group:
            for group in groups:
                if group is candidate_group:
                    continue
                gp = build_group_pk(comp, candidate_group, group)
                if gp["compared_count"] == 0:
                    gaps.append({
                        "id": group["id"], "name": group["label"],
                        "coverage_a": gp["coverage_a"], "coverage_b": gp["coverage_b"],
                    })
        if gaps:
            best_view = dict(best_view)
            best_view.update({
                "undetermined": True, "pk_gaps": gaps,
                "badge": "pending",
                "badge_text": f"无法判定 · 未与 {len(gaps)} 个聚合配置 PK",
            })
    return {
        "scenario": scenario, "scenario_id": scenario_id, "comp": comp,
        "suites": suites, "selected": selected, "recommended": recommended,
        "best_view": best_view,
        "table": build_table(comp, selected, mode) if selected else [],
        "chart_data": chart_payload(comp, selected) if selected else {},
        "best_data": best_data_view(comp, best_view["config"]["id"]) if best_view else [],
        "scenario_config_ids": scenario_config_ids,
        "selectable_configs": [_as_config_view(comp["configs"][cid])
                               for cid in scenario_config_ids],
    }


CONSOLE_LOG_PREVIEW_LIMIT = 2_000_000
CASE_LOG_NAMES = {"console.log": 0, "stdout.log": 1, "stderr.log": 2}


def _path_matches_case(rel_path: str, title: str | None, case_key: str) -> bool:
    text = f"{rel_path or ''}/{title or ''}".replace("\\", "/")
    parts = {part for part in re.split(r"[/\s]+", text.lower()) if part}
    return case_key.lower() in parts


def _run_case_logs(conn, run_id: int, case_keys: list[str]) -> dict[str, list[dict]]:
    """按 case 归档 run 下的 console.log/stdout.log/stderr.log，供 run 详情页展开。"""
    out: dict[str, list[dict]] = {}
    if not run_id or not case_keys:
        return out
    rows = conn.execute(
        "SELECT * FROM artifact WHERE run_id = ? AND kind = 'log' ORDER BY id",
        (run_id,)).fetchall()
    for row in rows:
        name = Path(row["title"] or row["rel_path"]).name.lower()
        if name not in CASE_LOG_NAMES:
            continue
        matched = [key for key in case_keys
                   if _path_matches_case(row["rel_path"], row["title"], key)]
        if not matched and len(case_keys) == 1:
            matched = list(case_keys)
        if not matched:
            continue
        entry = {
            "id": row["id"], "name": Path(row["title"] or row["rel_path"]).name,
            "rel_path": row["rel_path"], "title": row["title"],
            "run_id": row["run_id"],
        }
        path = artifact_path(row["rel_path"])
        if path is None or not path.is_file():
            entry["missing"] = True
        else:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                entry["missing"] = True
            else:
                entry["truncated"] = len(content) > CONSOLE_LOG_PREVIEW_LIMIT
                entry["content"] = content[:CONSOLE_LOG_PREVIEW_LIMIT]
        for key in matched:
            bucket = out.setdefault(key, [])
            if not any(x["rel_path"] == entry["rel_path"] for x in bucket):
                bucket.append(dict(entry))
    for bucket in out.values():
        bucket.sort(key=lambda x: CASE_LOG_NAMES.get(x["name"].lower(), 9))
    return out


def _artifacts_view(scenario: dict) -> tuple[list[dict], list[dict]]:
    artifacts_html = []
    for art in scenario["artifacts"]:
        path = artifact_path(art["rel_path"])
        if art["kind"] == "md" and path is not None and path.is_file():
            artifacts_html.append({
                "art": art,
                "html": render_markdown(path.read_text(encoding="utf-8"),
                                        scenario["artifacts"]),
            })
    other_artifacts = []
    for a in scenario["artifacts"]:
        if a["kind"] == "md":
            continue
        p = artifact_path(a["rel_path"])
        other_artifacts.append({"art": a, "exists": p is not None and p.is_file()})
    return artifacts_html, other_artifacts


@router.get("/models/{name:path}/scenarios/{scenario_id}/suites/{suite_id}")
def page_suite_compare(request: Request, name: str, scenario_id: int, suite_id: int,
                       mode: str = Query(default="same-node")):
    if mode not in ("same-node", "adjusted", "best"):
        mode = "same-node"
    with get_db() as conn:
        ctx = _scenario_context(conn, scenario_id, suite_id, mode)
        model = model_detail(conn, name)
    ctx["mode"] = mode
    return templates.TemplateResponse(request, "suite_compare.html", {
        "model": model, **ctx,
    })


@router.get("/models/{name:path}/scenarios/{scenario_id}")
def page_scenario(request: Request, name: str, scenario_id: int):
    with get_db() as conn:
        ctx = _scenario_context(conn, scenario_id, None, "same-node")
        model = model_detail(conn, name)
    return templates.TemplateResponse(request, "scenario.html", {
        "model": model, **ctx,
    })


@router.get("/models/{name:path}/scenarios/{scenario_id}/compare")
def page_scenario_compare(request: Request, name: str, scenario_id: int,
                          suite: int | None = Query(default=None),
                          mode: str = Query(default="same-node"),
                          configs: list[int] | None = Query(default=None),
                          diff: int = Query(default=1),
                          pk_a: int | None = Query(default=None),
                          pk_b: int | None = Query(default=None)):
    if mode not in ("same-node", "adjusted", "best"):
        mode = "same-node"
    with get_db() as conn:
        ctx = _scenario_context(conn, scenario_id, suite, mode)
        model = model_detail(conn, name)
        ctx["mode"] = mode
        all_ids = ctx["scenario_config_ids"]
        requested = [cid for cid in (configs or []) if cid in all_ids]
        selected_ids = requested or all_ids
        drift = [c for c in deploy_audit(conn, selected_ids)["configs"]
                 if any(item["evidence"] and not item["ok"] for item in c["runs"])]
    compare = deploy_compare_view(ctx["comp"], selected_ids, diff=bool(diff))
    compare["selectable"] = ctx["selectable_configs"]
    compare["selected_ids"] = selected_ids
    compare["drift"] = drift
    ctx["compare"] = compare

    groups = build_config_groups(ctx["comp"])
    inners = {g["id"]: build_group_inner(ctx["comp"], g) for g in groups}
    ctx["groups"] = groups
    ctx["group_tops"] = [inners[g["id"]] for g in groups]
    ctx["has_group_variants"] = any(len(g["configs"]) > 1 for g in groups)
    ctx["group_matrix"] = build_group_matrix(ctx["comp"], groups, inners)
    ctx["single_group_inner"] = (
        ctx["group_tops"][0] if len(groups) == 1 else None)
    best = None
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            gp = build_group_pk(ctx["comp"], ga, gb,
                                inners[ga["id"]], inners[gb["id"]])
            if best is None or gp["compared_count"] > best["compared_count"]:
                best = gp
    ctx["group_pk"] = best
    ctx["group_pk_data"] = [
        {"label": c["label"], "score": c["score"], "status": c["status"]}
        for c in (best["comparisons"] if best else [])
    ]
    return templates.TemplateResponse(request, "compare.html", {
        "model": model, **ctx,
    })


@router.get("/models/{name:path}")
def page_model(request: Request, name: str):
    with get_db() as conn:
        model = model_detail(conn, name)
    if not model:
        raise HTTPException(404, "模型不存在")
    return templates.TemplateResponse(request, "model.html", {"model": model})


@router.get("/partials/scenarios/{scenario_id}/comparison")
def partial_comparison(request: Request, scenario_id: int,
                       suite: int | None = Query(default=None),
                       mode: str = Query(default="same-node")):
    if mode not in ("same-node", "adjusted", "best"):
        mode = "same-node"
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
    suites = comp["suites"]
    selected = next((s for s in suites if suite is not None and s["suite"]["id"] == suite),
                    suites[0] if suites else None)
    table = build_table(comp, selected, mode) if selected else []
    return templates.TemplateResponse(request, "partials/comparison.html", {
        "comp": comp, "selected": selected, "mode": mode, "table": table,
        "scenario_id": scenario_id,
        "chart_data": chart_payload(comp, selected) if selected else {},
    })


@router.get("/partials/scenarios/{scenario_id}/pk")
def partial_pk(request: Request, scenario_id: int,
               pk_a: int = Query(...), pk_b: int = Query(...)):
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
    pk = build_pk_view(comp, pk_a, pk_b)
    pk_data = [
        {"label": c["label"], "score": c["score"], "status": c["status"]}
        for c in (pk["comparisons"] if pk else [])
    ]
    return templates.TemplateResponse(request, "partials/pk_detail.html", {
        "pk": pk, "pk_data": pk_data,
    })


@router.get("/partials/scenarios/{scenario_id}/pk_case")
def partial_pk_case(request: Request, scenario_id: int, case: int = Query(...),
                    pk_a: int = Query(...), pk_b: int = Query(...)):
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
        case_row = conn.execute("SELECT * FROM test_case WHERE id = ?", (case,)).fetchone()
    case_out = next((c for s in comp["suites"] for c in s["cases"]
                     if c["case"]["id"] == case), None)
    metrics = []
    if case_out:
        cell_a = case_out["cells"].get(pk_a, {})
        cell_b = case_out["cells"].get(pk_b, {})
        pw = next((p for p in case_out.get("pairwise") or []
                   if {p["a"], p["b"]} == {pk_a, pk_b}), None)
        for key in case_out["metric_keys"]:
            ca, cb = cell_a.get(key), cell_b.get(key)
            def rep(cell):
                if not cell:
                    return None, None
                means = [r["value"] for r in cell["same"].values()]
                return cell["best"]["value"], (sum(means) / len(means) if means else None)

            a_best, a_mean = rep(ca)
            b_best, b_mean = rep(cb)
            detail = (pw or {}).get("metrics", {}).get(key)
            score = None
            if detail:
                score = (detail.get("metric_score")
                         if pw["a"] == pk_a else detail.get("metric_score_b"))
            metrics.append({
                "key": key, "def": comp["metric_defs"][key],
                "a_best": a_best, "a_mean": a_mean,
                "b_best": b_best, "b_mean": b_mean, "score": score,
            })
    case_params = {}
    if case_row and case_row["params"]:
        try:
            case_params = json.loads(case_row["params"])
        except (TypeError, json.JSONDecodeError):
            case_params = {}
    repro = (case_out.get("repro") or {}) if case_out else {}
    return templates.TemplateResponse(request, "partials/pk_case_detail.html", {
        "case": dict(case_row) if case_row else None,
        "case_params": case_params,
        "a": comp["configs"].get(pk_a), "b": comp["configs"].get(pk_b),
        "repro_a": repro.get(pk_a), "repro_b": repro.get(pk_b),
        "metrics": metrics, "scenario_id": scenario_id,
    })


@router.get("/partials/scenarios/{scenario_id}/pk_variant")
def partial_pk_variant(request: Request, scenario_id: int,
                       pk_a: int = Query(...), pk_b: int = Query(...)):
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
    pk = build_pk_view(comp, pk_a, pk_b)
    pk_data = [
        {"label": c["label"], "score": c["score"], "status": c["status"]}
        for c in (pk["comparisons"] if pk else [])
    ]
    return templates.TemplateResponse(request, "partials/pk_detail_body.html", {
        "pk": pk, "pk_data": pk_data,
    })


@router.get("/partials/scenarios/{scenario_id}/pk_group_inner")
def partial_pk_group_inner(request: Request, scenario_id: int, g: int = Query(...)):
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
    groups = build_config_groups(comp)
    group = next((grp for grp in groups if grp["id"] == g), None)
    if not group:
        return templates.TemplateResponse(request, "partials/pk_group_inner.html", {
            "group": None, "matrix_rows": [], "variant_pk": None,
            "variant_pk_data": [], "scenario_id": scenario_id,
        })
    inner = build_group_inner(comp, group)
    return templates.TemplateResponse(request, "partials/pk_group_inner.html", {
        "group": group, "matrix_rows": inner["matrix_rows"],
        "top": inner["top"], "top_score": inner["top_score"],
        "top_stats": inner["top_stats"],
        "variant_pk": inner["variant_pk"],
        "variant_pk_data": inner["variant_pk_data"], "scenario_id": scenario_id,
    })


@router.get("/partials/scenarios/{scenario_id}/pk_group")
def partial_pk_group(request: Request, scenario_id: int,
                     ga: int = Query(...), gb: int = Query(...)):
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
    groups = build_config_groups(comp)
    group_a = next((g for g in groups if g["id"] == ga), None)
    group_b = next((g for g in groups if g["id"] == gb), None)
    if not group_a or not group_b:
        return templates.TemplateResponse(request, "partials/pk_group_detail.html", {
            "pk": None, "variant_rows": [], "variant_pk": None,
            "variant_pk_data": [], "scenario_id": scenario_id,
        })
    gp = build_group_pk(comp, group_a, group_b)
    variant_rows = build_group_variant_rows(comp, group_a, group_b)
    pk_data = [
        {"label": c["label"], "score": c["score"], "status": c["status"]}
        for c in gp["comparisons"]
    ]
    return templates.TemplateResponse(request, "partials/pk_group_detail.html", {
        "pk": gp, "pk_data": pk_data, "variant_rows": variant_rows,
        "scenario_id": scenario_id,
    })


@router.get("/partials/scenarios/{scenario_id}/drilldown")
def partial_drilldown(request: Request, scenario_id: int,
                      case: int = Query(...), metric: str = Query(...),
                      config: int = Query(...), mode: str = Query(default="same-node")):
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
        scope_ids = effective_scenario_ids(conn, scenario_id)
        scope_ph = ",".join("?" * len(scope_ids))
        case_row = conn.execute("SELECT * FROM test_case WHERE id = ?", (case,)).fetchone()
        cfg_row = conn.execute("SELECT * FROM config_set WHERE id = ?", (config,)).fetchone()
        runs = conn.execute(
            "SELECT r.*, n.label AS node_label, cr.metrics, cr.n_measured, cr.err_count"
            " FROM run r JOIN node n ON n.id = r.node_id"
            " JOIN case_result cr ON cr.run_id = r.id"
            " WHERE r.config_id = ? AND cr.case_id = ?"
            f" AND r.scenario_id IN ({scope_ph}) ORDER BY r.id",
            (config, case, *scope_ids)).fetchall()
        model_row = conn.execute(
            "SELECT name FROM model WHERE id = ?",
            (cfg_row["model_id"],)).fetchone() if cfg_row else None
    model_name = model_row["name"] if model_row else ""
    run_items = []
    for r in runs:
        metrics = json.loads(r["metrics"])
        manifest = json.loads(r["manifest"] or "{}")
        run_items.append({
            "id": r["id"], "node": r["node_label"], "value": metrics.get(metric),
            "n_measured": r["n_measured"], "valid": bool(r["valid"]),
            "excluded": bool(r["excluded"]), "started_at": r["started_at"],
            "worker_label": r["worker_label"],
            "repro_cmd": resolve_repro(
                model_name, dict(cfg_row) if cfg_row else {},
                {"manifest": manifest, "backend": r["backend"],
                 "node_label": r["node_label"]},
                dict(case_row) if case_row else {}),
        })
    cell = None
    for s in comp["suites"]:
        for c in s["cases"]:
            if c["case"]["id"] == case:
                cell = c["cells"].get(config, {}).get(metric)
    return templates.TemplateResponse(request, "partials/drilldown.html", {
        "case": dict(case_row) if case_row else None,
        "config": dict(cfg_row) if cfg_row else None,
        "metric": metric, "metric_def": comp["metric_defs"].get(metric),
        "mode": mode, "cell": cell, "runs": run_items,
    })


@router.get("/partials/scenarios/{scenario_id}/case_runs")
def partial_case_runs(request: Request, scenario_id: int, case: int = Query(...),
                      config: list[int] | None = Query(default=None)):
    """某个 case 的分节点执行数据；同一节点多条记录可展开查看。"""
    with get_db() as conn:
        comp = cached_comparison(conn, scenario_id, None)
        scope_ids = effective_scenario_ids(conn, scenario_id)
        scope_ph = ",".join("?" * len(scope_ids))
        case_row = conn.execute("SELECT * FROM test_case WHERE id = ?", (case,)).fetchone()
        sql = ("SELECT r.*, n.label AS node_label, cs.name AS config_name,"
               " cr.metrics, cr.n_measured, cr.err_count"
               " FROM run r JOIN node n ON n.id = r.node_id"
               " JOIN case_result cr ON cr.run_id = r.id"
               " JOIN config_set cs ON cs.id = r.config_id"
               f" WHERE cr.case_id = ? AND r.scenario_id IN ({scope_ph})")
        params: list = [case, *scope_ids]
        if config:
            sql += f" AND r.config_id IN ({','.join('?' * len(config))})"
            params.extend(config)
        runs = conn.execute(sql + " ORDER BY r.config_id, n.label, r.id", params).fetchall()

    metric_defs = comp["metric_defs"]
    run_items = []
    for r in runs:
        try:
            metrics = json.loads(r["metrics"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metrics = {}
        run_items.append({
            "id": r["id"], "config_id": r["config_id"], "config_name": r["config_name"],
            "node": r["node_label"], "metrics": metrics,
            "n_measured": r["n_measured"], "err_count": r["err_count"],
            "valid": bool(r["valid"]), "excluded": bool(r["excluded"]),
            "started_at": r["started_at"], "worker_label": r["worker_label"],
        })

    metric_keys: list[str] = []
    for item in run_items:
        for key in item["metrics"]:
            if key in metric_defs and key not in metric_keys:
                metric_keys.append(key)
    metric_keys = [k for k in metric_defs if k in metric_keys]

    config_order: list[int] = []
    if config:
        for cid in config:
            if cid not in config_order:
                config_order.append(cid)
    for item in run_items:
        if item["config_id"] not in config_order:
            config_order.append(item["config_id"])

    by_config: dict[int, list[dict]] = {}
    for item in run_items:
        by_config.setdefault(item["config_id"], []).append(item)

    configs_view = []
    for cid in config_order:
        items = by_config.get(cid)
        if not items:
            continue
        by_node: dict[str, list[dict]] = {}
        for item in items:
            by_node.setdefault(item["node"], []).append(item)
        nodes = []
        for label in sorted(by_node):
            records = sorted(by_node[label], key=lambda r: r["id"])
            valid_records = [r for r in records if r["valid"] and not r["excluded"]]
            means = {}
            for key in metric_keys:
                values = [r["metrics"].get(key) for r in valid_records
                          if r["metrics"].get(key) is not None]
                means[key] = (sum(values) / len(values)) if values else None
            nodes.append({
                "label": label, "records": records,
                "valid_count": len(valid_records), "means": means,
            })
        cfg = comp["configs"].get(cid) or {"id": cid, "name": items[0]["config_name"]}
        configs_view.append({"config": cfg, "nodes": nodes})

    return templates.TemplateResponse(request, "partials/case_runs.html", {
        "case": dict(case_row) if case_row else None,
        "configs": configs_view, "metric_keys": metric_keys,
        "metric_defs": metric_defs,
    })


@router.get("/runs/{run_id}")
def page_run(request: Request, run_id: int):
    with get_db() as conn:
        run = run_detail(conn, run_id)
        if not run:
            raise HTTPException(404, "run 不存在")
        metric_defs = {r["key"]: dict(r) for r in
                       conn.execute("SELECT * FROM metric_def").fetchall()}
        logs_by_case = _run_case_logs(
            conn, run_id, [cr["case_key"] for cr in run["case_results"]])
        for cr in run["case_results"]:
            cr["logs"] = logs_by_case.get(cr["case_key"], [])
    artifacts_render = []
    for art in run["artifacts"]:
        path = artifact_path(art["rel_path"])
        exists = path is not None and path.is_file()
        item = {"art": art, "exists": exists}
        if art["kind"] == "md" and exists:
            item["html"] = render_markdown(path.read_text(encoding="utf-8"), run["artifacts"])
        artifacts_render.append(item)
    manifest_pretty = json.dumps(run["manifest"], ensure_ascii=False, indent=2)
    repro_items = []
    model_name = (run.get("model") or {}).get("name") or ""
    suite_key = (run.get("suite") or {}).get("key") or ""
    for cr in run["case_results"]:
        repro_items.append({
            "case_key": cr["case_key"],
            "cmd": resolve_repro(model_name, run.get("config") or {}, run, cr, suite_key),
        })
    return templates.TemplateResponse(request, "run.html", {
        "run": run, "metric_defs": metric_defs, "artifacts_render": artifacts_render,
        "manifest_pretty": manifest_pretty, "repro_items": repro_items,
        "preview_limit_kb": CONSOLE_LOG_PREVIEW_LIMIT // 1000,
    })


@router.get("/catalog")
def page_catalog(request: Request):
    with get_db() as conn:
        models = [dict(r) for r in conn.execute("SELECT * FROM model ORDER BY name").fetchall()]
        for m in models:
            m["aliases"] = json.loads(m["aliases"])
        suites = [dict(r) for r in conn.execute("SELECT * FROM suite ORDER BY key").fetchall()]
        for s in suites:
            s["cases"] = [dict(r) for r in conn.execute(
                "SELECT * FROM test_case WHERE suite_id = ? ORDER BY sort_order, id",
                (s["id"],)).fetchall()]
            for c in s["cases"]:
                c["params"] = json.loads(c["params"])
        scenarios = [dict(r) for r in conn.execute(
            "SELECT * FROM scenario ORDER BY model_id, id").fetchall()]
        for s in scenarios:
            s["suite_keys"] = [r["key"] for r in conn.execute(
                "SELECT su.key FROM suite su JOIN scenario_suite ss ON ss.suite_id = su.id"
                " WHERE ss.scenario_id = ?", (s["id"],)).fetchall()]
            s["scoring"] = json.loads(s["scoring"] or "{}")
            s["source_scenario_ids"] = parse_source_scenario_ids(
                s.get("source_scenario_ids"))
            s["config_ids"] = parse_config_ids(s.get("config_ids"))
        configs = [dict(r) for r in conn.execute(
            "SELECT * FROM config_set ORDER BY model_id, id").fetchall()]
        model_names = {m["id"]: m["name"] for m in models}
        for c in configs:
            c["params"] = json.loads(c["params"])
            c["deploy"] = json.loads(c.get("deploy") or "{}")
            c["model_name"] = model_names.get(c["model_id"])
        metric_defs = [dict(r) for r in conn.execute(
            "SELECT * FROM metric_def ORDER BY rowid").fetchall()]
        nodes = [dict(r) for r in conn.execute("SELECT * FROM node ORDER BY label").fetchall()]
    return templates.TemplateResponse(request, "catalog.html", {
        "models": models, "suites": suites, "scenarios": scenarios, "configs": configs,
        "metric_defs": metric_defs, "nodes": nodes,
    })