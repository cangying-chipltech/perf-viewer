"""聚合评分引擎：配置优先的横向对比 + 节点性能校正。

代表值口径：
- 同节点比较 = 该节点×配置×case 全部有效运行均值。
- 最优值比较 = 该配置跨全部节点/运行历史最好值（延迟 min、吞吐 max）。
- 失败/不完整运行（valid=0）与人工剔除（excluded=1）不参与。

节点性能校正：
- 用「同一配置 × 同一 case 在不同节点」的运行估计节点倍率（对数空间最小二乘），
  节点差异可以来自场景内其他 case，不要求每套配置跑满全部节点。
- 当两套配置没有共同节点时，按节点倍率把双方代表值折算到同一基准，再做胜负比较；
  校正结果按节点差异的证据量（配对 case 数）打折，证据不足时退化为最优值比较。

两两比较：相对差 ≤ε 记平局 0.5，否则胜 1/负 0。
metric_score = w_same × 同节点胜率
             + w_adj × 校正置信度 × 跨节点校正胜率
             + w_best × 最优值胜率；
无共同节点且有校正证据时使用 w_adj 项，否则仅计 w_best 项。
case_score = Σ(metric_score*metric_weight)/Σmetric_weight（仅双方有数据的 metric）。
配置总分 = 对各对手×case 的 case_score 按 case weight 加权平均（仅计有数据且有对手的 case）。
"""

import heapq
import json
import math
from collections import defaultdict
from itertools import combinations

from . import cache
from .config import DEFAULT_SCORING
from .repro import resolve_repro
from .scope import effective_scenario_ids, parse_config_ids


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _compare(a: float, b: float, direction: str, epsilon: float) -> float:
    """返回 a 相对 b 的得分：1 胜 / 0.5 平 / 0 负。"""
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.5
    if abs(a - b) / denom <= epsilon:
        return 0.5
    a_wins = a < b if direction == "lower" else a > b
    return 1.0 if a_wins else 0.0


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pair_key_text(a: str, b: str) -> str:
    left, right = _pair_key(a, b)
    return f"{left}|{right}"


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """高斯消元（小矩阵，节点数量级）。奇异行返回 0。"""
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            continue
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
        divisor = a[col][col]
        for j in range(col, n):
            a[col][j] /= divisor
        b[col] /= divisor
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor == 0:
                continue
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]
    return b


def _edge_offsets(nodes: list[str], edges: dict) -> tuple[dict, set]:
    """按连通分量求 log 节点倍率 t_node（每个分量参考节点 t=0）。

    edges: {(a, b): (log(y_a/y_b), weight)}，a < b。
    返回 (offsets, known)：known 为有跨节点配对证据的节点。
    """
    adjacency: dict = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    offsets: dict = {}
    known: set = set()
    seen: set = set()
    for root in sorted(nodes):
        if root in seen:
            continue
        component: list[str] = []
        stack = [root]
        seen.add(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(component) == 1:
            offsets[component[0]] = 0.0
            continue
        component_set = set(component)
        known.update(component_set)

        weight_by_node: dict = defaultdict(float)
        for (a, b), (_, weight) in edges.items():
            if a in component_set:
                weight_by_node[a] += weight
                weight_by_node[b] += weight
        reference = max(component, key=lambda n: weight_by_node[n])
        unknown = [n for n in component if n != reference]
        index = {n: i for i, n in enumerate(unknown)}
        size = len(unknown)
        matrix = [[0.0] * size for _ in range(size)]
        rhs = [0.0] * size
        for (a, b), (diff, weight) in edges.items():
            if a not in component_set:
                continue
            terms = []
            if a != reference:
                terms.append((index[a], 1.0))
            if b != reference:
                terms.append((index[b], -1.0))
            for i, ci in terms:
                rhs[i] += weight * ci * diff
                for j, cj in terms:
                    matrix[i][j] += weight * ci * cj
        for i in range(size):
            matrix[i][i] += 1e-9
        solution = _solve_linear(matrix, rhs)
        offsets[reference] = 0.0
        for node, i in index.items():
            offsets[node] = solution[i]
    return offsets, known


def _widest_evidence(start: str, target: str, counts: dict) -> tuple[float, int]:
    """最大瓶颈路径：返回 (路径上最小配对数量, 跳数)；不连通返回 (0, 0)。"""
    if start == target:
        return 0.0, 0
    adjacency: dict = defaultdict(list)
    for (a, b), count in counts.items():
        adjacency[a].append((b, count))
        adjacency[b].append((a, count))
    best = {start: float("inf")}
    hops = {start: 0}
    queue = [(-float("inf"), start)]
    while queue:
        neg_bottleneck, node = heapq.heappop(queue)
        bottleneck = -neg_bottleneck
        if bottleneck < best.get(node, 0.0):
            continue
        if node == target:
            return bottleneck, hops[node]
        for neighbor, count in adjacency[node]:
            candidate = min(bottleneck, count)
            if candidate > best.get(neighbor, 0.0):
                best[neighbor] = candidate
                hops[neighbor] = hops[node] + 1
                heapq.heappush(queue, (-candidate, neighbor))
    return 0.0, 0


def _estimate_node_effects(data: dict, metric_defs: dict, evidence_full: float) -> dict:
    """从同配置同 case 的跨节点运行估计各 metric 的节点倍率与置信度。"""
    groups: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_nodes: set = set()
    for config_id, by_case in data.items():
        for case_id, by_metric in by_case.items():
            for metric, entries in by_metric.items():
                if metric not in metric_defs:
                    continue
                for node_label, value, _ in entries:
                    groups[metric][(config_id, case_id)][node_label].append(value)
                    all_nodes.add(node_label)

    effects: dict = {}
    full = max(float(evidence_full or 1.0), 1e-9)
    for metric, by_group in groups.items():
        edge_values: dict = defaultdict(list)
        for (_, _), node_values in by_group.items():
            means = {node: _mean(values) for node, values in node_values.items() if values}
            labels = sorted(means)
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    va, vb = means[a], means[b]
                    if va > 0 and vb > 0:
                        edge_values[(a, b)].append(math.log(va / vb))
        edges = {pair: (_median(values), float(len(values)))
                 for pair, values in edge_values.items()}
        counts = {pair: len(values) for pair, values in edge_values.items()}
        nodes = sorted(all_nodes | {n for pair in edges for n in pair})
        offsets, known = _edge_offsets(nodes, edges)

        pair_confidence: dict = {}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                direct = counts.get((a, b))
                if direct:
                    confidence = min(1.0, direct / full)
                else:
                    bottleneck, hops = _widest_evidence(a, b, counts)
                    confidence = min(1.0, bottleneck / full) if bottleneck > 0 else 0.0
                    if hops > 1:
                        confidence *= 0.5 ** (hops - 1)
                pair_confidence[_pair_key_text(a, b)] = confidence

        effects[metric] = {
            "offset": offsets,
            "factor": {node: math.exp(offset) for node, offset in offsets.items()},
            "known": sorted(known),
            "nodes": nodes,
            "pair_counts": {_pair_key_text(a, b): count
                            for (a, b), count in counts.items()},
            "pair_confidence": pair_confidence,
        }
    return effects


def _pair_confidence(effect: dict, a: str, b: str) -> float:
    if not effect:
        return 0.0
    return float(effect.get("pair_confidence", {}).get(_pair_key_text(a, b), 0.0))


def _adjusted_value(value: float, node: str, effect: dict) -> float | None:
    if not effect or node not in effect.get("known", ()):
        return None
    offset = effect.get("offset", {}).get(node, 0.0)
    return value * math.exp(-offset)


def scenario_scoring(row: dict) -> dict:
    scoring = dict(DEFAULT_SCORING)
    try:
        stored = json.loads(row.get("scoring") or "{}")
    except (TypeError, json.JSONDecodeError):
        stored = {}
    scoring.update({k: v for k, v in stored.items() if v is not None})
    scoring["metric_weights"] = dict(stored.get("metric_weights") or {})
    return scoring


def load_comparison(conn, scenario_id: int, suite_id: int | None = None) -> dict:
    scenario = conn.execute("SELECT * FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
    if not scenario:
        raise KeyError(f"scenario {scenario_id} 不存在")
    scenario = dict(scenario)
    scoring = scenario_scoring(scenario)
    model = dict(conn.execute("SELECT * FROM model WHERE id = ?",
                              (scenario["model_id"],)).fetchone())
    epsilon = float(scoring["epsilon"])
    w_same = float(scoring["w_same"])
    w_adj = float(scoring["w_adj"])
    w_best = float(scoring["w_best"])
    evidence_full = float(scoring.get("node_evidence_full", 3.0))
    metric_weights = scoring["metric_weights"]

    metric_defs = {r["key"]: dict(r) for r in conn.execute("SELECT * FROM metric_def").fetchall()}

    scope_ids = effective_scenario_ids(conn, scenario_id)
    scope_ph = ",".join("?" * len(scope_ids))
    selected_config_ids = parse_config_ids(scenario.get("config_ids"))
    config_ph = ",".join("?" * len(selected_config_ids))
    config_clause = f" AND config_id IN ({config_ph})" if selected_config_ids else ""
    suite_sql = ("SELECT DISTINCT s.* FROM suite s"
                 " JOIN scenario_suite ss ON ss.suite_id = s.id"
                 f" WHERE ss.scenario_id IN ({scope_ph})")
    suite_params: list = [*scope_ids]
    if suite_id is not None:
        suite_sql += " AND s.id = ?"
        suite_params.append(suite_id)
    suite_rows = conn.execute(suite_sql + " ORDER BY s.id", suite_params).fetchall()

    configs = {}
    config_sql = "SELECT * FROM config_set WHERE model_id = ?"
    config_params: list = [scenario["model_id"]]
    if selected_config_ids:
        config_sql += f" AND id IN ({config_ph})"
        config_params.extend(selected_config_ids)
    for r in conn.execute(config_sql, config_params).fetchall():
        cfg = dict(r)
        cfg["deploy"] = json.loads(cfg.get("deploy") or "{}")
        configs[cfg["id"]] = cfg
    nodes = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM node").fetchall()}

    # 读取该场景及其来源场景的有效运行，再按 suite 的 case 和配置白名单过滤。
    # 自定义场景通过 source_scenario_ids 聚合多个场景的 run 做横向对比。
    run_sql = ("SELECT * FROM run WHERE model_id = ? AND valid = 1 AND excluded = 0"
               f" AND scenario_id IN ({scope_ph}){config_clause}")
    run_params: list = [scenario["model_id"], *scope_ids, *selected_config_ids]
    res_sql = ("SELECT cr.* FROM case_result cr JOIN run r ON r.id = cr.run_id"
               " WHERE r.model_id = ? AND r.valid = 1 AND r.excluded = 0"
               f" AND r.scenario_id IN ({scope_ph})"
               + (f" AND r.config_id IN ({config_ph})" if selected_config_ids else ""))
    res_params: list = [scenario["model_id"], *scope_ids, *selected_config_ids]
    runs = conn.execute(run_sql, run_params).fetchall()
    run_by_id = {r["id"]: dict(r) for r in runs}
    results = conn.execute(res_sql, res_params).fetchall()

    # data[config_id][case_id][metric] = list of (node_label, value, run_id)
    data: dict = {}
    case_rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM test_case").fetchall()}
    for res in results:
        run = run_by_id.get(res["run_id"])
        if not run:
            continue
        case = case_rows.get(res["case_id"])
        if not case:
            continue
        metrics = json.loads(res["metrics"])
        node_label = nodes[run["node_id"]]["label"]
        bucket = data.setdefault(run["config_id"], {}).setdefault(case["id"], {})
        for key, value in metrics.items():
            if value is None:
                continue
            bucket.setdefault(key, []).append((node_label, float(value), run["id"]))

    # 每个 配置×case 数据点的复现命令：取该数据点最新一次有效 run。
    repro_by_case: dict = {}
    for cid, by_case in data.items():
        for case_id, by_metric in by_case.items():
            run_ids = sorted({run_id for entries in by_metric.values()
                              for _, _, run_id in entries})
            if not run_ids:
                continue
            run_id = run_ids[-1]
            run = run_by_id.get(run_id)
            case = case_rows.get(case_id)
            if not run or not case:
                continue
            repro_by_case[(cid, case_id)] = {
                "run_id": run_id,
                "cmd": resolve_repro(
                    model["name"], configs.get(run["config_id"], {}),
                    {**run, "node_label": nodes[run["node_id"]]["label"]}, case),
            }

    # 节点倍率按 suite 独立估计：只用本 suite 的 case 数据，避免跨测试集合借用校正因子。

    def reps(entries, metric, effects):
        direction = metric_defs[metric]["direction"]
        effect = effects.get(metric, {})
        by_node: dict = {}
        for node_label, value, run_id in entries:
            by_node.setdefault(node_label, []).append((value, run_id))
        same = {}
        for node_label, pairs in by_node.items():
            values = [v for v, _ in pairs]
            adjusted = [adj for v, _ in pairs
                        if (adj := _adjusted_value(v, node_label, effect)) is not None]
            adjusted_best = None
            if adjusted:
                adjusted_best = min(adjusted) if direction == "lower" else max(adjusted)
            same[node_label] = {
                "value": _mean(values),
                "n_runs": len(pairs),
                "min": min(values),
                "max": max(values),
                "adjusted": _mean(adjusted) if adjusted else None,
                "min_adjusted": min(adjusted) if adjusted else None,
                "max_adjusted": max(adjusted) if adjusted else None,
                "best_adjusted": adjusted_best,
                "run_ids": sorted({rid for _, rid in pairs}),
            }
        all_values = [v for _, v, _ in entries]
        best_value = min(all_values) if direction == "lower" else max(all_values)
        best = {
            "value": best_value,
            "n_runs": len(entries),
            "min": min(all_values),
            "max": max(all_values),
            "run_ids": sorted({rid for _, _, rid in entries}),
        }
        return same, best

    suites_out = []
    all_nodes_seen: set = set()
    for suite_row in suite_rows:
        suite = dict(suite_row)
        cases = [dict(r) for r in conn.execute(
            "SELECT * FROM test_case WHERE suite_id = ? ORDER BY sort_order, id", (suite["id"],)
        ).fetchall()]
        suite_case_ids = {c["id"] for c in cases}
        suite_data = {
            cid: {case_id: by_metric for case_id, by_metric in by_case.items()
                  if case_id in suite_case_ids}
            for cid, by_case in data.items()
            if suite_case_ids & set(by_case)
        }
        suite_effects = _estimate_node_effects(suite_data, metric_defs, evidence_full)
        suite_config_ids = sorted(suite_data)
        for cid in suite_config_ids:
            for case_id in data[cid]:
                if case_id not in suite_case_ids:
                    continue
                for entries in data[cid][case_id].values():
                    all_nodes_seen.update(n for n, _, _ in entries)

        cases_out = []
        for case in cases:
            cid_with_data = [cid for cid in suite_config_ids
                             if case["id"] in data.get(cid, {})]
            cells: dict = {}
            metric_keys: list = []
            for cid in cid_with_data:
                cells[cid] = {}
                for metric, entries in data[cid][case["id"]].items():
                    if metric not in metric_defs:
                        continue
                    if metric not in metric_keys:
                        metric_keys.append(metric)
                    same, best = reps(entries, metric, suite_effects)
                    cells[cid][metric] = {"same": same, "best": best}
            metric_keys = [k for k in metric_defs if k in metric_keys]

            pairwise = []
            for ca, cb in combinations(cid_with_data, 2):
                metric_details = {}
                for metric in metric_keys:
                    a_cell = cells.get(ca, {}).get(metric)
                    b_cell = cells.get(cb, {}).get(metric)
                    if not a_cell or not b_cell:
                        continue
                    direction = metric_defs[metric]["direction"]
                    effect = suite_effects.get(metric, {})
                    common = sorted(set(a_cell["same"]) & set(b_cell["same"]))
                    node_results = []
                    for node_label in common:
                        av = a_cell["same"][node_label]["value"]
                        bv = b_cell["same"][node_label]["value"]
                        node_results.append({
                            "node": node_label, "a": av, "b": bv,
                            "score": _compare(av, bv, direction, epsilon),
                        })
                    wr_same = (_mean([r["score"] for r in node_results])
                               if node_results else None)
                    wr_best_raw = _compare(a_cell["best"]["value"], b_cell["best"]["value"],
                                           direction, epsilon)

                    adj_results = []
                    wr_adj = None
                    node_confidence = 0.0
                    if not node_results:
                        for na, rep_a in a_cell["same"].items():
                            for nb, rep_b in b_cell["same"].items():
                                if na == nb:
                                    continue
                                confidence = _pair_confidence(effect, na, nb)
                                if confidence <= 0:
                                    continue
                                av = rep_a.get("adjusted")
                                bv = rep_b.get("adjusted")
                                if av is None or bv is None:
                                    continue
                                adj_results.append({
                                    "node_a": na, "node_b": nb,
                                    "a": rep_a["value"], "b": rep_b["value"],
                                    "a_adjusted": av, "b_adjusted": bv,
                                    "score": _compare(av, bv, direction, epsilon),
                                    "confidence": confidence,
                                })
                        if adj_results:
                            total_conf = sum(r["confidence"] for r in adj_results)
                            wr_adj = sum(r["score"] * r["confidence"]
                                         for r in adj_results) / total_conf
                            node_confidence = total_conf / len(adj_results)

                    if node_results:
                        node_mode = "same"
                        node_weight = w_same
                        win_rate_node = wr_same
                        win_rate_best = wr_best_raw
                        win_rate_best_adj = None
                        best_mode = "raw"
                    elif adj_results:
                        node_mode = "adjusted"
                        node_weight = w_adj * node_confidence
                        win_rate_node = wr_adj
                        a_best_values = [a_cell["same"][r["node_a"]].get("best_adjusted")
                                         for r in adj_results]
                        b_best_values = [b_cell["same"][r["node_b"]].get("best_adjusted")
                                         for r in adj_results]
                        a_best_values = [v for v in a_best_values if v is not None]
                        b_best_values = [v for v in b_best_values if v is not None]
                        if a_best_values and b_best_values:
                            a_best_adj = (min(a_best_values) if direction == "lower"
                                          else max(a_best_values))
                            b_best_adj = (min(b_best_values) if direction == "lower"
                                          else max(b_best_values))
                            win_rate_best_adj = _compare(a_best_adj, b_best_adj,
                                                         direction, epsilon)
                            win_rate_best = (node_confidence * win_rate_best_adj
                                             + (1 - node_confidence) * wr_best_raw)
                            best_mode = "adjusted_blend"
                        else:
                            win_rate_best = wr_best_raw
                            win_rate_best_adj = None
                            best_mode = "raw"
                    else:
                        node_mode = None
                        node_weight = 0.0
                        win_rate_node = None
                        win_rate_best = wr_best_raw
                        win_rate_best_adj = None
                        best_mode = "raw"

                    node_score = (node_weight * win_rate_node
                                  if win_rate_node is not None else 0.0)
                    node_score_b = (node_weight * (1.0 - win_rate_node)
                                    if win_rate_node is not None else 0.0)
                    metric_score = node_score + w_best * win_rate_best
                    metric_score_b = node_score_b + w_best * (1.0 - win_rate_best)
                    metric_details[metric] = {
                        "win_rate_same": wr_same,
                        "win_rate_adj": wr_adj,
                        "win_rate_node": win_rate_node,
                        "node_mode": node_mode,
                        "node_confidence": (1.0 if node_mode == "same" else
                                            (node_confidence if node_mode == "adjusted" else 0.0)),
                        "node_weight": node_weight,
                        "win_rate_best": win_rate_best,
                        "win_rate_best_raw": wr_best_raw,
                        "win_rate_best_adj": win_rate_best_adj,
                        "best_mode": best_mode,
                        "metric_score": metric_score,
                        "metric_score_b": metric_score_b,
                        "nodes_compared": len(node_results),
                        "nodes_compared_adj": len(adj_results),
                        "node_results": node_results,
                        "node_results_adj": adj_results,
                        "weight": float(metric_weights.get(metric, 1.0)),
                    }
                if not metric_details:
                    continue
                total_weight = sum(d["weight"] for d in metric_details.values())
                if total_weight <= 0:
                    continue
                case_score = sum(d["metric_score"] * d["weight"]
                                 for d in metric_details.values()) / total_weight
                case_score_b = sum(d["metric_score_b"] * d["weight"]
                                   for d in metric_details.values()) / total_weight
                pairwise.append({"a": ca, "b": cb, "metrics": metric_details,
                                 "case_score": case_score, "case_score_b": case_score_b})
            cases_out.append({
                "case": case,
                "metric_keys": metric_keys,
                "cells": cells,
                "pairwise": pairwise,
                "configs_with_data": cid_with_data,
                "repro": {cid: repro_by_case.get((cid, case["id"]))
                          for cid in cid_with_data},
            })

        # 配置总分：对每个对手×case 的 case_score 按 case weight 加权平均
        per_config: dict = {cid: {"weighted_sum": 0.0, "weight_sum": 0.0, "cases": 0}
                            for cid in suite_config_ids}
        for case_out in cases_out:
            case_weight = case_out["case"]["weight"]
            weight = float(1.0 if case_weight is None else case_weight)
            scores_by_config: dict = {}
            for pw in case_out["pairwise"]:
                scores_by_config.setdefault(pw["a"], []).append(pw["case_score"])
                scores_by_config.setdefault(pw["b"], []).append(pw["case_score_b"])
            for cid, scores in scores_by_config.items():
                agg = per_config[cid]
                agg["weighted_sum"] += weight * _mean(scores)
                agg["weight_sum"] += weight
                agg["cases"] += 1
        ranking = []
        for cid in suite_config_ids:
            agg = per_config[cid]
            total = agg["weighted_sum"] / agg["weight_sum"] if agg["weight_sum"] else None
            cfg = configs[cid]
            n_runs = sum(1 for r in run_by_id.values()
                         if r["config_id"] == cid and r["suite_id"] == suite["id"])
            ranking.append({
                "config_id": cid, "name": cfg["name"], "total_score": total,
                "cases_compared": agg["cases"], "n_runs": n_runs,
            })
        ranking.sort(key=lambda r: (-(r["total_score"] if r["total_score"] is not None else -1),
                                    r["name"]))
        for idx, item in enumerate(ranking):
            item["rank"] = idx + 1
        comparable = [r for r in ranking if r["total_score"] is not None]
        recommended = comparable[0]["config_id"] if comparable else None

        insufficient = []
        if len(suite_config_ids) == 0:
            insufficient.append("该 suite 暂无有效运行数据")
        elif len(suite_config_ids) == 1:
            insufficient.append("仅一套配置有数据，待补充对比配置")
        elif not any(c["pairwise"] for c in cases_out):
            insufficient.append("配置间无共同 case/指标数据，无法对比")
        else:
            metrics_no_node = [
                d for c in cases_out for p in c["pairwise"]
                for d in p["metrics"].values() if d.get("node_mode") is None
            ]
            if metrics_no_node:
                insufficient.append(
                    "部分对比无共同节点且缺少节点校正证据，相关指标仅按最优值比较"
                    "（节点性能差异未校正，结论置信度较低）")
            if any(d.get("node_mode") == "adjusted"
                   for c in cases_out for p in c["pairwise"]
                   for d in p["metrics"].values()):
                insufficient.append(
                    "部分对比使用了跨节点校正：节点倍率来自同配置跨节点用例，"
                    "已按配对证据量打折，详见评分依据中的节点倍率表")

        compared_cases = [c["case"]["case_key"] for c in cases_out
                          if c.get("pairwise")]
        suites_out.append({
            "suite": suite,
            "config_ids": suite_config_ids,
            "cases": cases_out,
            "ranking": ranking,
            "recommended_config_id": recommended,
            "insufficient": insufficient,
            "node_effects": suite_effects,
            "compared_cases": compared_cases,
            "compared_case_count": len(compared_cases),
        })

    return {
        "scenario": scenario,
        "model": model,
        "scoring": scoring,
        "metric_defs": metric_defs,
        "configs": configs,
        "nodes": sorted(all_nodes_seen),
        "suites": suites_out,
    }


def cached_comparison(conn, scenario_id: int, suite_id: int | None = None) -> dict:
    key = (scenario_id, suite_id)
    hit = cache.cache_get(key)
    if hit is not None:
        return hit
    result = load_comparison(conn, scenario_id, suite_id)
    cache.cache_set(key, result)
    return result
