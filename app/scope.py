"""场景可见范围：普通场景含自身，自定义场景递归包含来源场景。"""

import json


def effective_scenario_ids(conn, scenario_id: int) -> list[int]:
    """返回参与该场景对比的 run 归属场景 id（含自身与来源场景，去重排序）。"""
    seen: set[int] = set()
    stack = [scenario_id]
    while stack:
        sid = stack.pop()
        if sid in seen:
            continue
        row = conn.execute(
            "SELECT source_scenario_ids FROM scenario WHERE id = ?", (sid,)).fetchone()
        if row is None:
            continue
        seen.add(sid)
        try:
            sources = json.loads(row["source_scenario_ids"] or "[]")
        except (TypeError, json.JSONDecodeError):
            sources = []
        for source in sources:
            try:
                source_id = int(source)
            except (TypeError, ValueError):
                continue
            if source_id not in seen:
                stack.append(source_id)
    return sorted(seen)


def parse_source_scenario_ids(raw) -> list[int]:
    try:
        values = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    out = []
    for value in values if isinstance(values, list) else []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def parse_config_ids(raw) -> list[int]:
    """解析场景的对比配置白名单；空列表表示不限。"""
    try:
        values = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    out = []
    for value in values if isinstance(values, list) else []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out
