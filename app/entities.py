"""实体登记：模型别名解析、配置指纹去重、节点/suite/case 登记。"""

import hashlib
import json
import re
import shlex
from typing import Any

from .normalize import METRIC_SEED
from .security import redact, redact_cmd

_FLAG_ALIASES = {
    "tp": "tensor_parallel_size",
    "pp": "pipeline_parallel_size",
    "dcp": "decode_context_parallel_size",
}

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_FLAGS_RE = re.compile(r"(^|\s)-{1,2}[A-Za-z]")


def canonical_key(key: str) -> str:
    k = key.strip().lstrip("-").lower().replace("-", "_")
    return _FLAG_ALIASES.get(k, k)


def _scalar(value: str):
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        f = float(value)
        return int(f) if f.is_integer() else f
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def parse_serve_flags(text: str) -> dict[str, Any]:
    """把 serve 命令行参数解析为与顺序无关的规范 dict。"""
    tokens = shlex.split(text)
    flags: dict[str, Any] = {}
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-") and not _INT_RE.match(tok) and not _FLOAT_RE.match(tok):
            if "=" in tok:
                key, value = tok.split("=", 1)
                flags[canonical_key(key)] = _scalar(value)
            elif i + 1 < len(tokens) and (
                not tokens[i + 1].startswith("-")
                or _INT_RE.match(tokens[i + 1])
                or _FLOAT_RE.match(tokens[i + 1])
            ):
                flags[canonical_key(tok)] = _scalar(tokens[i + 1])
                i += 1
            else:
                flags[canonical_key(tok)] = True
        else:
            positional.append(tok)
        i += 1
    if positional:
        flags["_args"] = positional
    return flags


def normalize_params(params: Any) -> Any:
    if isinstance(params, str):
        if _FLAGS_RE.search(params):
            return normalize_params(parse_serve_flags(params))
        return _scalar(params)
    if isinstance(params, dict):
        out: dict[str, Any] = {}
        for k, v in params.items():
            out[canonical_key(str(k))] = normalize_params(v)
        return dict(sorted(out.items()))
    if isinstance(params, list):
        return [normalize_params(v) for v in params]
    if isinstance(params, bool):
        return params
    if isinstance(params, int):
        return params
    if isinstance(params, float):
        return int(params) if params.is_integer() else params
    return params


def config_fingerprint(params: Any) -> str:
    canonical = normalize_params(params)
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_env(env: Any) -> dict:
    """env 支持 dict、'K=V' 字符串列表、或多行/空格分隔的 K=V 文本。"""
    pairs: list[str] = []
    if isinstance(env, dict):
        out = {str(k): str(v) for k, v in env.items() if v is not None}
        return redact(out)
    if isinstance(env, str):
        for line in env.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pairs.extend(shlex.split(line))
    elif isinstance(env, list):
        pairs = [str(item) for item in env]
    else:
        raise ValueError("deploy.env 必须是对象、K=V 列表或 K=V 文本")
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"deploy.env 条目缺少 '=': {pair}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value
    return redact(out)


def normalize_deploy(deploy: Any) -> dict:
    """部署信息归一化：docker_cmd/env/serve_cmd/extra_cmds + 来源等附加字段；敏感值脱敏。

    extra_cmds：附加启动/编排命令列表，形如
    {"extra_cmds": [{"title": "head · ray start --head", "cmd": "docker exec ... ray start --head", "note": "..."}]}
    不参与配置指纹（指纹仅由 params 决定）。
    """
    if deploy is None:
        return {}
    if not isinstance(deploy, dict):
        raise ValueError("deploy 必须是 JSON 对象")
    out: dict = {}
    for key in ("docker_cmd", "serve_cmd"):
        val = deploy.get(key)
        if val is None:
            continue
        if not isinstance(val, str):
            raise ValueError(f"deploy.{key} 必须是字符串")
        val = val.strip()
        if val:
            out[key] = redact_cmd(val)
    if deploy.get("env") is not None:
        out["env"] = _normalize_env(deploy["env"])
    extra = deploy.get("extra_cmds")
    if extra is not None:
        if not isinstance(extra, list):
            raise ValueError("deploy.extra_cmds 必须是数组")
        items = []
        for item in extra:
            if not isinstance(item, dict) or not item.get("cmd"):
                raise ValueError("deploy.extra_cmds 每项必须是含 cmd 的对象")
            if not isinstance(item["cmd"], str):
                raise ValueError("deploy.extra_cmds[].cmd 必须是字符串")
            cmd = redact_cmd(item["cmd"].strip())
            if not cmd:
                continue
            entry = {"cmd": cmd}
            title = item.get("title")
            if title not in (None, ""):
                entry["title"] = redact_cmd(str(title).strip())
            note = item.get("note")
            if note not in (None, ""):
                entry["note"] = redact_cmd(str(note).strip())
            items.append(entry)
        if items:
            out["extra_cmds"] = items
    for key, val in deploy.items():
        if key in ("docker_cmd", "serve_cmd", "env", "extra_cmds") or val is None:
            continue
        if isinstance(val, str):
            out[key] = redact_cmd(val)
        elif isinstance(val, (dict, list)):
            out[key] = redact(val)
        else:
            out[key] = val
    return out


def seed_metric_defs(conn) -> None:
    for key, display, unit, direction in METRIC_SEED:
        conn.execute(
            "INSERT OR IGNORE INTO metric_def(key, display_name, unit, direction) VALUES(?,?,?,?)",
            (key, display, unit, direction),
        )


def resolve_model(conn, name: str, aliases: list[str] | None = None) -> dict:
    aliases = [a for a in (aliases or []) if a and a != name]
    candidates = {name, *aliases}
    rows = conn.execute("SELECT * FROM model").fetchall()
    for row in rows:
        existing = {row["name"], *json.loads(row["aliases"])}
        if existing & candidates:
            merged = sorted((existing | candidates) - {row["name"]})
            conn.execute(
                "UPDATE model SET aliases = ? WHERE id = ?",
                (json.dumps(merged, ensure_ascii=False), row["id"]),
            )
            return dict(conn.execute("SELECT * FROM model WHERE id = ?", (row["id"],)).fetchone())
    cur = conn.execute(
        "INSERT INTO model(name, aliases) VALUES(?,?)",
        (name, json.dumps(sorted(set(aliases)), ensure_ascii=False)),
    )
    return dict(conn.execute("SELECT * FROM model WHERE id = ?", (cur.lastrowid,)).fetchone())


def register_config(conn, model_id: int, name: str, params: Any, notes: str | None = None,
                    deploy: Any = None):
    fp = config_fingerprint(params)
    row = conn.execute(
        "SELECT * FROM config_set WHERE model_id = ? AND fingerprint = ?", (model_id, fp)
    ).fetchone()
    deploy_norm = normalize_deploy(deploy) if deploy is not None else {}
    if row:
        if deploy_norm:
            conn.execute("UPDATE config_set SET deploy = ? WHERE id = ?",
                         (json.dumps(deploy_norm, ensure_ascii=False), row["id"]))
            row = conn.execute("SELECT * FROM config_set WHERE id = ?", (row["id"],)).fetchone()
        return dict(row), False
    stored = normalize_params(params)
    base_name = name
    n = 1
    while True:
        named = conn.execute(
            "SELECT id, fingerprint FROM config_set WHERE model_id = ? AND name = ?",
            (model_id, name),
        ).fetchone()
        if named is None:
            break
        suffix = fp[:6] if n == 1 else f"{fp[:6]}-{n}"
        name = f"{base_name}-{suffix}"
        n += 1
    cur = conn.execute(
        "INSERT INTO config_set(model_id, name, params, fingerprint, notes, deploy)"
        " VALUES(?,?,?,?,?,?)",
        (model_id, name, json.dumps(stored, ensure_ascii=False), fp, notes,
         json.dumps(deploy_norm, ensure_ascii=False)),
    )
    return dict(conn.execute("SELECT * FROM config_set WHERE id = ?", (cur.lastrowid,)).fetchone()), True


def register_node(conn, label: str, description: str | None = None) -> dict:
    row = conn.execute("SELECT * FROM node WHERE label = ?", (label,)).fetchone()
    if row:
        return dict(row)
    cur = conn.execute("INSERT INTO node(label, description) VALUES(?,?)", (label, description))
    return dict(conn.execute("SELECT * FROM node WHERE id = ?", (cur.lastrowid,)).fetchone())


def ensure_suite(conn, key: str, name: str | None = None, description: str | None = None) -> dict:
    row = conn.execute("SELECT * FROM suite WHERE key = ?", (key,)).fetchone()
    if row:
        return dict(row)
    cur = conn.execute(
        "INSERT INTO suite(key, name, description) VALUES(?,?,?)", (key, name or key, description)
    )
    return dict(conn.execute("SELECT * FROM suite WHERE id = ?", (cur.lastrowid,)).fetchone())


def ensure_case(conn, suite_id: int, case_key: str, params: dict | None = None,
                description: str = "", sort_order: int = 0) -> dict:
    row = conn.execute(
        "SELECT * FROM test_case WHERE suite_id = ? AND case_key = ?", (suite_id, case_key)
    ).fetchone()
    if row:
        return dict(row)
    cur = conn.execute(
        "INSERT INTO test_case(suite_id, case_key, description, params, sort_order) VALUES(?,?,?,?,?)",
        (suite_id, case_key, description, json.dumps(params or {}, ensure_ascii=False), sort_order),
    )
    return dict(conn.execute("SELECT * FROM test_case WHERE id = ?", (cur.lastrowid,)).fetchone())