"""perf-viewer-import：按场景配置文件完成 EvalScope 等结果的上报。

用法：
    perf-viewer-import --config scenarios/evalscope-64k.yaml
    perf-viewer-import --config scenarios/evalscope-64k.yaml --apply

配置的最小形式：
    model: {name: cltech/glm53}
    scenario: {name: 单机最佳交付场景}
    client:
      tool: evalscope
      result_dir: ./result
      groups:
        - case_key: long-64k-cache90-256
        - case_key: long-64k-cache90-c2-256
    server: {node: 192.168.80.10, deploy_dir: ./raw/deploy-logs}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency error path
    yaml = None

# 触发 .env 加载（setdefault，不覆盖已导出的环境变量）
from . import config as viewer_config  # noqa: E402,F401

DEFAULT_IMPORT_KIT = (
    Path(__file__).resolve().parent.parent
    / ".kilo/skills/perf-viewer-import/scripts/import_kit.py"
)
SHORT_PARAMS = {
    "-tp": "tensor_parallel_size",
    "-dcp": "decode_context_parallel_size",
}
INTERESTING_PARAMS = {
    "tensor_parallel_size",
    "decode_context_parallel_size",
    "block_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "moe_ep_size",
    "num_speculative_tokens",
    "enable_prefix_caching",
    "max_model_len",
    "gpu_memory_utilization",
    "dtype",
    "enable_expert_parallel",
    "enable_chunked_prefill",
    "kv_cache_dtype",
    "long_prefill_token_threshold",
    "speculative_method",
    "enable_auto_tool_choice",
    "tool_call_parser",
    "reasoning_parser",
}
SKIP_PARAMS = {
    "host",
    "port",
    "served_model_name",
    "api_key",
    "token",
    "log",
    "uvicorn_log_level",
}


class ConfigError(Exception):
    """配置文件或环境不满足上报要求。"""


def _load_config(path: Path) -> dict:
    if yaml is None:
        raise ConfigError("缺少 PyYAML，请先安装 pyyaml")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"无法读取配置 {path}: {exc}") from exc
    except Exception as exc:  # yaml.YAMLError
        raise ConfigError(f"配置 YAML 解析失败 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是对象")
    return data


def _get(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _env_items(value: Any) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    items = value if isinstance(value, list) else str(value).replace(",", " ").split()
    out: dict[str, str] = {}
    for item in items:
        text = str(item).strip()
        if text.startswith("export "):
            text = text[len("export "):].strip()
        if "=" in text:
            key, _, val = text.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key:
                out[key] = val
    return out


def _viewer_settings(cfg: dict, args) -> tuple[str, str]:
    viewer = cfg.get("viewer") or {}
    url = (
        args.base_url
        or _get(viewer, "url")
        or os.environ.get("PERF_VIEWER_URL")
        or os.environ.get("PERF_VIEWER_BASE_URL")
        or "http://127.0.0.1:8600"
    )
    token_env = _get(viewer, "token_env") or "PERF_VIEWER_TOKEN"
    token = args.token or _get(viewer, "token") or os.environ.get(token_env, "")
    if not token:
        raise ConfigError(
            f"写接口需要 token：请 export {token_env}=...，或在配置 viewer.token_env 指向已设置的变量"
        )
    return str(url).rstrip("/"), str(token)


def _get_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取 {url}: {exc}") from exc


def _model_aliases(model: dict) -> set[str]:
    aliases = model.get("aliases") or []
    if isinstance(aliases, str):
        try:
            aliases = json.loads(aliases)
        except json.JSONDecodeError:
            aliases = [aliases]
    return {str(model.get("name") or ""), *[str(a) for a in aliases]}


def _resolve_scenario(cfg: dict, viewer_url: str, model_name: str) -> int | None:
    scenario = cfg.get("scenario") or {}
    scenario_id = _get(scenario, "id")
    if scenario_id is not None:
        try:
            return int(scenario_id)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"scenario.id 不是整数: {scenario_id}") from exc
    name = _get(scenario, "name")
    if not name:
        return None
    catalog = _get_json(f"{viewer_url}/api/catalog")
    models = catalog.get("models") or []
    model = next((m for m in models if model_name in _model_aliases(m)), None)
    if model is None:
        raise ConfigError(f"/api/catalog 中找不到模型 {model_name}")
    matches = [
        s for s in (catalog.get("scenarios") or [])
        if s.get("name") == name and s.get("model_id") == model.get("id")
    ]
    if not matches:
        names = sorted({str(s.get("name")) for s in (catalog.get("scenarios") or [])})
        raise ConfigError(f"模型 {model_name} 下找不到场景 {name}；可选: {', '.join(names)}")
    if len(matches) > 1:
        raise ConfigError(f"模型 {model_name} 下场景 {name} 不唯一")
    return int(matches[0]["id"])


def _logical_lines(text: str):
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        buffer = f"{buffer} {line.strip()}".strip() if buffer else line.strip()
        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue
        yield buffer
        buffer = ""
    if buffer:
        yield buffer


def _parse_deploy_dir(path: Path) -> dict:
    """从已保存的部署日志/脚本中提取 docker / vllm / ray 命令和 env。"""
    if not path.is_dir():
        raise ConfigError(f"deploy_dir 不存在或不是目录: {path}")
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".sh", ".txt", ".env", ".log"}
    )
    docker_cmd = ""
    serve_cmd = ""
    extra_cmds: list[dict] = []
    env: dict[str, str] = {}
    seen_extra: set[str] = set()
    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        for line in _logical_lines(text):
            if "docker run" in line and not docker_cmd:
                docker_cmd = line
            if "vllm serve" in line and not serve_cmd:
                serve_cmd = line
            if re.search(r"\bray start\b", line) and line not in seen_extra:
                title = "Ray head" if "--head" in line else "Ray join"
                extra_cmds.append({"title": title, "cmd": line})
                seen_extra.add(line)
            match = re.match(
                r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip()
            )
            if match:
                key, value = match.groups()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                env[key] = value
    return {
        "docker_cmd": docker_cmd,
        "serve_cmd": serve_cmd,
        "extra_cmds": extra_cmds,
        "env": env,
    }


def _scalar(value: str):
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_serve_params(command: str) -> dict:
    """从 vllm serve 命令解析影响性能的参数。"""
    if not command:
        return {}
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    params: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"nohup", "&", "|", "&&"} or token.startswith(("2>", ">", "<")):
            index += 1
            continue
        if not token.startswith("-"):
            index += 1
            continue
        if token.startswith("--") and "=" in token:
            raw_key, _, raw_value = token.partition("=")
            key = raw_key.lstrip("-").replace("-", "_")
            value = _scalar(raw_value)
            index += 1
        elif token in SHORT_PARAMS:
            key = SHORT_PARAMS[token]
            value = _scalar(tokens[index + 1]) if index + 1 < len(tokens) else True
            index += 2
        else:
            key = token.lstrip("-").replace("-", "_")
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                value = _scalar(tokens[index + 1])
                index += 2
            else:
                value = True
                index += 1
        if key in SKIP_PARAMS:
            continue
        if key == "speculative_config":
            try:
                spec = value if isinstance(value, dict) else json.loads(value)
            except (TypeError, json.JSONDecodeError):
                continue
            if spec.get("num_speculative_tokens") is not None:
                params["num_speculative_tokens"] = spec["num_speculative_tokens"]
            if spec.get("method"):
                params["speculative_method"] = spec["method"]
            continue
        if key in INTERESTING_PARAMS:
            params[key] = value
    return params


def _image_from_docker(command: str) -> str:
    if not command:
        return ""
    matches = re.findall(
        r"(?<![\w./-])([A-Za-z0-9][A-Za-z0-9._/-]*:[A-Za-z0-9._-]+)(?![\w./-])",
        command,
    )
    return matches[-1] if matches else ""


def _auto_config_name(model: str, docker_cmd: str, params: dict, tool: str) -> str:
    model_short = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-") or "model"
    container = ""
    match = re.search(r"--name\s+([^\s]+)", docker_cmd or "")
    if match:
        container = match.group(1)
    container = re.sub(r"^(?:(?:workflow|vllm)_)+", "", container)
    container = re.sub(r"[^a-z0-9]+", "-", container.lower()).strip("-")
    parts = [model_short]
    if params.get("tensor_parallel_size"):
        parts.append(f"tp{params['tensor_parallel_size']}")
    if params.get("decode_context_parallel_size"):
        parts.append(f"dcp{params['decode_context_parallel_size']}")
    if params.get("num_speculative_tokens"):
        parts.append(f"mtp{params['num_speculative_tokens']}")
    if container:
        suffix = container.replace(model_short, "").strip("-")
        if suffix:
            parts.append(suffix)
    if len(parts) == 1:
        parts.append(tool)
    name = "-".join(dict.fromkeys(part for part in parts if part))
    return re.sub(r"-+", "-", name).strip("-")


def _find_summary(result_dir: Path, case_key: str) -> Path:
    root = result_dir / case_key
    matches = sorted(root.glob("**/benchmark_summary.json"))
    if not matches:
        raise ConfigError(
            f"在 {root} 下找不到 benchmark_summary.json；请检查 result_dir/case_key 或显式 input"
        )
    if len(matches) > 1:
        raise ConfigError(
            f"{root} 下有多个 benchmark_summary.json，请用 input 指定: "
            + ", ".join(str(m) for m in matches)
        )
    return matches[0]


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)) or "case"


def _case_groups(client: dict, result_dir: Path | None) -> list[dict]:
    groups = _as_list(client.get("groups"))
    if groups:
        return [dict(g) for g in groups if isinstance(g, dict)]
    if result_dir and result_dir.is_dir():
        return [
            {"case_key": child.name}
            for child in sorted(result_dir.iterdir())
            if child.is_dir()
            and "warmup" not in child.name.lower()
            and list(child.glob("**/benchmark_summary.json"))
        ]
    raise ConfigError("client.groups 为空，且 client.result_dir 下没有可发现的 case")


def _case_input(tool: str, group: dict, result_dir: Path | None) -> tuple[str, dict]:
    case_key = str(group.get("case_key") or "")
    if tool in ("evalscope", "evalscope-perf"):
        if group.get("input"):
            return str(group["input"]), {"case_key": case_key}
        if not case_key or result_dir is None:
            raise ConfigError("evalscope 需要 case_key 或 input")
        return str(_find_summary(result_dir, case_key)), {"case_key": case_key}
    if tool in ("vllm-bench", "vllm_bench", "benchmark_serving"):
        if group.get("input"):
            return str(group["input"]), {"case_key": case_key}
        if not case_key or result_dir is None:
            raise ConfigError("vllm-bench 需要 case_key 或 input")
        root = result_dir / case_key
        matches = sorted(root.glob("**/*.json"))
        if not matches:
            raise ConfigError(f"在 {root} 下找不到 vllm-bench 结果 JSON")
        return str(matches[-1]), {"case_key": case_key}
    return "", {}


def _common_args(
    *,
    out: Path,
    model: str,
    aliases: list[str],
    config_name: str,
    params: dict,
    image: str,
    node: str,
    worker_label: str,
    suite: str,
    scenario_id: int | None,
    docker_cmd: str,
    serve_cmd: str,
    env: dict[str, str],
    extra_cmds: list[dict],
    source: str,
    notes: str,
    archive_raw: bool,
    deploy_dir: Path | None,
) -> list[str]:
    args = ["--out", str(out), "--model", model, "--config-name", config_name,
            "--node", node, "--suite", suite]
    for alias in aliases:
        args += ["--alias", str(alias)]
    for key, value in params.items():
        args += ["--param", f"{key}={json.dumps(value, ensure_ascii=False)}"]
    if image:
        args += ["--image", image]
    if worker_label:
        args += ["--worker-label", worker_label]
    if scenario_id is not None:
        args += ["--scenario-id", str(scenario_id)]
    if docker_cmd:
        args += ["--docker-cmd", docker_cmd]
    if serve_cmd:
        args += ["--serve-cmd", serve_cmd]
    for key, value in env.items():
        args += ["--env", f"{key}={json.dumps(str(value), ensure_ascii=False)}"]
    for item in extra_cmds:
        if isinstance(item, dict):
            command = item.get("cmd")
            title = item.get("title") or ""
        else:
            command = str(item)
            title = ""
        if command:
            args += ["--bench-cmd", f"{title}::{command}"]
    if source:
        args += ["--source", source]
    if notes:
        args += ["--notes", notes]
    args += ["--archive-raw"] if archive_raw else ["--no-archive-raw"]
    if deploy_dir is not None:
        args += ["--deploy-dir", str(deploy_dir)]
    return args


def _run_import_kit(script: Path, subcommand: str, common: list[str], extra: list[str]) -> None:
    if not script.is_file():
        raise ConfigError(
            f"找不到 import_kit.py: {script}；可设置 PERF_VIEWER_IMPORT_KIT 指向脚本"
        )
    command = [sys.executable, str(script), subcommand, *common, *extra]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ConfigError(f"import_kit.py 失败（{subcommand}）: {detail}")


def _copy_optional_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _include_environment(out: Path, includes: list) -> None:
    """把用户指定的环境信息文件/目录复制到 out/environment/，随 zip 一起上传。"""
    dest_root = out / "environment"
    dest_root.mkdir(parents=True, exist_ok=True)
    indexed: list[str] = []
    for raw in includes:
        source = Path(str(raw)).expanduser()
        if not source.exists():
            raise ConfigError(f"archive.include 路径不存在: {source}")
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(source)
                dest = dest_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                indexed.append(str(dest.relative_to(out)))
        else:
            dest = dest_root / source.name
            shutil.copy2(source, dest)
            indexed.append(str(dest.relative_to(out)))
    (dest_root / "_index.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": indexed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _time_min(values: list[str | None]) -> str | None:
    valid = [v for v in values if v]
    return min(valid) if valid else None


def _time_max(values: list[str | None]) -> str | None:
    valid = [v for v in values if v]
    return max(valid) if valid else None


def _merge_parts(
    parts: list[dict],
    out: Path,
    cases_cfg: dict,
    tool: str,
) -> tuple[dict, dict]:
    normalizeds = []
    manifests = []
    for part in parts:
        part_dir = Path(part["out"])
        normalized = json.loads((part_dir / "normalized.json").read_text(encoding="utf-8"))
        manifest = json.loads((part_dir / "manifest.json").read_text(encoding="utf-8"))
        normalizeds.append(normalized)
        manifests.append(manifest)
        for name in ("case-logs", "benchmark-suite", "server-start", "console-logs"):
            _copy_optional_tree(part_dir / name, out / name)
        raw_results = part_dir / "raw-results"
        if raw_results.exists():
            _copy_optional_tree(raw_results, out / "raw-results" / _safe_key(part["case_key"]))

    config_names = {n.get("config", {}).get("name") for n in normalizeds}
    if len(config_names) > 1:
        raise ConfigError(f"多个 case 的配置不一致: {', '.join(map(str, config_names))}")
    nodes = {n.get("node_label") for n in normalizeds}
    if len(nodes) > 1:
        raise ConfigError(f"多个 case 的节点不一致: {', '.join(map(str, nodes))}")
    suites = {n.get("suite_key") for n in normalizeds}
    if len(suites) > 1:
        raise ConfigError(f"多个 case 的 suite 不一致: {', '.join(map(str, suites))}")

    base = dict(normalizeds[0])
    cases: list[dict] = []
    seen: set[str] = set()
    for normalized in normalizeds:
        for case in normalized.get("cases") or []:
            key = str(case.get("case_key") or "")
            if key in seen:
                raise ConfigError(f"case_key 重复: {key}")
            seen.add(key)
            cases.append(case)
    base["cases"] = cases
    base["started_at"] = _time_min([n.get("started_at") for n in normalizeds])
    base["ended_at"] = _time_max([n.get("ended_at") for n in normalizeds])
    base["valid"] = all(bool(n.get("valid", True)) for n in normalizeds)

    manifest = dict(manifests[0])
    manifest_cases: dict[str, dict] = {}
    for item in manifests:
        manifest_cases.update(item.get("cases") or {})
    if manifest_cases:
        manifest["cases"] = manifest_cases
    manifest["started_at"] = base["started_at"]
    manifest["ended_at"] = base["ended_at"]

    case_by_key = {str(c.get("case_key")): c for c in cases}
    for key, meta in (cases_cfg or {}).items():
        if not isinstance(meta, dict) or str(key) not in case_by_key:
            continue
        case = case_by_key[str(key)]
        if meta.get("description"):
            case["description"] = meta["description"]
        if meta.get("repro_cmd"):
            case["repro_cmd"] = meta["repro_cmd"]
        for field in ("stdout_log", "stderr_log", "console_log"):
            source = meta.get(field)
            if not source:
                continue
            path = Path(source)
            if not path.is_file():
                raise ConfigError(f"cases.{key}.{field} 不存在: {source}")
            dest = out / "case-logs" / _safe_key(key) / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            case[field] = str(dest.relative_to(out))

    for part in parts:
        warmup = part.get("warmup")
        if not warmup:
            continue
        key = str(part["case_key"])
        source = Path(warmup)
        if not source.is_file():
            raise ConfigError(f"预热结果不存在: {source}")
        dest = out / "case-logs" / _safe_key(key) / "warmup.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if key in case_by_key:
            case_by_key[key]["warmup_log"] = str(dest.relative_to(out))
        deploy = manifest.setdefault("config", {}).setdefault("deploy", {})
        extra = deploy.setdefault("extra_cmds", [])
        command = part.get("warmup_command")
        if command:
            extra.append({"title": f"{key} · warmup", "cmd": command})

    (out / "normalized.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "parts": [{"case_key": p["case_key"], "source": p["input"]} for p in parts],
    }
    (out / "archive.json").write_text(
        json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, base


def _ensure_suite_link(viewer_url: str, token: str, scenario_id: int | None, suite_key: str) -> None:
    if scenario_id is None:
        return
    try:
        catalog = _get_json(f"{viewer_url}/api/catalog")
        scenario = next(
            (s for s in (catalog.get("scenarios") or []) if s.get("id") == scenario_id),
            None,
        )
        if scenario is None:
            return
        suite_by_id = {s.get("id"): s.get("key") for s in (catalog.get("suites") or [])}
        keys = [
            suite_by_id[i] for i in (scenario.get("suite_ids") or [])
            if i in suite_by_id and suite_by_id[i]
        ]
        if suite_key in keys:
            return
        keys.append(suite_key)
        body = json.dumps(
            {
                "name": scenario.get("name"),
                "description": scenario.get("description") or "",
                "suite_keys": keys,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{viewer_url}/api/catalog/scenarios/{scenario_id}",
            data=body,
            method="PATCH",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
        print(f"已关联 suite {suite_key} 到场景 {scenario.get('name')}", file=sys.stderr)
    except Exception as exc:  # 关联失败不应让已完成的导入失败
        print(f"warning: suite 关联失败: {exc}", file=sys.stderr)


def _host_from_url(url: str | None) -> str:
    if not url:
        return ""
    match = re.match(r"^[a-z]+://([^/:]+)", str(url))
    return match.group(1) if match else ""


def _run(args) -> int:
    cfg_path = Path(args.config).expanduser()
    cfg = _load_config(cfg_path)
    viewer_url, token = _viewer_settings(cfg, args)

    model = _get(cfg, "model", "name")
    if not model:
        raise ConfigError("model.name 必填")
    aliases = [str(a) for a in _as_list(_get(cfg, "model", "aliases"))]
    scenario_id = _resolve_scenario(cfg, viewer_url, str(model))

    client = cfg.get("client") or {}
    raw_tool = str(client.get("tool") or "evalscope").lower()
    if raw_tool in {"evalscope", "evalscope-perf"}:
        tool = "evalscope"
    elif raw_tool in {"vllm-bench", "vllm_bench", "benchmark_serving"}:
        tool = "vllm-bench"
    elif raw_tool == "benchmark-suite":
        tool = "benchmark-suite"
    elif raw_tool in {"guidellm", "guide-llm"}:
        tool = "guidellm"
    else:
        raise ConfigError(
            f"暂不支持 client.tool={raw_tool}；当前支持 evalscope / vllm-bench / benchmark-suite / guidellm"
        )
    suite = _get(cfg, "suite", "key") or tool

    result_dir = client.get("result_dir")
    result_path = Path(result_dir).expanduser() if result_dir else None
    groups = _case_groups(client, result_path)

    server = cfg.get("server") or {}
    deploy_cfg = server.get("deploy") or {}
    deploy_dir_value = server.get("deploy_dir") or _get(cfg, "archive", "deploy_dir")
    deploy_dir = Path(deploy_dir_value).expanduser() if deploy_dir_value else None
    parsed_deploy = _parse_deploy_dir(deploy_dir) if deploy_dir else {}
    docker_cmd = deploy_cfg.get("docker_cmd") or parsed_deploy.get("docker_cmd", "")
    serve_cmd = deploy_cfg.get("serve_cmd") or parsed_deploy.get("serve_cmd", "")
    env = {**parsed_deploy.get("env", {}), **_env_items(deploy_cfg.get("env"))}
    extra_cmds = list(parsed_deploy.get("extra_cmds") or []) + list(deploy_cfg.get("extra_cmds") or [])
    if deploy_cfg.get("source"):
        source = deploy_cfg["source"]
    else:
        node_for_source = server.get("node") or _host_from_url(client.get("url"))
        source = f"部署记录 · {node_for_source or 'unknown'} · {datetime.now().strftime('%Y-%m-%d')}"

    raw_params = server.get("params")
    if isinstance(raw_params, dict):
        params = dict(raw_params)
    elif isinstance(raw_params, str):
        params = _parse_serve_params(raw_params)
    else:
        params = {}
    if not params:
        params = _parse_serve_params(serve_cmd)
    image = server.get("image") or _image_from_docker(docker_cmd)
    config_name = server.get("config_name") or _auto_config_name(
        str(model), docker_cmd, params, tool
    )
    node = server.get("node") or _host_from_url(client.get("url"))
    if not node:
        raise ConfigError("server.node 必填（或 client.url 能提供主机名）")
    worker_label = str(server.get("worker_label") or "")
    notes = str(server.get("notes") or "")
    archive_raw = bool(_get(cfg, "archive", "raw", default=True))
    out = Path(
        _get(cfg, "archive", "output")
        or f"./import/{re.sub(r'[^A-Za-z0-9._-]+', '-', str(model).split('/')[-1])}-"
           f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ).expanduser()

    if out.exists() and any(out.iterdir()) and not args.force:
        raise ConfigError(f"输出目录已存在且非空: {out}；加 --force 或换 archive.output")
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    includes = _as_list(_get(cfg, "archive", "include"))
    if includes:
        _include_environment(out, includes)

    dry_run = bool(_get(cfg, "import", "dry_run", default=True))
    if args.dry_run:
        dry_run = True
    if args.apply:
        dry_run = False

    if tool == "guidellm":
        for index, group in enumerate(groups):
            directory = group.get("dir")
            if not directory:
                raise ConfigError("guidellm 每组需要 dir（GuideLLM 结果目录）")
            directory_path = Path(directory).expanduser()
            if not directory_path.is_dir():
                raise ConfigError(f"guidellm dir 不存在: {directory_path}")
            case_key = str(group.get("case_key") or directory_path.name or f"guidellm-{index}")
            shutil.copytree(directory_path, out / _safe_key(case_key), dirs_exist_ok=True)
        if deploy_dir and deploy_dir.is_dir():
            shutil.copytree(
                deploy_dir, out / "server-start" / deploy_dir.name, dirs_exist_ok=True
            )
        if image:
            params = {**params, "image": params.get("image") or image}
        deploy = {}
        if docker_cmd:
            deploy["docker_cmd"] = docker_cmd
        if serve_cmd:
            deploy["serve_cmd"] = serve_cmd
        if env:
            deploy["env"] = env
        if extra_cmds:
            deploy["extra_cmds"] = extra_cmds
        if source:
            deploy["source"] = source
        manifest = {
            "model": {"name": str(model), "aliases": aliases},
            "config": {
                "name": config_name,
                "params": params,
                **({"notes": notes} if notes else {}),
                **({"deploy": deploy} if deploy else {}),
            },
            "suite_key": suite,
            "backend": "guidellm-suite",
            "node": node,
            "worker_label": worker_label or None,
            "scenario_id": scenario_id,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        from .cli import cmd_import
        import argparse as _argparse

        print(json.dumps({
            "viewer": viewer_url,
            "model": model,
            "scenario_id": scenario_id,
            "suite": suite,
            "config": config_name,
            "node": node,
            "tool": tool,
            "output": str(out),
            "dry_run": dry_run,
        }, ensure_ascii=False, indent=2))
        rc = cmd_import(_argparse.Namespace(
            source=str(out),
            manifest=str(out / "manifest.json"),
            dry_run=dry_run,
            base_url=viewer_url,
            token=token,
        ))
        if rc == 0 and not dry_run:
            _ensure_suite_link(viewer_url, token, scenario_id, suite)
        return rc

    script = Path(os.environ.get("PERF_VIEWER_IMPORT_KIT", DEFAULT_IMPORT_KIT)).expanduser()
    parts_root = Path(tempfile.mkdtemp(prefix="perf-viewer-import-"))
    parts: list[dict] = []
    try:
        for index, group in enumerate(groups):
            tool_key = tool
            if tool_key in {"evalscope", "evalscope-perf"}:
                input_path, info = _case_input("evalscope", group, result_path)
                case_key = info["case_key"] or Path(input_path).parent.name
                part_out = parts_root / f"{index:02d}-{_safe_key(case_key)}"
                common = _common_args(
                    out=part_out, model=str(model), aliases=aliases, config_name=config_name,
                    params=params, image=image, node=node, worker_label=worker_label,
                    suite=suite, scenario_id=scenario_id, docker_cmd=docker_cmd,
                    serve_cmd=serve_cmd, env=env, extra_cmds=extra_cmds, source=source,
                    notes=notes, archive_raw=archive_raw,
                    deploy_dir=deploy_dir if index == 0 else None,
                )
                case_params = group.get("case_params") or {}
                case_args = ["--summary", input_path, "--case-key", case_key,
                             "--time-unit", str(client.get("time_unit") or "s")]
                for key, value in case_params.items():
                    case_args += ["--case-param", f"{key}={json.dumps(value, ensure_ascii=False)}"]
                if group.get("description"):
                    case_args += ["--description", str(group["description"])]
                _run_import_kit(script, "evalscope", common, case_args)
                parts.append({
                    "case_key": case_key,
                    "input": input_path,
                    "out": str(part_out),
                    "warmup": _get(group, "warmup", "input"),
                    "warmup_command": _get(group, "warmup", "command"),
                })
            elif tool_key in {"vllm-bench", "vllm_bench", "benchmark_serving"}:
                input_path, info = _case_input("vllm-bench", group, result_path)
                case_key = info["case_key"] or Path(input_path).stem
                part_out = parts_root / f"{index:02d}-{_safe_key(case_key)}"
                common = _common_args(
                    out=part_out, model=str(model), aliases=aliases, config_name=config_name,
                    params=params, image=image, node=node, worker_label=worker_label,
                    suite=suite, scenario_id=scenario_id, docker_cmd=docker_cmd,
                    serve_cmd=serve_cmd, env=env, extra_cmds=extra_cmds, source=source,
                    notes=notes, archive_raw=archive_raw,
                    deploy_dir=deploy_dir if index == 0 else None,
                )
                case_args = ["--result", input_path, "--case-key", case_key]
                for key, value in (group.get("case_params") or {}).items():
                    case_args += ["--case-param", f"{key}={json.dumps(value, ensure_ascii=False)}"]
                if group.get("description"):
                    case_args += ["--description", str(group["description"])]
                _run_import_kit(script, "vllm-bench", common, case_args)
                parts.append({
                    "case_key": case_key,
                    "input": input_path,
                    "out": str(part_out),
                    "warmup": _get(group, "warmup", "input"),
                    "warmup_command": _get(group, "warmup", "command"),
                })
            else:
                summary = group.get("summary")
                config_path = group.get("config")
                if not summary or not config_path:
                    raise ConfigError("benchmark-suite 每组需要 summary 和 config")
                case_key = str(group.get("case_key") or f"benchmark-suite-{index}")
                part_out = parts_root / f"{index:02d}-{_safe_key(case_key)}"
                common = _common_args(
                    out=part_out, model=str(model), aliases=aliases, config_name=config_name,
                    params=params, image=image, node=node, worker_label=worker_label,
                    suite=suite, scenario_id=scenario_id, docker_cmd=docker_cmd,
                    serve_cmd=serve_cmd, env=env, extra_cmds=extra_cmds, source=source,
                    notes=notes, archive_raw=archive_raw,
                    deploy_dir=deploy_dir if index == 0 else None,
                )
                case_args = ["--summary", str(summary), "--config", str(config_path)]
                if group.get("requests"):
                    case_args += ["--requests", str(group["requests"])]
                if group.get("include_failed"):
                    case_args.append("--include-failed")
                _run_import_kit(script, "benchmark-suite", common, case_args)
                parts.append({
                    "case_key": case_key,
                    "input": str(summary),
                    "out": str(part_out),
                    "warmup": _get(group, "warmup", "input"),
                    "warmup_command": _get(group, "warmup", "command"),
                })

        manifest, normalized = _merge_parts(parts, out, cfg.get("cases") or {}, tool)

        from .cli import cmd_import
        import argparse as _argparse

        print(json.dumps({
            "viewer": viewer_url,
            "model": model,
            "scenario_id": scenario_id,
            "suite": suite,
            "config": config_name,
            "node": node,
            "tool": tool,
            "cases": [c.get("case_key") for c in normalized.get("cases") or []],
            "output": str(out),
            "dry_run": dry_run,
        }, ensure_ascii=False, indent=2))
        rc = cmd_import(_argparse.Namespace(
            source=str(out),
            manifest=str(out / "manifest.json"),
            dry_run=dry_run,
            base_url=viewer_url,
            token=token,
        ))
        if rc == 0 and not dry_run:
            _ensure_suite_link(viewer_url, token, scenario_id, suite)
        return rc
    finally:
        if args.keep_parts:
            print(f"parts 保留在: {parts_root}", file=sys.stderr)
        else:
            shutil.rmtree(parts_root, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="perf-viewer-import",
        description="按场景配置完成压测结果上报（EvalScope 等）",
    )
    parser.add_argument("--config", required=True, help="场景配置文件（YAML）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="只校验，不写库")
    group.add_argument("--apply", action="store_true", help="正式导入（覆盖 import.dry_run）")
    parser.add_argument("--base-url", default=None, help="perf-viewer 地址，默认取环境变量")
    parser.add_argument("--token", default=None, help="写接口 token，默认取环境变量")
    parser.add_argument("--force", action="store_true", help="输出目录非空时先删除")
    parser.add_argument("--keep-parts", action="store_true", help="保留中间转换目录，便于排障")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
