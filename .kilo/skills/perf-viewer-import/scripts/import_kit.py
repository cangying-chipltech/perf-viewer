#!/usr/bin/env python3
"""perf-viewer 数据导入辅助：把常见压测产物转换为 generic-json 可导入的目录。

用法:
  import_kit.py vllm-bench --result result.json --out ./import/run1 --model ... --config-name ... --node ...
  import_kit.py evalscope --summary benchmark_summary.json --out ./import/run1 ...
  import_kit.py benchmark-suite --summary summary.json --config config.json --out ./import/run1 ...

每次转换都会把本次执行的原始产物归档进 <out>（benchmark-suite 原始摘要/case 命令、
vllm-bench/evalscope 源文件），并在给出 --deploy-execution / --deploy-dir 时把
服务端启动文件（docker run / vllm serve / ray start 的 commands/*.executed.sh 与
summary.json）归档到 <out>/server-start/。导入时这些文件作为 run 产物上传，
perf-viewer 会长期保留在 data/artifacts 下，可在 run 详情页下载。

生成 <out>/normalized.json、<out>/manifest.json 与 <out>/archive.json（归档索引），然后:
  cd /home/cangying/perf-viewer
  .venv/bin/python -m app.cli import <out> --manifest <out>/manifest.json --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SECONDARY_IMAGE_SUFFIXES = ("_with_ray",)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_json(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"无法读取 JSON {path}: {exc}")


def parse_pairs(items: list[str] | None, label: str) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            die(f"{label} 需要 K=V 格式: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            die(f"{label} key 为空: {item}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def parse_raw_pairs(items: list[str] | None, label: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            die(f"{label} 需要 K=V 格式: {item}")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            die(f"{label} key 为空: {item}")
        out[key] = value
    return out


def normalize_image(image: str) -> str:
    if not image:
        return image
    short = image.strip().split("/")[-1]
    for suffix in SECONDARY_IMAGE_SUFFIXES:
        short = short.replace(suffix, "")
    return short


def image_from_docker_cmd(cmd: str) -> str:
    if not cmd:
        return ""
    import shlex
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return ""
    return tokens[-1] if tokens else ""


# --------------------------------------------------------------- 执行归档
# 每次转换都把原始用例结果与服务端启动文件一并放进输出目录；perf-viewer 导入时
# 未经适配器消费的文件会落成 run 产物（data/artifacts），因此这些文件就随导入归档。

ARCHIVE_DIR = "server-start"
RAW_DIR = "benchmark-suite"
RESULT_DIR = "raw-results"
_SAFE_EXECUTION_ID = re.compile(r"[A-Za-z0-9_.-]+")
DEFAULT_SERVICE_PORT = "18729"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_copy(out: Path, src: Path, rel: str) -> str:
    """把 src 复制到 out/rel；返回相对 out 的路径。"""
    src = Path(src)
    if not src.is_file():
        die(f"归档文件不存在: {src}")
    dest = Path(out) / rel
    if dest.exists() and dest.resolve() == src.resolve():
        return dest.relative_to(out).as_posix()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.relative_to(out).as_posix()


def _archive_dir(out: Path, src: Path, rel: str) -> list[str]:
    """把目录内容复制到 out/rel；返回相对 out 的文件路径列表。"""
    src = Path(src)
    if not src.exists():
        die(f"归档路径不存在: {src}")
    written: list[str] = []
    if src.is_file():
        return [_archive_copy(out, src, Path(rel) / src.name)]
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        sub = path.relative_to(src)
        written.append(_archive_copy(out, path, str(Path(rel) / sub)))
    return written


def is_startup_file(name: str) -> bool:
    """部署执行里属于服务端启动的文件：容器启动、vllm serve、ray start。"""
    return name == "summary.json" or (name.startswith("commands/") and name.endswith(".sh"))


def _service_urls(values: list[str] | None) -> list[str]:
    urls: list[str] = []
    for value in values or []:
        url = str(value or "").strip().rstrip("/")
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            die(f"--service-url 需要 http(s) 地址: {url}")
        if url not in urls:
            urls.append(url)
    if urls:
        return urls
    for name in ("PROFILE_SERVICE_URL", "PROFILE_SERVICE_LOCAL_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return [value.rstrip("/")]
    port = os.environ.get("PROFILE_SERVICE_PORT", DEFAULT_SERVICE_PORT).strip() or DEFAULT_SERVICE_PORT
    return [f"http://127.0.0.1:{port}"]


def _service_get(base: str, path: str, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(base + path, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        exc.close()
        raise RuntimeError(f"{base}{path}: HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"{base}{path}: {exc}") from None
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{base}{path}: 返回的不是合法 JSON ({exc})") from None
    if not isinstance(data, dict):
        raise RuntimeError(f"{base}{path}: 返回的不是 JSON 对象")
    return data


def fetch_server_start(out: Path, execution_id: str, urls: list[str]) -> dict:
    """从 profile-service 拉取一个部署执行的启动命令并归档到 <out>/server-start/。"""
    execution_id = str(execution_id or "").strip()
    if not _SAFE_EXECUTION_ID.fullmatch(execution_id):
        die(f"--deploy-execution 非法 id: {execution_id!r}")
    errors: list[str] = []
    for base in urls:
        exec_path = urllib.parse.quote(execution_id, safe="")
        try:
            listing = _service_get(base, f"/api/executions/{exec_path}/logs")
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        names = [str(name) for name in (listing.get("files") or [])]
        wanted = sorted(name for name in names if is_startup_file(name))
        if not wanted:
            errors.append(f"{base}: 执行 {execution_id} 没有启动命令文件")
            continue
        written: list[str] = []
        for name in wanted:
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts:
                die(f"部署执行返回了不安全的文件路径: {name}")
            try:
                response = _service_get(
                    base, f"/api/executions/{exec_path}/logs/{urllib.parse.quote(name, safe='')}"
                )
            except RuntimeError as exc:
                errors.append(str(exc))
                written = []
                break
            content = response.get("content")
            if not isinstance(content, str):
                errors.append(f"{base}: {name} 没有可下载内容")
                written = []
                break
            dest = Path(out) / ARCHIVE_DIR / execution_id / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written.append(dest.relative_to(out).as_posix())
        if written:
            return {"execution": execution_id, "service": base, "files": written}
        if not errors:
            errors.append(f"{base}: 执行 {execution_id} 的启动命令读取失败")
    die("无法归档服务端启动文件: " + "; ".join(errors))


def archive_sources(out: Path, args) -> dict:
    """归档服务端启动文件（部署执行 / 部署目录），返回写入 archive.json 的段落。"""
    started: list[dict] = []
    urls = _service_urls(args.service_url)
    for execution_id in args.deploy_execution or []:
        started.append(fetch_server_start(out, execution_id, urls))
    for directory in args.deploy_dir or []:
        src = Path(directory).expanduser()
        written = _archive_dir(out, src, str(Path(ARCHIVE_DIR) / src.name))
        started.append({"dir": str(src), "files": written})
    return {"server_start": started}


def write_archive_index(out: Path, meta: dict) -> Path:
    """写 <out>/archive.json：本次执行归档了哪些文件，带大小与 sha256。"""
    out = Path(out)
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out).as_posix()
        if rel in {"normalized.json", "manifest.json", "archive.json"}:
            continue
        files.append({"path": rel, "size": path.stat().st_size, "sha256": _sha256(path)})
    payload = {
        "archived_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **meta,
        "files": files,
    }
    target = out / "archive.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def archive_execution(out: Path, args, raw_files: list[tuple[Path, str]]) -> None:
    """完成一次转换的归档：原始结果 + 服务端启动文件 + archive.json。"""
    raw: list[str] = []
    if args.archive_raw:
        for src, rel in raw_files:
            if Path(src).is_file():
                raw.append(_archive_copy(out, src, rel))
    meta: dict = {"execution": args.execution or None, "raw": raw}
    meta.update(archive_sources(out, args))
    write_archive_index(out, meta)


def common_parser(suite_default: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", required=True, help="输出目录（生成 normalized.json / manifest.json）")
    p.add_argument("--model", required=True, help="served model name，如 cltech/glm53")
    p.add_argument("--alias", action="append", default=[], help="模型别名，可重复")
    p.add_argument("--config-name", required=True)
    p.add_argument("--param", action="append", default=[], help="配置参数 K=V，可重复；值按 JSON 解析")
    p.add_argument("--image", default="", help="容器镜像；会归一化为 name:tag 后写入 params.image")
    p.add_argument("--node", required=True, help="节点 IP/hostname")
    p.add_argument("--worker-label", default="")
    p.add_argument("--suite", default=suite_default)
    p.add_argument("--scenario-id", type=int, default=None, help="挂载报告/产物到该场景")
    p.add_argument("--started-at", default=None)
    p.add_argument("--ended-at", default=None)
    p.add_argument("--docker-cmd", default="", help="服务端 docker run 命令")
    p.add_argument("--serve-cmd", default="", help="服务端 vllm serve 命令")
    p.add_argument("--env", action="append", default=[], help="服务端环境变量 K=V，可重复")
    p.add_argument("--bench-cmd", action="append", default=[],
                   help="压测复现命令，'title::cmd' 或只给 cmd，可重复；写入 config.deploy.extra_cmds")
    p.add_argument("--source", default="", help="来源说明，如 profile-service execution id")
    p.add_argument("--notes", default="")
    p.add_argument("--repro-cmd", default="",
                   help="run 级复现命令；导入后展示在每个 case 的复现命令中")
    p.add_argument("--case-repro", action="append", default=[],
                   help="case 级复现命令 KEY=CMD，可重复；优先于 --repro-cmd 与自动生成")
    p.add_argument("--case-console-log", action="append", default=[],
                   help="case 级 console.log KEY=PATH，可重复；文件会复制进输出目录并在 run 详情页按 case 展示")
    p.add_argument("--copy", action="append", default=[], help="额外复制进输出目录的文件/目录")
    p.add_argument("--execution", default="", help="本次压测执行 id，写进 archive.json 便于回溯")
    p.add_argument("--deploy-execution", action="append", default=[],
                   help="部署执行 id，可重复；自动从 profile-service 拉取容器/vllm/ray 启动命令归档到 <out>/server-start/")
    p.add_argument("--service-url", action="append", default=[],
                   help="profile-service 地址，可重复（多机 Ray 时传对端节点）；默认 $PROFILE_SERVICE_URL")
    p.add_argument("--deploy-dir", action="append", default=[],
                   help="已下载的部署日志目录，可重复；整目录归档到 <out>/server-start/<目录名>/")
    p.add_argument("--archive-raw", dest="archive_raw", action="store_true", default=True,
                   help="归档本次执行的原始用例结果（默认开启）")
    p.add_argument("--no-archive-raw", dest="archive_raw", action="store_false",
                   help="不归档原始用例结果，只上传指标")
    return p


def write_payload(args, cases: list[dict], meta: dict) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = parse_pairs(args.param, "--param")
    image = normalize_image(args.image or image_from_docker_cmd(args.docker_cmd))
    if image:
        params.setdefault("image", image)
    deploy: dict = {}
    if args.docker_cmd:
        deploy["docker_cmd"] = args.docker_cmd
    if args.serve_cmd:
        deploy["serve_cmd"] = args.serve_cmd
    env = parse_pairs(args.env, "--env")
    if env:
        deploy["env"] = env
    if args.source:
        deploy["source"] = args.source
    bench_cmds = []
    for item in args.bench_cmd or []:
        title, sep, cmd = item.partition("::")
        if not sep:
            title, cmd = "", item
        entry = {"cmd": cmd.strip()}
        if title.strip():
            entry["title"] = title.strip()
        if entry["cmd"]:
            bench_cmds.append(entry)
    if bench_cmds:
        deploy["extra_cmds"] = bench_cmds

    config = {"name": args.config_name, "params": params}
    if deploy:
        config["deploy"] = deploy
    case_repro = parse_raw_pairs(args.case_repro, "--case-repro")
    case_console = parse_raw_pairs(args.case_console_log, "--case-console-log")
    case_decls: dict[str, dict] = {}
    for case in cases:
        case_key = case.get("case_key")
        repro = case_repro.get(case_key)
        if repro:
            case["repro_cmd"] = repro
            case_decls.setdefault(case_key, {})["repro_cmd"] = repro
        console_src = case_console.get(case_key)
        if console_src:
            src = Path(console_src)
            if not src.is_file():
                die(f"--case-console-log 文件不存在: {console_src}")
            safe_key = str(case_key or "case").replace("/", "_").replace("\\", "_")
            dest = out / "console-logs" / safe_key / "console.log"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            rel = str(dest.relative_to(out))
            case["console_log"] = rel
            case_decls.setdefault(case_key, {})["console_log"] = rel
    normalized = {
        "model": {"name": args.model, "aliases": args.alias},
        "config": {"name": args.config_name, "params": params},
        "suite_key": args.suite,
        "node_label": args.node,
        "worker_label": args.worker_label or None,
        "backend": "generic-json",
        "valid": True,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "repro_cmd": args.repro_cmd or None,
        "cases": cases,
        "meta": meta,
    }
    manifest = {
        "model": {"name": args.model, "aliases": args.alias},
        "config": dict(config),
        "suite_key": args.suite,
        "backend": "generic-json",
        "node": args.node,
        "worker_label": args.worker_label or None,
        "started_at": args.started_at,
        "ended_at": args.ended_at,
    }
    if args.notes:
        manifest["config"]["notes"] = args.notes
    if args.repro_cmd:
        manifest["repro_cmd"] = args.repro_cmd
    if case_decls:
        manifest["cases"] = case_decls
    if args.scenario_id is not None:
        manifest["scenario_id"] = args.scenario_id
    (out / "normalized.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for src in args.copy:
        path = Path(src)
        if not path.exists():
            die(f"--copy 路径不存在: {src}")
        dest = out / path.name
        if path.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)
    print(out / "normalized.json")
    print(out / "manifest.json")
    return out


def pick(mapping: dict, names: list[str]):
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for name in names:
        if name.lower() not in lower or lower[name.lower()] is None:
            continue
        value = lower[name.lower()]
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            try:
                return float(text) if any(ch in text for ch in ".eE") else int(text)
            except ValueError:
                return value
        return value
    return None


def flatten_numbers(obj, out: dict) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out.setdefault(key, value)
            elif isinstance(value, (dict, list)):
                flatten_numbers(value, out)
    elif isinstance(obj, list):
        for item in obj:
            flatten_numbers(item, out)


def clean_metrics(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if v is not None}


# ---------------------------------------------------------------- vllm-bench

def cmd_vllm_bench(args) -> None:
    data = load_json(args.result)
    items = data if isinstance(data, list) else [data]
    if not items or not isinstance(items[0], dict):
        die("vllm-bench 结果为空或格式不符")
    case_params = parse_pairs(args.case_param, "--case-param")
    cases = []
    for index, item in enumerate(items, start=1):
        case_key = args.case_key
        if not case_key:
            case_key = item.get("case_key") or (Path(args.result).stem if len(items) == 1
                                                else f"{Path(args.result).stem}-{index}")
        elif len(items) > 1:
            case_key = f"{case_key}-{index}"
        metrics = clean_metrics({
            "e2e_mean_s": item.get("mean_e2el_ms") / 1000 if item.get("mean_e2el_ms") is not None else None,
            "e2e_p99_s": item.get("p99_e2el_ms") / 1000 if item.get("p99_e2el_ms") is not None else None,
            "ttft_mean_s": item.get("mean_ttft_ms") / 1000 if item.get("mean_ttft_ms") is not None else None,
            "ttft_p99_s": item.get("p99_ttft_ms") / 1000 if item.get("p99_ttft_ms") is not None else None,
            "tpot_mean_ms": item.get("mean_tpot_ms"),
            "tpot_p99_ms": item.get("p99_tpot_ms"),
            "itl_mean_ms": item.get("mean_itl_ms"),
            "itl_p99_ms": item.get("p99_itl_ms"),
            "out_tps": item.get("output_throughput"),
        })
        if not metrics:
            die("未识别到 vLLM benchmark_serving 指标（确认使用 --save-result 输出）")
        cases.append({
            "case_key": case_key,
            "metrics": metrics,
            "n_measured": item.get("completed"),
            "err_count": item.get("failed"),
            "in_tok": item.get("total_input_tokens"),
            "out_tok": item.get("total_output_tokens"),
            "params": case_params,
            "description": args.description or "",
            "meta": {k: item.get(k) for k in ("max_concurrency", "duration", "request_rate", "burstiness")
                     if item.get(k) is not None},
        })
    write_payload(args, cases, {"source": args.source or "vllm-bench", "source_file": str(args.result)})
    archive_execution(Path(args.out), args, [
        (Path(args.result), str(Path(RESULT_DIR) / Path(args.result).name)),
    ])


# ----------------------------------------------------------------- evalscope

def cmd_evalscope(args) -> None:
    data = load_json(args.summary)
    raw = {}
    metrics_field = data.get("metrics") if isinstance(data, dict) else None
    if isinstance(metrics_field, dict):
        raw.update({k: v for k, v in metrics_field.items() if isinstance(v, (int, float))})
    else:
        flatten_numbers(data, raw)
    if not raw:
        die("未从 evalscope summary 中找到数值指标")
    def seconds(value):
        if value is None:
            return None
        return value / 1000.0 if args.time_unit == "ms" else value
    def millis(value):
        if value is None:
            return None
        return value if args.time_unit == "ms" else value * 1000.0
    n = pick(raw, ["TotalRequests", "num_requests", "completed", "request_count", "Number of requests"])
    metrics = clean_metrics({
        "e2e_mean_s": seconds(pick(raw, ["AverageLatency", "AverageE2E", "mean_e2e", "e2e_mean",
                                         "AverageE2ELatency"])),
        "e2e_p99_s": seconds(pick(raw, ["P99Latency", "P99E2E", "e2e_p99"])),
        "ttft_mean_s": seconds(pick(raw, ["AverageTTFT", "mean_ttft", "ttft_mean", "avg_ttft"])),
        "ttft_p99_s": seconds(pick(raw, ["P99TTFT", "p99_ttft", "ttft_p99"])),
        "tpot_mean_ms": millis(pick(raw, ["AverageTPOT", "mean_tpot", "tpot_mean"])),
        "tpot_p99_ms": millis(pick(raw, ["P99TPOT", "p99_tpot"])),
        "itl_mean_ms": millis(pick(raw, ["AverageITL", "mean_itl", "itl_mean"])),
        "itl_p99_ms": millis(pick(raw, ["P99ITL", "p99_itl"])),
        "out_tps": pick(raw, ["AverageTPS", "OutputTokenThroughput", "output_throughput",
                              "OutputTPS", "AverageOutputTPS"]),
    })
    if not metrics:
        die("evalscope 指标映射为空。可用字段: " + ", ".join(sorted(raw)[:30]))
    case_key = args.case_key or (data.get("dataset") if isinstance(data, dict) else None) \
        or (data.get("benchmark") if isinstance(data, dict) else None) or Path(args.summary).stem
    cases = [{
        "case_key": case_key,
        "metrics": metrics,
        "n_measured": n,
        "err_count": pick(raw, ["FailedRequests", "failed", "errors"]),
        "in_tok": pick(raw, ["TotalInputTokens", "total_input_tokens", "InputTokens"]),
        "out_tok": pick(raw, ["TotalOutputTokens", "total_output_tokens", "OutputTokens"]),
        "params": parse_pairs(args.case_param, "--case-param"),
        "description": args.description or "",
        "meta": {},
    }]
    write_payload(args, cases, {"source": args.source or "evalscope", "source_file": str(args.summary)})
    archive_execution(Path(args.out), args, [
        (Path(args.summary), str(Path(RESULT_DIR) / Path(args.summary).name)),
    ])


# --------------------------------------------------------- benchmark-suite

def _bs_case_desc(case: dict) -> str:
    label = {"cold": "全冷输入", "suffix": "前缀复用+冷后缀", "full": "全量前缀缓存"}.get(
        case.get("cache"), case.get("cache", ""))
    return (f"{label} · 输入 {case.get('input_tokens')} tok · 输出 {case.get('output_tokens')} tok"
            f" · 并发 {case.get('concurrency')} · {case.get('requests')} 请求")


CASE_LOG_FILES = (("stdout_log", "stdout.log"), ("stderr_log", "stderr.log"))
# 归档每个用例的原始记录：日志（stdout/stderr）优先，其余为便于回溯的扩展名。
_CASE_ARCHIVE_NAMES = ("stdout.log", "stderr.log", "console.log", "case.json", "command.json")
_SUMMARY_ARCHIVE_NAMES = ("summary.json", "summary.csv", "config.json", "requests.jsonl", "manifest.json")


def _case_dirs_by_key(summary_path: str) -> dict[str, Path]:
    """benchmark-suite 执行目录中的 case-*/case.json 映射到 case_key。"""
    root = Path(summary_path).parent
    out: dict[str, Path] = {}
    for case_dir in sorted(root.glob("case-*")):
        meta_path = case_dir / "case.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        case_key = meta.get("id")
        if isinstance(case_key, str) and case_key:
            out[case_key] = case_dir
    return out


def _archive_benchmark_suite(out: Path, summary_path: str) -> list[tuple[Path, str]]:
    """测试端原始产物：摘要/manifest 与每个用例的 stdout.log、stderr.log 等。

    失败用例没有入库，但它们的 stdout/stderr 也一并归档；这些文件不会被适配器
    消费，导入时按 run 产物落盘，可在 run 详情页下载。
    """
    root = Path(summary_path).parent
    sources: list[tuple[Path, str]] = []
    for name in _SUMMARY_ARCHIVE_NAMES:
        path = root / name
        if path.is_file():
            sources.append((path, str(Path(RAW_DIR) / name)))
    for case_dir in sorted(root.glob("case-*")):
        for name in _CASE_ARCHIVE_NAMES:
            path = case_dir / name
            if path.is_file():
                sources.append((path, str(Path(RAW_DIR) / case_dir.name / name)))
    return sources


def cmd_benchmark_suite(args) -> None:
    summary = load_json(args.summary)
    config = load_json(args.config)
    definitions = {c["id"]: c for c in config.get("cases", []) if "id" in c}
    case_dirs = _case_dirs_by_key(args.summary)
    out_root = Path(args.out)
    requests_by_case: dict = {}
    if args.requests:
        try:
            for line in Path(args.requests).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("warmup") or rec.get("error"):
                    continue
                requests_by_case.setdefault(rec.get("case_id"), []).append(rec)
        except (OSError, json.JSONDecodeError) as exc:
            die(f"无法读取 requests.jsonl: {exc}")
    cases = []
    for item in summary.get("cases", []):
        if item.get("status") != "success" and not args.include_failed:
            continue
        case_id = item.get("id")
        base = definitions.get(case_id, {})
        records = requests_by_case.get(case_id, [])
        metrics = clean_metrics({
            "e2e_mean_s": item.get("e2e_mean_seconds"),
            "e2e_p99_s": item.get("e2e_p99_seconds"),
            "ttft_mean_s": item.get("ttft_mean_seconds"),
            "ttft_p99_s": item.get("ttft_p99_seconds"),
            "tpot_mean_ms": item["tpot_mean_seconds"] * 1000 if item.get("tpot_mean_seconds") is not None else None,
            "tpot_p99_ms": item["tpot_p99_seconds"] * 1000 if item.get("tpot_p99_seconds") is not None else None,
            "out_tps": item.get("output_tokens_per_second"),
        })
        if records:
            prompt = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in records)
            cached = sum(int(r.get("cached_tokens") or 0) for r in records)
            if prompt:
                metrics["cache_hit_rate"] = cached / prompt
        measured = int(item.get("requested_requests") or base.get("requests") or 0) \
            - int(item.get("failed_requests") or 0)
        if records:
            in_total = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in records)
            out_total = sum(int(r.get("completion_tokens") or 0) for r in records)
        else:
            in_total = int(base.get("input_tokens") or 0) * max(measured, 0)
            out_total = int(base.get("output_tokens") or 0) * max(measured, 0)
        case = {
            "case_key": case_id,
            "metrics": metrics,
            "n_measured": measured,
            "err_count": int(item.get("failed_requests") or 0),
            "in_tok": in_total or None,
            "out_tok": out_total or None,
            "params": {k: v for k, v in base.items() if k != "id"},
            "description": _bs_case_desc(base or item),
            "meta": {"cache_validation": item.get("cache_validation"), "backend": item.get("backend")},
        }
        case_dir = case_dirs.get(case_id)
        if case_dir:
            safe_key = str(case_id).replace("/", "_").replace("\\", "_")
            for field, filename in CASE_LOG_FILES:
                src = case_dir / filename
                if not src.is_file():
                    continue
                dest = out_root / "case-logs" / safe_key / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                case[field] = str(dest.relative_to(out_root))
        cases.append(case)
    if not cases:
        die("benchmark-suite summary 没有可导入的 success case")
    write_payload(args, cases, {
        "source": args.source or "profile-service benchmark-suite",
        "source_file": str(args.summary),
        "benchmark_client_config": config,
    })
    sources = _archive_benchmark_suite(out_root, args.summary)
    root = Path(args.summary).parent
    for extra in (args.config, args.requests):
        if not extra:
            continue
        path = Path(extra)
        if path.is_file() and path.parent != root:
            sources.append((path, str(Path(RAW_DIR) / path.name)))
    archive_execution(out_root, args, sources)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="perf-viewer 导入数据转换工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("vllm-bench", parents=[common_parser("vllm-bench")],
                       help="vLLM benchmark_serving.py --save-result JSON")
    p.add_argument("--result", required=True)
    p.add_argument("--case-key", default=None)
    p.add_argument("--case-param", action="append", default=[])
    p.add_argument("--description", default="")
    p.set_defaults(func=cmd_vllm_bench)

    p = sub.add_parser("evalscope", parents=[common_parser("evalscope")],
                       help="EvalScope benchmark_summary.json")
    p.add_argument("--summary", required=True)
    p.add_argument("--case-key", default=None)
    p.add_argument("--case-param", action="append", default=[])
    p.add_argument("--description", default="")
    p.add_argument("--time-unit", choices=["s", "ms"], default="s",
                   help="summary 时间字段单位；默认 s（TTFT/E2E 秒，TPOT/ITL 转 ms）")
    p.set_defaults(func=cmd_evalscope)

    p = sub.add_parser("benchmark-suite", parents=[common_parser("benchmark-suite")],
                       help="profile-service benchmark-suite 执行目录")
    p.add_argument("--summary", required=True, help="benchmark-suite/summary.json")
    p.add_argument("--config", required=True, help="benchmark-suite/config.json")
    p.add_argument("--requests", default="", help="benchmark-suite/requests.jsonl（可选，用于 cache_hit_rate）")
    p.add_argument("--include-failed", action="store_true", help="也导入失败 case（无指标的不导入）")
    p.set_defaults(func=cmd_benchmark_suite)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
