#!/usr/bin/env python3
"""perf-viewer 数据导入辅助：把常见压测产物转换为 generic-json 可导入的目录。

用法:
  import_kit.py vllm-bench --result result.json --out ./import/run1 --model ... --config-name ... --node ...
  import_kit.py evalscope --summary benchmark_summary.json --out ./import/run1 ...
  import_kit.py benchmark-suite --summary summary.json --config config.json --out ./import/run1 ...

生成 <out>/normalized.json 与 <out>/manifest.json，然后:
  cd /home/cangying/perf-viewer
  .venv/bin/python -m app.cli import <out> --manifest <out>/manifest.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
    p.add_argument("--source", default="", help="来源说明，如 profile-service execution id")
    p.add_argument("--notes", default="")
    p.add_argument("--repro-cmd", default="",
                   help="run 级复现命令；导入后展示在每个 case 的复现命令中")
    p.add_argument("--case-repro", action="append", default=[],
                   help="case 级复现命令 KEY=CMD，可重复；优先于 --repro-cmd 与自动生成")
    p.add_argument("--copy", action="append", default=[], help="额外复制进输出目录的文件/目录")
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

    config = {"name": args.config_name, "params": params}
    if deploy:
        config["deploy"] = deploy
    case_repro = parse_raw_pairs(args.case_repro, "--case-repro")
    for case in cases:
        repro = case_repro.get(case.get("case_key"))
        if repro:
            case["repro_cmd"] = repro
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
    if case_repro:
        manifest["cases"] = {key: {"repro_cmd": value} for key, value in case_repro.items()}
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


# --------------------------------------------------------- benchmark-suite

def _bs_case_desc(case: dict) -> str:
    label = {"cold": "全冷输入", "suffix": "前缀复用+冷后缀", "full": "全量前缀缓存"}.get(
        case.get("cache"), case.get("cache", ""))
    return (f"{label} · 输入 {case.get('input_tokens')} tok · 输出 {case.get('output_tokens')} tok"
            f" · 并发 {case.get('concurrency')} · {case.get('requests')} 请求")


def cmd_benchmark_suite(args) -> None:
    summary = load_json(args.summary)
    config = load_json(args.config)
    definitions = {c["id"]: c for c in config.get("cases", []) if "id" in c}
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
        cases.append({
            "case_key": case_id,
            "metrics": metrics,
            "n_measured": measured,
            "err_count": int(item.get("failed_requests") or 0),
            "in_tok": in_total or None,
            "out_tok": out_total or None,
            "params": {k: v for k, v in base.items() if k != "id"},
            "description": _bs_case_desc(base or item),
            "meta": {"cache_validation": item.get("cache_validation"), "backend": item.get("backend")},
        })
    if not cases:
        die("benchmark-suite summary 没有可导入的 success case")
    write_payload(args, cases, {
        "source": args.source or "profile-service benchmark-suite",
        "source_file": str(args.summary),
        "benchmark_client_config": config,
    })


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
