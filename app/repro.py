"""复现命令：为每个数据点（配置 × suite × case）给出可复制执行的复现方案。

优先使用导入时记录的显式命令（manifest.repro_cmd / manifest.cases.<key>.repro_cmd），
否则按 backend、case 参数和服务端部署信息生成。
"""

import json
import shlex

from .security import redact_cmd


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _case_params(case: dict) -> dict:
    params = _as_dict(case.get("params"))
    extra = _as_dict(case.get("case_params"))
    params.update(extra)
    return params


def _client_config(manifest: dict) -> dict:
    return _as_dict(manifest.get("benchmark_client_config"))


def _client_case(manifest: dict, case_key: str) -> dict:
    for item in _client_config(manifest).get("cases") or []:
        if not isinstance(item, dict):
            continue
        if item.get("id") == case_key or item.get("case_key") == case_key:
            return item
    return {}


def _explicit(manifest: dict, case_key: str) -> str | None:
    candidates = []
    case_repro = _as_dict(manifest.get("case_repro"))
    if case_repro.get(case_key):
        candidates.append(case_repro[case_key])
    imported = _as_dict(manifest.get("import_manifest"))
    imported_case = _as_dict((_as_dict(imported.get("cases")).get(case_key)))
    if imported_case.get("repro_cmd"):
        candidates.append(imported_case["repro_cmd"])
    for key in ("repro_cmd", "repro"):
        value = manifest.get(key)
        if value:
            candidates.append(value)
        value = imported.get(key)
        if value:
            candidates.append(value)
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return redact_cmd(value.strip())
    return None


def _int_value(params: dict, *names):
    for name in names:
        value = params.get(name)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _base_url(manifest: dict, node_label: str) -> str:
    client = _client_config(manifest)
    url = client.get("base_url") or manifest.get("base_url")
    if url:
        return str(url).rstrip("/")
    if node_label:
        return f"http://{node_label}:8000/v1"
    return "http://127.0.0.1:8000/v1"


def _bench_backend(manifest: dict, run_backend: str) -> str:
    client = _client_config(manifest)
    backend = str(manifest.get("benchmark_backend") or client.get("backend")
                  or run_backend or "").lower()
    if backend in ("", "generic-json"):
        source = str(manifest.get("source") or "").lower()
        for name in ("evalscope", "vllm-bench", "guidellm", "benchmark-suite",
                     "probe-json", "node-csv"):
            if name in source:
                return name
    return backend


def _quote(value) -> str:
    return shlex.quote(str(value))


def _bench_command(backend: str, model: str, base_url: str, params: dict,
                   case_key: str, manifest: dict) -> str:
    inp = _int_value(params, "input_tokens", "prompt_tokens", "random_input_len")
    out = _int_value(params, "output_tokens", "max_tokens", "random_output_len")
    conc = _int_value(params, "concurrency", "max_concurrency")
    requests = _int_value(params, "requests", "num_prompts", "num_requests")

    if backend in ("evalscope",):
        # 与 benchmark_backends.build_command 保持一致的真实 evalscope CLI。
        url = base_url.rstrip("/")
        if not url.endswith("/completions"):
            url += "/completions"
        seed = _int_value(params, "seed")
        tokenizer = (manifest.get("tokenizer_path")
                     or _client_config(manifest).get("tokenizer_path") or "")
        parts = ["evalscope perf",
                 f"--model {_quote(model)}",
                 f"--url {_quote(url)}",
                 "--api openai",
                 "--dataset random"]
        if requests:
            parts.append(f"--number {requests}")
        if inp:
            parts.append(f"--min-prompt-length {inp} --max-prompt-length {inp}")
        if out:
            parts.append(f"--min-tokens {out} --max-tokens {out}")
        if conc:
            parts.append(f"--parallel {conc}")
        if seed is not None:
            parts.append(f"--seed {seed}")
        if tokenizer:
            parts.append(f"--tokenizer-path {_quote(tokenizer)}")
        parts.append("--stream --extra-args '{\"ignore_eos\":true}'")
        parts.append(f"--outputs-dir {_quote('./repro/' + case_key)} --no-test-connection --tokenize-prompt")
        return " \\\n  ".join(parts)

    if backend in ("vllm-bench", "vllm_bench", "benchmark-serving", "benchmark_serving"):
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        parts = ["python3 benchmark_serving.py",
                 "--backend openai-chat",
                 f"--base-url {_quote(root)}",
                 f"--model {_quote(model)}",
                 "--dataset-name random"]
        if inp:
            parts.append(f"--random-input-len {inp}")
        if out:
            parts.append(f"--random-output-len {out}")
        if requests:
            parts.append(f"--num-prompts {requests}")
        if conc:
            parts.append(f"--max-concurrency {conc}")
        parts.append("--save-result")
        return " \\\n  ".join(parts)

    if backend in ("guidellm", "guidellm-suite"):
        parts = ["guidellm benchmark run",
                 f"--target {_quote(base_url)}",
                 f"--model {_quote(model)}"]
        if requests:
            parts.append(f"--max-requests {requests}")
        if inp:
            parts.append(f"--data-prompt-tokens {inp}")
        if out:
            parts.append(f"--data-output-tokens {out}")
        if conc:
            parts.append(f"--rate-type concurrent --rate {conc}")
        return " \\\n  ".join(parts)

    # 未知 backend：不再伪造压测脚本模板，交由 resolve_repro 提示缺失
    return ""


def _comment(value) -> str:
    """注释行内只允许单行文本，避免 manifest 字段注入可执行行。"""
    return " ".join(str(value).split())


def resolve_repro(model_name: str, config: dict, run: dict, case: dict,
                  suite_key: str = "") -> str:
    """返回该 配置×suite×case 数据点的客户端压测命令。

    config: config_set 行；run: run 行（含 backend/manifest/node_id）；
    case: test_case 行（含 case_key/params）。
    """
    manifest = _as_dict(run.get("manifest"))
    case_key = case.get("case_key") or ""
    explicit = _explicit(manifest, case_key)
    if explicit:
        return explicit

    suite = _comment(suite_key or "")
    node_label = _comment((run.get("node") or {}).get("label")
                          or run.get("node_label") or "")
    config_name = _comment(config.get("name") or "")
    backend = _bench_backend(manifest, run.get("backend") or "")
    base_url = _base_url(manifest, node_label)
    client = _client_config(manifest)
    model = client.get("model") or model_name or ""

    params = _case_params(case)
    client_case = _client_case(manifest, case_key)
    merged = dict(client_case)
    merged.update(params)

    title = f"# 复现数据：{suite + '/' if suite else ''}{_comment(case_key)}"
    if config_name:
        title += f"（配置 {config_name}"
        title += f" · 节点 {node_label}）" if node_label else "）"

    lines = [title]
    bench = _bench_command(backend, model, base_url, merged, case_key, manifest)
    if bench:
        lines.append(bench)
    else:
        lines.append("# 未记录压测客户端命令：请在导入 manifest 的 cases."
                     f"{_comment(case_key)}.repro_cmd 或顶层 repro_cmd 补充")
    if merged:
        lines.append("# case 参数: " + json.dumps(merged, ensure_ascii=False, sort_keys=True))
    return redact_cmd("\n".join(lines))
