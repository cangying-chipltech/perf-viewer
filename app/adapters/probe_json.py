"""probe-json：chat_probe / mixed_impact 输出 JSON。

chat_probe 特征：ok/complete/latency_s/ttft_s；n=1，p99=mean。
output_tok_s 为 0.0（1-token 输出）是合法值，不当作缺失。
mixed_impact 特征：baseline/long/probe_<delay> 键。
"""

import json
from pathlib import Path
from urllib.parse import urlparse

from ..normalize import NormalizedArtifact, NormalizedCase, NormalizedRun, empty_metrics
from .base import AdapterError, manifest_config, manifest_model, manifest_node


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_chat_probe(data) -> bool:
    return isinstance(data, dict) and "ttft_s" in data and "latency_s" in data


def _is_mixed_impact(data) -> bool:
    return isinstance(data, dict) and "long" in data and (
        "baseline" in data or any(k.startswith("probe_") for k in data)
    )


def _node_from_base_url(base_url: str) -> str:
    try:
        host = urlparse(base_url).hostname
        return host or ""
    except ValueError:
        return ""


class ProbeJsonAdapter:
    name = "probe-json"

    def _probe_files(self, files: list[Path]) -> list[tuple[Path, dict, str]]:
        out = []
        for path in files:
            if path.suffix.lower() != ".json":
                continue
            data = _load(path)
            if _is_chat_probe(data):
                out.append((path, data, "chat_probe"))
            elif _is_mixed_impact(data):
                out.append((path, data, "mixed_impact"))
        return out

    def detect(self, files: list[Path], manifest: dict) -> bool:
        return bool(self._probe_files(files))

    def parse(self, root: Path, manifest: dict) -> list[NormalizedRun]:
        probes = self._probe_files([p for p in sorted(root.rglob("*")) if p.is_file()])
        if not probes:
            raise AdapterError("未找到探针 JSON")
        model_name, aliases = manifest_model(manifest)
        config_name, config_params = manifest_config(manifest)
        suite_key = manifest.get("suite_key") or ""
        if not suite_key:
            raise AdapterError("manifest 缺少 suite_key")

        runs = []
        for path, data, flavor in probes:
            if flavor == "chat_probe":
                run = self._parse_chat_probe(path, data, root, manifest, model_name, aliases,
                                             config_name, config_params, suite_key)
            else:
                run = self._parse_mixed_impact(path, data, root, manifest, model_name, aliases,
                                               config_name, config_params, suite_key)
            runs.append(run)
        return runs

    def _common(self, path, data, root, manifest, model_name, aliases,
                config_name, config_params, suite_key) -> NormalizedRun:
        node = manifest_node(manifest) or _node_from_base_url(data.get("base_url", ""))
        if not node:
            raise AdapterError(f"{path.name}: 无法推导节点，请在 manifest 声明 node")
        run = NormalizedRun(
            model_name=data.get("model") or model_name,
            model_aliases=aliases,
            config_name=config_name,
            config_params=config_params,
            node_label=node,
            suite_key=suite_key,
            backend=self.name,
            started_at=data.get("started_at") or manifest.get("started_at"),
            ended_at=data.get("ended_at") or manifest.get("ended_at"),
            worker_label=data.get("worker_label") or manifest.get("worker_label"),
            artifacts=[NormalizedArtifact(kind="json", path=str(path.relative_to(root)),
                                          title=f"探针原始结果 {path.name}")],
            meta={"source": "probe-json", "file": path.name},
        )
        return run

    def _parse_chat_probe(self, path, data, root, manifest, model_name, aliases,
                          config_name, config_params, suite_key) -> NormalizedRun:
        run = self._common(path, data, root, manifest, model_name, aliases,
                           config_name, config_params, suite_key)
        ok = bool(data.get("ok"))
        complete = bool(data.get("complete"))
        run.valid = ok and complete
        metrics = empty_metrics()
        latency = data.get("latency_s")
        ttft = data.get("ttft_s")
        metrics["e2e_mean_s"] = latency
        metrics["e2e_p99_s"] = latency
        metrics["ttft_mean_s"] = ttft
        metrics["ttft_p99_s"] = ttft
        metrics["out_tps"] = data.get("output_tok_s")
        case_key = manifest.get("case_key") or "probe"
        case = NormalizedCase(
            case_key=case_key,
            metrics=metrics,
            n_measured=1,
            err_count=0 if ok else 1,
            in_tok=data.get("prompt_tokens"),
            out_tok=data.get("completion_tokens"),
            params={
                "input_target_tokens": data.get("input_target_tokens"),
                "output_limit_tokens": data.get("output_limit_tokens"),
            },
            meta={
                "response_id": data.get("response_id"),
                "finish_reason": data.get("finish_reason"),
                "cached_tokens": data.get("cached_tokens"),
                "content_chunks": data.get("content_chunks"),
                "peak_running": data.get("peak_running"),
                "peak_waiting": data.get("peak_waiting"),
                "error": data.get("error"),
                "base_url": data.get("base_url"),
            },
        )
        run.cases = [case]
        run.meta.update({
            "response_id": data.get("response_id"),
            "finish_reason": data.get("finish_reason"),
            "cached_tokens": data.get("cached_tokens"),
            "input_target_tokens": data.get("input_target_tokens"),
            "output_limit_tokens": data.get("output_limit_tokens"),
            "prompt_tokens": data.get("prompt_tokens"),
            "completion_tokens": data.get("completion_tokens"),
            "base_url": data.get("base_url"),
        })
        run.clean_metrics()
        return run

    def _parse_mixed_impact(self, path, data, root, manifest, model_name, aliases,
                            config_name, config_params, suite_key) -> NormalizedRun:
        run = self._common(path, data, root, manifest, model_name, aliases,
                           config_name, config_params, suite_key)
        prefix = manifest.get("case_prefix") or "mixed"
        cases = []
        all_ok = True
        for key in sorted(data):
            entry = data[key]
            if not isinstance(entry, dict) or "latency" not in entry:
                continue
            ok = bool(entry.get("ok", True))
            all_ok = all_ok and ok
            usage = entry.get("usage") or {}
            metrics = empty_metrics()
            metrics["e2e_mean_s"] = entry.get("latency")
            metrics["e2e_p99_s"] = entry.get("latency")
            if entry.get("ttft_s") is not None:
                metrics["ttft_mean_s"] = entry["ttft_s"]
                metrics["ttft_p99_s"] = entry["ttft_s"]
            if key == "baseline":
                case_key = f"{prefix}_baseline"
            elif key == "long":
                case_key = f"{prefix}_long"
            else:
                case_key = f"{prefix}_{key}"
            cases.append(NormalizedCase(
                case_key=case_key,
                metrics=metrics,
                n_measured=1,
                err_count=0 if ok else 1,
                in_tok=usage.get("prompt_tokens"),
                out_tok=usage.get("completion_tokens"),
                meta={"sent_at": entry.get("sent_at"), "error": entry.get("error"),
                      "finish_reason": entry.get("finish_reason")},
            ))
        if not cases:
            raise AdapterError(f"{path.name}: mixed_impact 无可用条目")
        run.cases = cases
        run.valid = all_ok
        run.clean_metrics()
        return run