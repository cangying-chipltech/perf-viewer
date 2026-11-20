"""guidellm-suite：meta.json + guidellm.json + server_metrics.json 的 case 目录套件。

口径（与 12-case 报告一致）：
- 仅统计测量窗口（successful）；warmup 已由 GuideLLM profile 排除。
- valid = request_totals.errored==0 且 incomplete==0 且 total>0。
- out_tps = 输出 token 总量 / E2E 耗时之和（聚合口径，非逐请求均值）。
- cache_hit_rate 仅取自 server_metrics.json.prefix_cache_hit_rate，缺失留空。
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ..normalize import (NormalizedArtifact, NormalizedCase, NormalizedRun, empty_metrics,
                         kind_of_suffix)
from .base import AdapterError, manifest_config, manifest_model, manifest_node


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _stat(metrics: dict, name: str, stat: str, status: str = "successful"):
    entry = metrics.get(name) or {}
    for scope in (status, "all"):
        block = entry.get(scope) or {}
        if stat == "p99":
            percentiles = block.get("percentiles") or {}
            for key in ("p99", "p990", "99"):
                if percentiles.get(key) is not None:
                    return percentiles[key]
            continue
        if block.get(stat) is not None:
            return block[stat]
    return None


def _case_dirs(root: Path) -> list[Path]:
    dirs = []
    for meta_path in sorted(root.rglob("meta.json")):
        case_dir = meta_path.parent
        if (case_dir / "guidellm.json").exists():
            dirs.append(case_dir)
    return dirs


class GuidellmSuiteAdapter:
    name = "guidellm-suite"

    def detect(self, files: list[Path], manifest: dict) -> bool:
        names = {p.name for p in files}
        return "guidellm.json" in names and "meta.json" in names

    def parse(self, root: Path, manifest: dict) -> NormalizedRun:
        case_dirs = _case_dirs(root)
        if not case_dirs:
            raise AdapterError("未找到含 meta.json + guidellm.json 的 case 目录")
        model_name, aliases = manifest_model(manifest)
        config_name, config_params = manifest_config(manifest)
        suite_key = manifest.get("suite_key") or ""
        if not suite_key:
            raise AdapterError("manifest 缺少 suite_key")

        cases: list[NormalizedCase] = []
        artifacts: list[NormalizedArtifact] = []
        started_candidates: list[str] = []
        ended_candidates: list[str] = []
        worker = manifest.get("worker_label")
        node = manifest_node(manifest)
        all_valid = True

        for case_dir in case_dirs:
            meta = _load_json(case_dir / "meta.json") or {}
            guidellm = _load_json(case_dir / "guidellm.json") or {}
            benchmarks = guidellm.get("benchmarks") or [{}]
            metrics = benchmarks[0].get("metrics") or {}
            totals = metrics.get("request_totals") or {}
            total = totals.get("total") or 0
            errored = totals.get("errored") or 0
            incomplete = totals.get("incomplete") or 0
            successful = totals.get("successful")
            case_valid = bool(total > 0 and errored == 0 and incomplete == 0)
            all_valid = all_valid and case_valid

            norm = empty_metrics()
            e2e_mean = _stat(metrics, "request_latency", "mean")
            e2e_p99 = _stat(metrics, "request_latency", "p99")
            norm["e2e_mean_s"] = e2e_mean
            norm["e2e_p99_s"] = e2e_p99
            ttft_mean = _stat(metrics, "time_to_first_token_ms", "mean")
            ttft_p99 = _stat(metrics, "time_to_first_token_ms", "p99")
            norm["ttft_mean_s"] = ttft_mean / 1000.0 if ttft_mean is not None else None
            norm["ttft_p99_s"] = ttft_p99 / 1000.0 if ttft_p99 is not None else None
            norm["tpot_mean_ms"] = _stat(metrics, "time_per_output_token_ms", "mean")
            norm["tpot_p99_ms"] = _stat(metrics, "time_per_output_token_ms", "p99")
            norm["itl_mean_ms"] = _stat(metrics, "inter_token_latency_ms", "mean")
            norm["itl_p99_ms"] = _stat(metrics, "inter_token_latency_ms", "p99")
            out_total = _stat(metrics, "output_tokens_count", "total")
            latency_total = _stat(metrics, "request_latency", "total")
            if out_total is not None and latency_total:
                norm["out_tps"] = out_total / latency_total
            in_total = _stat(metrics, "prompt_tokens_count", "total")

            server_metrics = _load_json(case_dir / "server_metrics.json") or {}
            norm["cache_hit_rate"] = server_metrics.get("prefix_cache_hit_rate")

            case_key = meta.get("case_id") or case_dir.name
            params = {k: meta[k] for k in (
                "group", "concurrency", "requests", "prefix_tokens", "suffix_tokens",
                "input_tokens", "output_tokens", "warmup", "seed",
            ) if k in meta}
            cases.append(NormalizedCase(
                case_key=case_key,
                metrics=norm,
                n_measured=int(successful) if successful is not None else None,
                err_count=int(errored) + int(incomplete),
                in_tok=int(in_total) if in_total is not None else None,
                out_tok=int(out_total) if out_total is not None else None,
                params=params,
                description=meta.get("description", ""),
                meta={"case_valid": case_valid, "request_totals": totals},
            ))

            if meta.get("started_at"):
                started_candidates.append(meta["started_at"])
            timing_path = case_dir / "timing.csv"
            if timing_path.exists():
                try:
                    with timing_path.open(encoding="utf-8", newline="") as fh:
                        timing = next(csv.DictReader(fh), {})
                    end_epoch = float(timing.get("end_epoch", 0))
                    if end_epoch:
                        ended_candidates.append(
                            datetime.fromtimestamp(end_epoch, tz=timezone.utc).isoformat()
                        )
                except (OSError, ValueError, StopIteration):
                    pass
            worker = worker or meta.get("worker_label")
            node = node or meta.get("node") or ""

            for file in sorted(case_dir.iterdir()):
                if file.is_file() and not file.name.startswith("."):
                    kind = kind_of_suffix(file.suffix)
                    artifacts.append(NormalizedArtifact(
                        kind=kind, path=str(file.relative_to(root)),
                        title=f"{case_key}/{file.name}",
                    ))

        if not node:
            raise AdapterError("无法推导节点：请在 manifest 声明 node（或 meta.json 提供 node）")

        run = NormalizedRun(
            model_name=model_name,
            model_aliases=aliases,
            config_name=config_name,
            config_params=config_params,
            node_label=node,
            suite_key=suite_key,
            backend=self.name,
            cases=cases,
            valid=all_valid,
            started_at=min(started_candidates) if started_candidates else manifest.get("started_at"),
            ended_at=max(ended_candidates) if ended_candidates else manifest.get("ended_at"),
            worker_label=worker,
            artifacts=artifacts,
            meta={"source": "guidellm-suite", "case_count": len(cases)},
        )
        run.clean_metrics()
        return run