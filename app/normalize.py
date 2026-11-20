"""归一化 schema：metric_def 种子 + NormalizedRun 数据结构。"""

from dataclasses import dataclass, field
from typing import Any, Optional

METRIC_SEED = [
    ("e2e_mean_s", "E2E 均值", "s", "lower"),
    ("e2e_p99_s", "E2E P99", "s", "lower"),
    ("ttft_mean_s", "TTFT 均值", "s", "lower"),
    ("ttft_p99_s", "TTFT P99", "s", "lower"),
    ("tpot_mean_ms", "TPOT 均值", "ms", "lower"),
    ("tpot_p99_ms", "TPOT P99", "ms", "lower"),
    ("itl_mean_ms", "ITL 均值", "ms", "lower"),
    ("itl_p99_ms", "ITL P99", "ms", "lower"),
    ("out_tps", "输出吞吐", "tok/s", "higher"),
    ("cache_hit_rate", "前缀缓存命中率", "ratio", "higher"),
]

METRIC_KEYS = [m[0] for m in METRIC_SEED]

KIND_BY_SUFFIX = {
    ".md": "md", ".png": "png", ".jpg": "png", ".jpeg": "png", ".gif": "png", ".svg": "png",
    ".csv": "csv", ".tsv": "csv", ".log": "log", ".txt": "log", ".prom": "log", ".json": "json",
}


def kind_of_suffix(suffix: str) -> str:
    return KIND_BY_SUFFIX.get(suffix.lower(), "log")


@dataclass
class NormalizedCase:
    case_key: str
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    n_measured: Optional[int] = None
    err_count: Optional[int] = None
    in_tok: Optional[int] = None
    out_tok: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedArtifact:
    kind: str
    path: str
    title: str = ""


@dataclass
class NormalizedRun:
    model_name: str = ""
    model_aliases: list[str] = field(default_factory=list)
    config_name: str = ""
    config_params: dict[str, Any] = field(default_factory=dict)
    node_label: str = ""
    suite_key: str = ""
    backend: str = ""
    cases: list[NormalizedCase] = field(default_factory=list)
    valid: bool = True
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    worker_label: Optional[str] = None
    artifacts: list[NormalizedArtifact] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def clean_metrics(self) -> None:
        for case in self.cases:
            case.metrics = {
                k: (float(v) if v is not None else None)
                for k, v in case.metrics.items()
                if k in METRIC_KEYS
            }


def empty_metrics() -> dict[str, Optional[float]]:
    return {k: None for k in METRIC_KEYS}