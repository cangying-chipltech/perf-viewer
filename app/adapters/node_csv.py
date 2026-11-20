"""node-csv：现有节点明细 CSV（表头 node,case,n_measured,err,e2e_mean_s,...）。

一文件多节点多 case → 每节点一个 run。
ITL 映射：itl_mean_ms ← itl_tok_mean_ms（token 级，与报告口径一致），
缺失时回退 itl_reqmean_ms；itl_p99_ms ← itl_p99_ms。
in_tok/out_tok 为测量请求总量（非单请求）。
"""

import csv
from pathlib import Path

from ..normalize import NormalizedArtifact, NormalizedCase, NormalizedRun, empty_metrics
from .base import AdapterError, manifest_config, manifest_model

REQUIRED_HEADER = ["node", "case", "n_measured", "err", "e2e_mean_s"]

_DIRECT = {
    "e2e_mean_s": "e2e_mean_s",
    "e2e_p99_s": "e2e_p99_s",
    "ttft_mean_s": "ttft_mean_s",
    "ttft_p99_s": "ttft_p99_s",
    "tpot_mean_ms": "tpot_mean_ms",
    "tpot_p99_ms": "tpot_p99_ms",
    "itl_p99_ms": "itl_p99_ms",
    "out_tps": "out_tps",
}


def _float(value):
    if value is None or value == "":
        return None
    return float(value)


def _int(value):
    if value is None or value == "":
        return None
    return int(float(value))


class NodeCsvAdapter:
    name = "node-csv"

    def _csv_files(self, files: list[Path]) -> list[Path]:
        out = []
        for path in files:
            if path.suffix.lower() != ".csv":
                continue
            try:
                with path.open(encoding="utf-8-sig", newline="") as fh:
                    header = next(csv.reader(fh), [])
            except (OSError, StopIteration):
                continue
            if header[: len(REQUIRED_HEADER)] == REQUIRED_HEADER:
                out.append(path)
        return out

    def detect(self, files: list[Path], manifest: dict) -> bool:
        return bool(self._csv_files(files))

    def parse(self, root: Path, manifest: dict) -> list[NormalizedRun]:
        csv_files = self._csv_files([p for p in files_iter(root)])
        if not csv_files:
            raise AdapterError("未找到符合表头的节点明细 CSV")
        model_name, aliases = manifest_model(manifest)
        config_name, config_params = manifest_config(manifest)
        suite_key = manifest.get("suite_key") or ""
        if not suite_key:
            raise AdapterError("manifest 缺少 suite_key")

        by_node: dict[str, list[NormalizedCase]] = {}
        sources: dict[str, Path] = {}
        for path in csv_files:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    node = (row.get("node") or "").strip()
                    case_key = (row.get("case") or "").strip()
                    if not node or not case_key:
                        continue
                    metrics = empty_metrics()
                    for src, dst in _DIRECT.items():
                        metrics[dst] = _float(row.get(src))
                    itl = _float(row.get("itl_tok_mean_ms"))
                    if itl is None:
                        itl = _float(row.get("itl_reqmean_ms"))
                    metrics["itl_mean_ms"] = itl
                    case = NormalizedCase(
                        case_key=case_key,
                        metrics=metrics,
                        n_measured=_int(row.get("n_measured")),
                        err_count=_int(row.get("err")),
                        in_tok=_int(row.get("in_tok")),
                        out_tok=_int(row.get("out_tok")),
                    )
                    by_node.setdefault(node, []).append(case)
                    sources.setdefault(node, path)

        runs = []
        for node in sorted(by_node):
            cases = by_node[node]
            run = NormalizedRun(
                model_name=model_name,
                model_aliases=aliases,
                config_name=config_name,
                config_params=config_params,
                node_label=node,
                suite_key=suite_key,
                backend=self.name,
                cases=cases,
                valid=all((c.err_count or 0) == 0 for c in cases),
                started_at=manifest.get("started_at"),
                ended_at=manifest.get("ended_at"),
                worker_label=manifest.get("worker_label"),
                artifacts=[
                    NormalizedArtifact(
                        kind="csv",
                        path=str(sources[node].relative_to(root)),
                        title="节点明细 CSV（源文件）",
                    )
                ],
                meta={"source": "node-csv", "itl_mapping": "itl_tok_mean_ms -> itl_mean_ms"},
            )
            run.clean_metrics()
            runs.append(run)
        return runs


def files_iter(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path