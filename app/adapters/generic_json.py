"""generic-json：直接符合 NormalizedRun schema 的 JSON。

接受单个对象或对象数组，字段：
model_name/model{name,aliases}, config_name/config{name,params}, node_label/node,
suite_key, cases[{case_key,metrics,n_measured,err_count,in_tok,out_tok,params}],
valid, started_at, ended_at, worker_label。
"""

import json
from pathlib import Path

from ..normalize import NormalizedCase, NormalizedRun
from .base import AdapterError


def _is_normalized(data) -> bool:
    if isinstance(data, list):
        return bool(data) and all(_is_normalized(item) for item in data)
    if not isinstance(data, dict):
        return False
    has_model = "model_name" in data or isinstance(data.get("model"), dict)
    has_config = "config_name" in data or isinstance(data.get("config"), dict)
    return has_model and has_config and isinstance(data.get("cases"), list)


def _from_dict(data: dict, manifest: dict) -> NormalizedRun:
    model = data.get("model") or {}
    config = data.get("config") or {}
    cases = []
    for item in data["cases"]:
        cases.append(NormalizedCase(
            case_key=item["case_key"],
            metrics=item.get("metrics") or {},
            n_measured=item.get("n_measured"),
            err_count=item.get("err_count"),
            in_tok=item.get("in_tok"),
            out_tok=item.get("out_tok"),
            params=item.get("params") or {},
            description=item.get("description", ""),
            repro_cmd=item.get("repro_cmd") or item.get("repro"),
            meta=item.get("meta") or {},
        ))
    run = NormalizedRun(
        model_name=data.get("model_name") or model.get("name") or manifest.get("model", ""),
        model_aliases=data.get("model_aliases") or model.get("aliases") or [],
        config_name=data.get("config_name") or config.get("name") or "",
        config_params=data.get("config_params") if data.get("config_params") is not None
        else config.get("params") or {},
        node_label=data.get("node_label") or data.get("node") or manifest.get("node", ""),
        suite_key=data.get("suite_key") or manifest.get("suite_key", ""),
        backend="generic-json",
        cases=cases,
        valid=bool(data.get("valid", True)),
        started_at=data.get("started_at"),
        ended_at=data.get("ended_at"),
        worker_label=data.get("worker_label"),
        repro_cmd=data.get("repro_cmd") or data.get("repro"),
        meta=data.get("meta") or {},
    )
    if not run.config_name or run.config_params is None:
        raise AdapterError("generic-json 缺少 config{name,params}：未声明配置的导入拒绝入库")
    if not run.node_label:
        raise AdapterError("generic-json 缺少 node_label")
    if not run.suite_key:
        raise AdapterError("generic-json 缺少 suite_key")
    run.clean_metrics()
    return run


class GenericJsonAdapter:
    name = "generic-json"

    def _normalized_files(self, files: list[Path]) -> list[tuple[Path, object]]:
        out = []
        for path in files:
            if path.suffix.lower() != ".json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if _is_normalized(data):
                out.append((path, data))
        return out

    def detect(self, files: list[Path], manifest: dict) -> bool:
        return bool(self._normalized_files(files))

    def parse(self, root: Path, manifest: dict) -> list[NormalizedRun]:
        found = self._normalized_files([p for p in sorted(root.rglob("*")) if p.is_file()])
        if not found:
            raise AdapterError("未找到符合 NormalizedRun schema 的 JSON")
        runs = []
        for _path, data in found:
            items = data if isinstance(data, list) else [data]
            for item in items:
                runs.append(_from_dict(item, manifest))
        return runs