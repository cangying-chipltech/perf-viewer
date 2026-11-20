"""适配器接口与注册表。

新增后端 = 新增适配器并在 REGISTRY 显式登记。
parse 返回 NormalizedRun 或其列表（node-csv 一文件多节点 → 每节点一个 run）。
"""

from pathlib import Path
from typing import Protocol, Union

from ..normalize import NormalizedRun


class BackendAdapter(Protocol):
    name: str

    def detect(self, files: list[Path], manifest: dict) -> bool: ...

    def parse(self, root: Path, manifest: dict) -> Union[NormalizedRun, list[NormalizedRun]]: ...


class AdapterError(ValueError):
    pass


def manifest_model(manifest: dict) -> tuple[str, list[str]]:
    model = manifest.get("model")
    if isinstance(model, dict):
        name = model.get("name") or ""
        aliases = list(model.get("aliases") or [])
    else:
        name = model or ""
        aliases = list(manifest.get("model_aliases") or [])
    if not name:
        raise AdapterError("manifest 缺少 model")
    return name, aliases


def manifest_config(manifest: dict) -> tuple[str, dict]:
    config = manifest.get("config")
    if not isinstance(config, dict) or not config.get("name") or config.get("params") is None:
        raise AdapterError("manifest 缺少 config{name,params}：未声明配置的导入拒绝入库")
    return config["name"], config["params"]


def manifest_node(manifest: dict) -> str:
    node = manifest.get("node")
    if isinstance(node, dict):
        return node.get("label") or ""
    return node or ""


def as_runs(parsed) -> list[NormalizedRun]:
    if isinstance(parsed, NormalizedRun):
        return [parsed]
    return list(parsed)


def build_registry() -> dict[str, BackendAdapter]:
    from .guidellm_suite import GuidellmSuiteAdapter
    from .probe_json import ProbeJsonAdapter
    from .node_csv import NodeCsvAdapter
    from .generic_json import GenericJsonAdapter

    registry: dict[str, BackendAdapter] = {}
    for adapter in (GuidellmSuiteAdapter(), ProbeJsonAdapter(), NodeCsvAdapter(), GenericJsonAdapter()):
        registry[adapter.name] = adapter
    return registry


REGISTRY: dict[str, BackendAdapter] = build_registry()

STUBS = {
    "ssh-pull": "预留：从远程 worker SSH/rsync 拉取产物。v1 需人工 rsync 回本机后用 CLI 导入。",
    "profile-workflow": "预留：对接 profile-service 工作流执行结果（如 wf_62e1ceefa2a4）。v1 不实现。",
}


def detect_adapter(files: list[Path], manifest: dict) -> BackendAdapter:
    declared = manifest.get("backend")
    if declared:
        if declared in STUBS:
            raise AdapterError(f"适配器 {declared} 仅为预留接口：{STUBS[declared]}")
        if declared not in REGISTRY:
            raise AdapterError(f"未知 backend: {declared}（可选：{', '.join(REGISTRY)}）")
        return REGISTRY[declared]
    for adapter in REGISTRY.values():
        if adapter.detect(files, manifest):
            return adapter
    raise AdapterError("无法自动探测 backend，请在 manifest 中显式声明")