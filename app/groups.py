"""核心配置聚合：核心参数相同的配置归为一组，其余参数作为变体差异。"""
import hashlib
import json
import re
from collections import Counter

SECONDARY_KEYS = {
    "compilation_config", "hf_overrides", "gpu_memory_utilization",
    "enable_log_requests", "no_scheduler_reserve_full_isl",
}

# 配置名 = 部署名 + 8 位指纹；聚合标签只保留部署名，变体数量另行标注。
_FINGERPRINT = re.compile(r"-[0-9a-f]{8}$")


def core_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in SECONDARY_KEYS}


def config_stem(name: str) -> str:
    """配置名去掉指纹后缀，用于聚合标签。"""
    value = str(name or "").strip()
    return _FINGERPRINT.sub("", value) or value


def params_label(core: dict) -> str:
    parts = []
    for key, prefix in (("tensor_parallel_size", "tp"),
                        ("decode_context_parallel_size", "dcp"),
                        ("num_gpu_blocks_override", "gb"),
                        ("long_prefill_token_threshold", "lpt")):
        value = core.get(key)
        if value is not None:
            parts.append(f"{prefix}{value}")
    image = core.get("image")
    if image:
        parts.append(str(image).split(":")[-1])
    return "-".join(parts)


def group_label(core: dict, configs: list[dict] | None = None) -> str:
    """聚合组标签：优先用配置名（部署名），没有配置名时回退到核心参数。"""
    stems: list[str] = []
    for cfg in configs or []:
        stem = config_stem(cfg.get("name"))
        if stem and stem not in stems:
            stems.append(stem)
    if stems:
        if len(stems) == 1:
            label = stems[0]
        else:
            label = f"{stems[0]} 等 {len(stems)} 种部署"
        count = len(configs or [])
        if count > 1:
            label = f"{label}（{count} 个变体）"
        return label
    return params_label(core) or "config"


def group_configs(configs: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    order: list[str] = []
    for cfg in configs:
        params = cfg.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        core = core_params(params)
        key = json.dumps(core, sort_keys=True, ensure_ascii=False, default=str)
        if key not in groups:
            groups[key] = {"key": key, "label": "", "configs": [], "id": cfg["id"]}
            order.append(key)
        groups[key]["configs"].append(cfg)
        groups[key]["id"] = min(groups[key]["id"], cfg["id"])
    out = [groups[k] for k in order]
    for group in out:
        core = json.loads(group["key"])
        group["label"] = group_label(core, group["configs"])
        group["params_label"] = params_label(core) or group["label"]
    counts = Counter(g["label"] for g in out)
    for group in out:
        if counts[group["label"]] > 1:
            digest = hashlib.sha1(group["key"].encode("utf-8")).hexdigest()[:6]
            group["label"] = f"{group['label']}-{digest}"
    return out
