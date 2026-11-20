"""核心配置聚合：核心参数相同的配置归为一组，其余参数作为变体差异。"""
import hashlib
import json
from collections import Counter

SECONDARY_KEYS = {
    "compilation_config", "hf_overrides", "gpu_memory_utilization",
    "enable_log_requests", "no_scheduler_reserve_full_isl",
}


def core_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in SECONDARY_KEYS}


def group_label(core: dict) -> str:
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
    return "-".join(parts) or "config"


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
            label = group_label(core)
            if label == "config":
                label = cfg["name"]
            groups[key] = {"key": key, "label": label,
                           "configs": [], "id": cfg["id"]}
            order.append(key)
        groups[key]["configs"].append(cfg)
        groups[key]["id"] = min(groups[key]["id"], cfg["id"])
    out = [groups[k] for k in order]
    counts = Counter(g["label"] for g in out)
    for group in out:
        if counts[group["label"]] > 1:
            digest = hashlib.sha1(group["key"].encode("utf-8")).hexdigest()[:6]
            group["label"] = f"{group['label']}-{digest}"
    return out
