"""初始目录种子 + 验收数据导入。

用法：python -m app.seed [--assets-dir /home/cangying]
目录种子幂等；数据导入依赖 importer 的重复检测（同指纹配置+同 run 特征 → duplicate）。
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from . import config
from .db import get_db, init_db
from .entities import ensure_case, ensure_suite, resolve_model, seed_metric_defs
from .importer import import_runs

TWELVE_CASES = [
    ("A_256", "A", 1, 3, 0, 200000, 256, 0, "200k 输入全冷（无前缀），256 输出"),
    ("B_256", "B", 1, 10, 180000, 20000, 256, 1, "180k 前缀 + 20k 后缀，256 输出，预热 1"),
    ("C_c1", "C", 1, 10, 180000, 20000, 256, 1, "同 B，1 并发"),
    ("C_c2", "C", 2, 10, 180000, 20000, 256, 1, "同 B，2 并发"),
    ("E_64k_1024", "E", 1, 10, 57600, 6400, 1024, 1, "57.6k 前缀 + 6.4k 后缀，1024 输出"),
    ("E_64k_10240", "E", 1, 3, 57600, 6400, 10240, 1, "57.6k 前缀 + 6.4k 后缀，10240 输出"),
    ("F_200k_cold1k", "F", 1, 5, 199000, 1000, 256, 1, "199k 前缀 + 1k 冷后缀，256 输出"),
    ("F_200k_cold2k", "F", 1, 5, 198000, 2000, 256, 1, "198k 前缀 + 2k 冷后缀，256 输出"),
    ("F_200k_cold3k", "F", 1, 5, 197000, 3000, 256, 1, "197k 前缀 + 3k 冷后缀，256 输出"),
    ("F_64k_cold1k", "F", 1, 5, 63000, 1000, 256, 1, "63k 前缀 + 1k 冷后缀，256 输出"),
    ("F_64k_cold2k", "F", 1, 5, 62000, 2000, 256, 1, "62k 前缀 + 2k 冷后缀，256 输出"),
    ("F_64k_cold3k", "F", 1, 5, 61000, 3000, 256, 1, "61k 前缀 + 3k 冷后缀，256 输出"),
]

TP8_PARAMS = {
    "tensor_parallel_size": 8,
    "decode_context_parallel_size": 8,
    "block_size": 256,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 4096,
    "moe_ep_size": 2,
    "num_speculative_tokens": 5,
    "enable_prefix_caching": True,
}

PREFILL_COMMON = {
    "tensor_parallel_size": 16,
    "num_gpu_blocks_override": 1024,
    "block_size": 256,
    "max_model_len": "1020k",
    "decode_context_parallel_size": 4,
    "moe_ep_size": 2,
    "num_speculative_tokens": 5,
}

# 部署信息取证自 profile-service 执行记录（trace_commands 展开命令），api-key 已脱敏。
TP8_DEPLOY = {
    "docker_cmd": (
        "docker run -itd \\\n"
        "  --name workflow_vllm_glm53_swe \\\n"
        "  --net=host \\\n"
        "  --runtime=chipltech -e CHIPLTECH_VISIBLE_DEVICES=all \\\n"
        "  --shm-size 256G \\\n"
        "  --cap-add=IPC_LOCK \\\n"
        "  --cap-add=SYS_NICE \\\n"
        "  --ulimit memlock=-1:-1 \\\n"
        "  -v /dev/hugepages:/dev/hugepages \\\n"
        "  -v /data:/data hangzhou-office-harbor.infraai.top/qingting/vllm_glm:20261111_base1102"
    ),
    "env": {
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "86400",
        "VLLM_DLC_KV_STAGING_ENABLE": "1",
        "DLCCL_ORDERED_BF16_ALLREDUCE": "1",
        "DLCCL_LYP8_ALLREDUCE_ALGO": "tree",
        "DLC_SPARSE_MLA_DCP_LOCAL_TOPK": "0",
        "DLC_SYN_CACHESIZE": "256",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "DLC_SYN_GRAPH_CACHE": "1",
        "OMP_NUM_THREADS": "8",
        "VLLM_DLC_INDEX_TOPK": "256",
        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "1024",
        "DLC_SYN_COPY_ASYNC": "O2",
        "DLC_SYN_URING": "0",
        "VLLM_USE_DLC_COL_MAJOR_MATMUL": "1",
    },
    "serve_cmd": (
        "vllm serve /data/models/GLM-5.3-MTP-Mix \\\n"
        "  --served-model-name cltech/glm53 \\\n"
        "  --host 0.0.0.0 --port 8000 \\\n"
        "  --api-key *** \\\n"
        "  --dtype bfloat16 --trust-remote-code -tp 8 -dcp 8 \\\n"
        "  --enable-prefix-caching --enable-expert-parallel \\\n"
        "  --enable-auto-tool-choice --enable-prompt-tokens-details \\\n"
        "  --tool-call-parser glm47 --reasoning-parser glm45 \\\n"
        "  --max-model-len 1048576 --block-size 256 --max-num-seqs 4 \\\n"
        "  --gpu-memory-utilization 0.96 --max-num-batched-tokens 4096 \\\n"
        "  --long-prefill-token-threshold 1024 --no-scheduler-reserve-full-isl \\\n"
        "  --compilation-config '{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \\\n"
        "  --hf-overrides '{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}' \\\n"
        "  --speculative-config '{\"num_speculative_tokens\":5,\"method\":\"deepseek_mtp\"}' \\\n"
        "  --moe-ep-size 2"
    ),
    "source": ("profile-service 192.168.80.6 · vllm-deploy (wf_2193501a6e1b) · "
               "exec_1118104d5693 · 2026-11-17 11:30 +08:00；"
               "serve 在容器 workflow_vllm_glm53_swe 内执行；报告称 .6/.8/.9/.10 四节点参数一致"),
}

TP16_DOCKER_CMD = (
    "docker run -itd --name workflow-glm-53-tp16 \\\n"
    "  --net=host --runtime=chipltech -e CHIPLTECH_VISIBLE_DEVICES=all \\\n"
    "  --shm-size 256G --cap-add=IPC_LOCK --cap-add=SYS_NICE \\\n"
    "  --ulimit memlock=-1:-1 --privileged \\\n"
    "  -p 8000:8000 \\\n"
    "  -v /dev/hugepages:/dev/hugepages -v /data:/data \\\n"
    "  hangzhou-office-harbor.infraai.top/qingting/vllm_glm:20261102_base1102_with_ray"
)

TP16_ENV = {
    "DLC_SPARSE_MLA_DCP_LOCAL_TOPK": "1",
    "DLCCL_ENABLE_HOST_D2H2D_KERNEL": "0",
    "DLCCL_LYP_INTER_NODE": "1",
    "DLCCL_RAS_STREAM_TIMEOUT_ENABLE": "0",
    "GLOO_SOCKET_IFNAME": "eno1",
    "DLCCL_SOCKET_IFNAME": "eno1",
    "VLLM_DLC_INDEX_TOPK": "256",
    "DLC_SYN_COPY_ASYNC": "O3",
    "DLC_SYN_GRAPH_CACHE": "1",
    "DLCCL_DISABLE_MEM_MANAGER": "1",
    "VLLM_TORCH_PROFILER_TOKENS": "50",
    "VLLM_USE_DLC_COL_MAJOR_MATMUL": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "86400",
}

_TP16_RAY_ENV = (
    "export DLC_SPARSE_MLA_DCP_LOCAL_TOPK=1 DLC_SYN_COPY_ASYNC=O3 "
    "VLLM_USE_DLC_COL_MAJOR_MATMUL=1 DLC_SYN_GRAPH_CACHE=1 "
    "DLCCL_ENABLE_HOST_D2H2D_KERNEL=0 DLCCL_LYP_INTER_NODE=1 "
    "DLCCL_RAS_STREAM_TIMEOUT_ENABLE=0 GLOO_SOCKET_IFNAME=eno1 DLCCL_SOCKET_IFNAME=eno1; "
    "ray stop --force >/dev/null 2>&1 || true; "
)

TP16_RAY_HEAD_CMD = (
    "docker exec workflow-glm-53-tp16 bash -lc '" + _TP16_RAY_ENV
    + "ray start --head --port=6379 --node-ip-address=192.168.80.4 --dashboard-host=0.0.0.0'"
)

TP16_RAY_WORKER_CMD = (
    "docker exec workflow-glm-53-tp16 bash -lc '" + _TP16_RAY_ENV
    + "ray start --address=192.168.80.4:6379'"
)


def tp16_deploy(threshold: int, workflow: str, execution: str) -> dict:
    return {
        "docker_cmd": TP16_DOCKER_CMD,
        "env": dict(TP16_ENV),
        "serve_cmd": (
            "vllm serve /data/models/GLM-5.3-MTP-Mix \\\n"
            "  --served-model-name cltech/glm53 \\\n"
            "  --host 0.0.0.0 --port 8000 \\\n"
            "  -tp 16 --distributed-executor-backend ray \\\n"
            "  --api-key *** \\\n"
            "  --dtype bfloat16 --gpu-memory-utilization 0.97 --trust-remote-code \\\n"
            "  --enable-expert-parallel --enable-prefix-caching \\\n"
            "  --enable-auto-tool-choice --enable-prompt-tokens-details \\\n"
            "  --tool-call-parser glm47 --reasoning-parser glm45 \\\n"
            "  --block-size 256 --max-num-seqs 4 --max-num-batched-tokens 4096 \\\n"
            "  --enable-log-requests --num-gpu-blocks-override 1024 --max-model-len 1020k \\\n"
            f"  --long-prefill-token-threshold {threshold} \\\n"
            "  --compilation-config '{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' \\\n"
            "  --speculative-config '{\"num_speculative_tokens\":5,\"method\":\"deepseek_mtp\"}' \\\n"
            "  --moe-ep-size 2 -dcp 4"
        ),
        "extra_cmds": [
            {"title": "head 192.168.80.4 · 容器内启动 Ray（--head）", "cmd": TP16_RAY_HEAD_CMD},
            {"title": "worker 192.168.80.5 · 容器内加入 Ray 集群（--address）",
             "cmd": TP16_RAY_WORKER_CMD},
        ],
        "source": (f"profile-service 192.168.80.4 · {workflow} · {execution} · 2026-11-17；"
                   "head 192.168.80.4 / worker 192.168.80.5，"
                   "serve 在容器 workflow-glm-53-tp16 内执行（docker exec）；"
                   "Ray 启动命令取自 tp16-cluster 执行记录"
                   "（head exec_bf21ed311644 / worker exec_f5759a90e7ed）"),
    }


def seed_catalog(conn) -> dict:
    seed_metric_defs(conn)
    model = resolve_model(conn, "cltech/glm53", ["GLM-5.3-MTP-Mix"])

    suite_12 = ensure_suite(conn, "glm53-12case", "GLM-5.3 12 档基准套件",
                            "200k 交付：A/B/C/E/F 组 12 case（cases.tsv 矩阵）")
    for idx, (key, group, conc, reqs, prefix, suffix, out, warmup, desc) in enumerate(TWELVE_CASES):
        ensure_case(conn, suite_12["id"], key, {
            "group": group, "concurrency": conc, "requests": reqs,
            "prefix_tokens": prefix, "suffix_tokens": suffix,
            "input_tokens": prefix + suffix, "output_tokens": out, "warmup": warmup,
        }, desc, idx)

    suite_1m = ensure_suite(conn, "1m-prefill", "1M prefill 探针",
                            "1m 交付：1M 输入 + 1 输出长上下文 prefill 探针")
    ensure_case(conn, suite_1m["id"], "1m_in_1out", {
        "input_tokens": 1000000, "output_tokens": 1, "temperature": 0, "ignore_eos": True,
    }, "1,000,000 输入 token + 1 输出 token，temperature 0，ignore_eos", 0)

    scenarios = {}
    for name, desc, suite_keys in (
        ("性能配置", "性能调优场景（空壳，待关联测试集合）", []),
        ("精度配置", "精度验证场景（空壳，待关联测试集合）", []),
        ("200k交付配置", "200k 上下文交付：12-case 套件跨节点对比", ["glm53-12case"]),
        ("1m交付配置", "1M 上下文交付：1M prefill 探针阈值对比", ["1m-prefill"]),
    ):
        row = conn.execute("SELECT * FROM scenario WHERE model_id = ? AND name = ?",
                           (model["id"], name)).fetchone()
        if not row:
            cur = conn.execute(
                "INSERT INTO scenario(model_id, name, description, scoring) VALUES(?,?,?,?)",
                (model["id"], name, desc, json.dumps(config.DEFAULT_SCORING)))
            sid = cur.lastrowid
        else:
            sid = row["id"]
        for key in suite_keys:
            suite = conn.execute("SELECT * FROM suite WHERE key = ?", (key,)).fetchone()
            conn.execute("INSERT OR IGNORE INTO scenario_suite(scenario_id, suite_id) VALUES(?,?)",
                         (sid, suite["id"]))
        scenarios[name] = sid
    return {"model": model, "scenarios": scenarios}


def _stage(files: list[Path]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="pv-seed-", dir=str(config.DATA_DIR)))
    for f in files:
        if f.exists():
            shutil.copyfile(f, root / f.name)
    return root


def seed_imports(assets: Path, scenarios: dict) -> list[dict]:
    reports = []
    csv = assets / "glm53_12case_node_details_20261117.csv"
    report_md = assets / "glm53_12case_node_report_20261117.md"
    pngs = [assets / "1m_prefill_first100k_10k_segments.png",
            assets / "1m_prefill_segment_rate_4nodes.png"]
    if csv.exists():
        root = _stage([csv, report_md, *pngs])
        try:
            manifest = {
                "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
                "config": {
                    "name": "tp8-dcp8-bs256-mns4-mnbt4096-ep2-mtp5",
                    "params": TP8_PARAMS,
                    "notes": "4 节点参数完全一致；MTP-5，prefix cache 开启（2026-11-17 报告）",
                    "deploy": TP8_DEPLOY,
                },
                "suite_key": "glm53-12case",
                "backend": "node-csv",
                "scenario_id": scenarios["200k交付配置"],
                "started_at": "2026-11-17T15:34:09+08:00",
                "ended_at": "2026-11-17T16:42:12+08:00",
            }
            reports.append(import_runs(root, manifest, source=f"seed:{csv.name}"))
        finally:
            shutil.rmtree(root, ignore_errors=True)
    else:
        reports.append({"status": "skipped", "reason": f"缺少 {csv}"})

    summary_md = assets / "llm-load-results" / "1m-prefill-summary.md"
    for threshold, wf, node_wf, node_exec in (
            (1024, "tp16-serve (wf_de00b5035579)", "tp16-serve (wf_80028b2b74d9)",
             "exec_00f002f1bd18"),
            (2048, "tp16-serve-prefill-2048 (wf_62e1ceefa2a4)",
             "tp16-serve-prefill-2048 (wf_2b63529065f1)", "exec_d84e2980b871")):
        probe = assets / "llm-load-results" / f"1m-prefill-threshold{threshold}.json"
        if not probe.exists():
            reports.append({"status": "skipped", "reason": f"缺少 {probe}"})
            continue
        extras = [summary_md] if threshold == 1024 else []
        root = _stage([probe, *extras])
        try:
            params = dict(PREFILL_COMMON)
            params["long_prefill_token_threshold"] = threshold
            manifest = {
                "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
                "config": {
                    "name": f"tp16-dcp4-gb1024-lpt{threshold}",
                    "params": params,
                    "notes": f"工作流 {wf}；节点 192.168.80.4，worker wenling-5；"
                             "其余参数两套配置相同，仅 --long-prefill-token-threshold 不同",
                    "deploy": tp16_deploy(threshold, node_wf, node_exec),
                },
                "suite_key": "1m-prefill",
                "backend": "probe-json",
                "case_key": "1m_in_1out",
                "node": "192.168.80.4",
                "scenario_id": scenarios["1m交付配置"],
            }
            reports.append(import_runs(root, manifest, source=f"seed:{probe.name}"))
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description="perf-viewer 初始种子")
    parser.add_argument("--assets-dir", default="/home/cangying")
    args = parser.parse_args(argv)
    init_db()
    with get_db() as conn:
        info = seed_catalog(conn)
    reports = seed_imports(Path(args.assets_dir), info["scenarios"])
    print(json.dumps({
        "model": info["model"]["name"],
        "scenarios": info["scenarios"],
        "imports": [{k: r.get(k) for k in ("status", "batch_id", "backend", "source")}
                    | {"runs": [(x.get("node"), x.get("status"), x.get("run_id"))
                                for x in r.get("runs", [])]}
                    for r in reports],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()