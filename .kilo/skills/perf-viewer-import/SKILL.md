---
name: perf-viewer-import
version: 1.0.0
description: 把压测结果导入 perf-viewer（模型→场景→配置集→节点×运行→case）。支持手工启动 vLLM 后用 vllm-bench、evalscope、guidellm 产出的数据，也支持 profile-service 的 vllm-deploy / tp16-cluster / tp16-serve 部署记录与 benchmark-suite 执行结果；生成 manifest/normalized.json 并通过 CLI/HTTP dry-run 导入。Use when 用户提到 上传/导入 perf-viewer 数据、生成 manifest、normalized.json、evalscope、guidellm、vllm-bench、benchmark-suite、vllm-deploy、tp16-cluster、tp16-serve。
---

# perf-viewer 数据导入

将压测产物归一化为 perf-viewer 的 `generic-json` 输入，再调用导入 API/CLI。**先 dry-run，确认 `status=ok` 且 `would-import/duplicate` 符合预期后再正式导入。**

本 skill 位于项目根目录：

```bash
cd /home/cangying/perf-viewer
K=.kilo/skills/perf-viewer-import/scripts/import_kit.py
```

## 1. 导入流程

```bash
# 1) 生成 <out>/normalized.json 与 <out>/manifest.json
python3 $K <vllm-bench|evalscope|benchmark-suite> ... --out ./import/<run-name> ...

# 2) dry-run（token 由 .env 自动读取）
.venv/bin/python -m app.cli import ./import/<run-name> \
  --manifest ./import/<run-name>/manifest.json --dry-run

# 3) 正式导入
.venv/bin/python -m app.cli import ./import/<run-name> \
  --manifest ./import/<run-name>/manifest.json
```

也可直接调 API（`PERF_VIEWER_TOKEN` 为空时写接口返回 503）：

```bash
TOKEN=$(grep '^PERF_VIEWER_TOKEN=' .env | cut -d= -f2-)
curl -X POST http://127.0.0.1:8600/api/import \
  -H "Authorization: Bearer $TOKEN" \
  -F manifest=@./import/run1/manifest.json \
  -F files=@./import/run1/normalized.json -F dry_run=1
```

同一配置指纹 + 同一 run 特征重复导入会返回 `duplicate`，不会重复入库。

## 2. 必填信息

manifest/归一化数据至少要有：

- `model.name`（served name；别名放 `model.aliases`）
- `config.name` + `config.params`（**未声明配置拒绝入库**；参数会归一化后做指纹）
- `suite_key`（suite/case 不存在会按数据自动创建）
- `node_label`（节点 IP/hostname）
- `backend`（本工具统一写 `generic-json`）

建议同时提供 `config.deploy`（`docker_cmd` / `serve_cmd` / `env` / `source`）和 `scenario_id`，用于部署信息展示与场景关联。

### 配置集归一口径

- 服务端配置 = `config.params`（serve 参数 + `image`）+ `config.deploy`。
- `image` 建议只写 `name:tag`；`import_kit.py` 会自动去掉 registry/namespace 与 `_with_ray` 后缀，避免同镜像不同仓库被拆成多套配置。
- 次要参数（如 `compilation_config`、`hf_overrides`、`gpu_memory_utilization`、`enable_log_requests`、`no_scheduler_reserve_full_isl`）在 perf-viewer 的 **PK 聚合**中会被视为变体差异，但它们仍会进入配置指纹，因此同一核心配置的不同变体不会被静默合并。

## 3. 手工启动 server + 手工压测

### 3.1 手工启动 vLLM（采集部署信息）

```bash
IMAGE=hangzhou-office-harbor.infraai.top/qingting/vllm_glm:20261111_base1102
docker run -itd --name vllm-tp8 --net=host --shm-size 256G --privileged \
  -v /data:/data "$IMAGE"

docker exec vllm-tp8 vllm serve /data/models/GLM-5.3-MTP-Mix \
  --served-model-name cltech/glm53 --host 0.0.0.0 --port 8000 --api-key '***' \
  --dtype bfloat16 -tp 8 -dcp 8 --max-model-len 1048576 --block-size 256 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.96 \
  --enable-prefix-caching --enable-expert-parallel --long-prefill-token-threshold 1024
```

把真实 `docker run`、`vllm serve`、环境变量写进导入参数（示例中的 key 用占位符，导入时会脱敏）：

```bash
--docker-cmd 'docker run -itd ... vllm_glm:20261111_base1102' \
--serve-cmd  'vllm serve ... -tp 8 -dcp 8 ...' \
--env VLLM_USE_V2_MODEL_RUNNER=0 --env GLOO_SOCKET_IFNAME=eno1 \
--image vllm_glm:20261111_base1102 \
--source '手工启动 · 2026-11-20'
```

### 3.2 vllm-bench（benchmark_serving.py）

执行时加 `--save-result` 保存 JSON：

```bash
python3 benchmark_serving.py --backend openai-chat \
  --base-url http://127.0.0.1:8000 --model cltech/glm53 \
  --dataset-name random --random-input-len 64000 --random-output-len 1024 \
  --num-prompts 100 --max-concurrency 64 --save-result \
  --result-filename vllm-bench-c64.json
```

转换并导入：

```bash
python3 $K vllm-bench --result vllm-bench-c64.json \
  --out ./import/vllm-c64 \
  --model cltech/glm53 --alias GLM-5.3-MTP-Mix \
  --config-name tp8-dcp8-mns4 \
  --param tensor_parallel_size=8 --param decode_context_parallel_size=8 \
  --param max_num_seqs=4 --param max_num_batched_tokens=4096 \
  --image vllm_glm:20261111_base1102 \
  --node 192.168.80.4 --worker-label wenling-4 --suite vllm-bench \
  --case-key c64 --case-param max_concurrency=64 \
  --case-param input_tokens=64000 --case-param output_tokens=1024 \
  --started-at 2026-11-20T10:00:00+08:00 --ended-at 2026-11-20T10:05:00+08:00 \
  --scenario-id 4 --source '手工 vllm-bench'
```

`import_kit.py` 映射：`mean/p99_ttft_ms→*_s`、`mean/p99_e2el_ms→*_s`、`mean/p99_tpot_ms`、`mean/p99_itl_ms`、`output_throughput→out_tps`、`completed→n_measured`、`failed→err_count`。

### 3.3 evalscope

EvalScope 执行后通常得到 `benchmark_summary.json`（不同版本目录可能不同）：

```bash
evalscope perf --model cltech/glm53 --api-url http://127.0.0.1:8000/v1 \
  --dataset random --num-prompts 80 --max-concurrency 64 \
  --min-prompt-length 64000 --max-prompt-length 64000 --max-tokens 1024 \
  --outputs ./evalscope-out
```

转换：

```bash
python3 $K evalscope --summary ./evalscope-out/**/benchmark_summary.json \
  --out ./import/evalscope-c64 \
  --model cltech/glm53 \
  --config-name tp8-dcp8-mns4 \
  --param tensor_parallel_size=8 --param decode_context_parallel_size=8 \
  --image vllm_glm:20261111_base1102 \
  --node 192.168.80.4 --suite evalscope --case-key c64 \
  --time-unit s \
  --scenario-id 4 --source '手工 evalscope'
```

- 默认 `--time-unit s`：TTFT/E2E 按秒，TPOT/ITL 由秒换算为 ms。
- 若 summary 中时间字段实际是毫秒，改 `--time-unit ms`。
- 脚本会按 `AverageTTFT/P99TTFT/AverageLatency/P99Latency/AverageTPOT/P99TPOT/AverageITL/P99ITL/AverageTPS/TotalRequests` 等常见字段匹配；若报“指标映射为空”，先查看 summary 的字段名，必要时手工生成 normalized.json（schema 见第 5 节）。

### 3.4 GuideLLM

GuideLLM 完整套件使用内置 `guidellm-suite` 适配器，无需转换。目录需包含每个 case 的：

```
results/
  B_256/
    guidellm.json
    meta.json
    server_metrics.json
    timing.csv
```

manifest：

```json
{
  "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
  "config": {
    "name": "tp16-dcp4-gb1024-lpt1024",
    "params": {"tensor_parallel_size": 16, "decode_context_parallel_size": 4,
               "num_gpu_blocks_override": 1024, "long_prefill_token_threshold": 1024,
               "image": "vllm_glm:20261102_base1102"},
    "deploy": {"docker_cmd": "...", "serve_cmd": "...", "env": {}, "source": "..."}
  },
  "suite_key": "glm53-12case",
  "backend": "guidellm-suite",
  "node": "192.168.80.4",
  "worker_label": "wenling-4",
  "scenario_id": 4
}
```

```bash
.venv/bin/python -m app.cli import ./results --manifest ./manifest.json --dry-run
```

## 4. profile-service 部署记录与 benchmark-suite

使用 `profile-workflow-ops` skill 的 CLI（示例路径）：

```bash
S=$HOME/.agents/skills/profile-workflow-ops/scripts/profile_ops.py
# 若已安装到项目内，则改为 .kilo/skills/profile-workflow-ops/scripts/profile_ops.py
U=http://127.0.0.1:18729
python3 $S health --base-url $U
```

> 部署配置取 **服务端执行记录**；benchmark-suite 的 `config.json` 是压测客户端配置（backend/tokenizer/base_url），不能当服务端配置。

### 4.1 vllm-deploy（单机部署，如 tp8）

1. 找部署执行：`executions --json` 或 `execution-get <id>`，确认 `workflow_name=vllm-deploy`、节点、时间。
2. 拉取执行命令：

```bash
python3 $S logs --execution <exec_id> --base-url $U --list
python3 $S logs --execution <exec_id> --file commands/vllm_deploy.container_ensure.executed.sh --base-url $U
python3 $S logs --execution <exec_id> --file commands/vllm_deploy.model_start.sh --base-url $U
```

从输出提取 `docker run`、`vllm serve`、`-e K=V`、镜像，填入 `--docker-cmd/--serve-cmd/--env/--image`。

### 4.2 tp16-cluster + tp16-serve（多机 Ray）

- `tp16-cluster`：Ray head/worker 启动命令，通常包含 `ray start --head --address ...`。
- `tp16-serve`：head 节点容器内的 `vllm serve` 与 `docker run`，以及 `compilation-config`、`long-prefill-token-threshold` 等关键参数。
- 执行记录里的 `commands/*.executed.sh`（或 `tp16.serve_start.executed.sh` / `tp16.container_ensure.executed.sh` / `tp16.ray_start.executed.sh`）就是实际命令，作为 `config.deploy` 的来源，`source` 写 `profile-service <node> · <workflow> · <execution>`。

### 4.3 benchmark-suite 结果

执行产物（服务端执行目录）包含：

```
benchmark-suite/
  summary.json      # 逐 case 指标与 status
  config.json       # 压测客户端配置（不是服务端配置）
  requests.jsonl    # 可选，用于 cache_hit_rate
  manifest.json
run.log
```

拉取：

```bash
python3 $S logs --execution <exec_id> --base-url $U --list
python3 $S logs --execution <exec_id> --file benchmark-suite/summary.json --base-url $U --save-dir ./raw
python3 $S logs --execution <exec_id> --file benchmark-suite/config.json  --base-url $U --save-dir ./raw
python3 $S logs --execution <exec_id> --file benchmark-suite/requests.jsonl --base-url $U --save-dir ./raw
```

转换（服务端配置来自 4.1/4.2）：

```bash
python3 $K benchmark-suite \
  --summary ./raw/<exec_id>/benchmark-suite/summary.json \
  --config  ./raw/<exec_id>/benchmark-suite/config.json \
  --requests ./raw/<exec_id>/benchmark-suite/requests.jsonl \
  --out ./import/bench-<exec_id> \
  --model cltech/glm53 \
  --config-name tp16-dcp4-gb1024-lpt1024-20261102 \
  --param tensor_parallel_size=16 --param decode_context_parallel_size=4 \
  --param num_gpu_blocks_override=1024 --param long_prefill_token_threshold=1024 \
  --param 'compilation_config={"mode":0,"cudagraph_mode":"NONE"}' \
  --image vllm_glm:20261102_base1102 \
  --docker-cmd 'docker run -itd ...' \
  --serve-cmd 'vllm serve ... -tp 16 -dcp 4 ...' \
  --env GLOO_SOCKET_IFNAME=eno6 \
  --node 172.31.8.31 --worker-label hangdian-31 \
  --started-at 2026-11-20T09:47:23+08:00 --ended-at 2026-11-20T10:57:58+08:00 \
  --scenario-id 4 \
  --source 'profile-service 172.31.8.31 · benchmark-suite · <exec_id>'
```

说明：脚本只导入 `status=success` 的 case；`cache_hit_rate` 仅当提供 `requests.jsonl` 时按 `sum(cached_tokens)/sum(prompt_tokens)` 计算。

## 5. 手工 normalized.json（适配器无法覆盖时）

在输出目录放一个符合以下 schema 的 JSON，即可用 `generic-json` 导入：

```json
{
  "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
  "config": {
    "name": "tp8-dcp8",
    "params": {"tensor_parallel_size": 8, "decode_context_parallel_size": 8,
               "image": "vllm_glm:20261111_base1102"}
  },
  "suite_key": "my-suite",
  "node_label": "192.168.80.4",
  "worker_label": "wenling-4",
  "backend": "generic-json",
  "valid": true,
  "started_at": "2026-11-20T10:00:00+08:00",
  "ended_at": "2026-11-20T10:05:00+08:00",
  "cases": [{
    "case_key": "c64",
    "metrics": {
      "e2e_mean_s": 5.12, "e2e_p99_s": 7.2,
      "ttft_mean_s": 0.32, "ttft_p99_s": 0.88,
      "tpot_mean_ms": 18.2, "tpot_p99_ms": 24.0,
      "itl_mean_ms": 17.9, "itl_p99_ms": 23.5,
      "out_tps": 212.4, "cache_hit_rate": 0.87
    },
    "n_measured": 100, "err_count": 0,
    "in_tok": 128000, "out_tok": 25600,
    "params": {"max_concurrency": 64, "input_tokens": 64000, "output_tokens": 1024},
    "description": "并发 64 · 输入 64000 · 输出 1024"
  }]
}
```

manifest 与 normalized 的 `model/config/suite_key/node` 保持一致；manifest 额外支持 `scenario_id` 与 `config.deploy`。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `503` / 写接口失败 | `.env` 未配置 `PERF_VIEWER_TOKEN`，或请求未带 Bearer token |
| `duplicate` | 同配置指纹 + 同 run 特征已存在；确认是否确实重复 |
| `node_label` 缺失 | 手工指定 `--node`；CSV/探针/guidellm 可从产物推导 |
| 指标映射为空 | 检查 summary 字段名；用第 5 节手工 normalized.json |
| 配置被拆成两套 | 检查 `image` 是否带 registry/`_with_ray`；用 `--image vllm_glm:<tag>` |
| 失败 case 没导入 | benchmark-suite 默认只导入 success；调试可加 `--include-failed` |
| 想挂报告/产物 | 传 `--scenario-id`，未消费文件会挂到场景 |
