---
name: perf-viewer-import
version: 1.1.0
description: 把压测结果导入 perf-viewer（模型→场景→配置集→节点×运行→case），并把每次执行的测试端原始产物（每个用例的 stdout.log/stderr.log）与服务端启动命令（docker run / vllm serve / ray start）随导入一起归档。支持手工启动 vLLM 后用 vllm-bench、evalscope、guidellm 产出的数据，也支持 profile-service 的 vllm-deploy / tp16-cluster / tp16-serve 部署记录与 benchmark-suite 执行结果；生成 manifest/normalized.json 并通过 CLI/HTTP dry-run 导入。Use when 用户提到 上传/导入 perf-viewer 数据、归档执行结果、生成 manifest、normalized.json、evalscope、guidellm、vllm-bench、benchmark-suite、vllm-deploy、tp16-cluster、tp16-serve。
---

# perf-viewer 数据导入

将压测产物归一化为 perf-viewer 的 `generic-json` 输入，再调用导入 API/CLI。**先 dry-run，确认 `status=ok` 且 `would-import/duplicate` 符合预期后再正式导入。**

每次转换都会把本次执行的**测试端原始产物**和**服务端启动命令**归档进输出目录，导入时随 run 上传，perf-viewer 长期保留在 `data/artifacts/<batch>` 下可下载（见第 2 节）。

本 skill 位于项目根目录：

```bash
cd /home/cangying/perf-viewer
K=.kilo/skills/perf-viewer-import/scripts/import_kit.py
```

## 1. 导入流程

```bash
# 1) 生成 <out>/normalized.json、<out>/manifest.json、<out>/archive.json 与归档文件
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

## 2. 每次执行自动归档

`import_kit.py` 转换时会把执行产物复制进 `<out>`，导入时按 run 产物上传（未入库的失败用例日志、
服务端启动命令都会保留）。目录约定：

```
<out>/
  normalized.json / manifest.json / archive.json   # archive.json = 归档索引（含 sha256）
  benchmark-suite/            # benchmark-suite：本次压测的测试端原始产物
    summary.json  summary.csv  config.json  requests.jsonl  manifest.json
    case-<id>/stdout.log  case-<id>/stderr.log  case-<id>/console.log
    case-<id>/case.json   case-<id>/command.json
  raw-results/                # vllm-bench / evalscope：原始结果文件
    <result>.json
  server-start/<exec_id>/     # 服务端启动文件：容器启动、vllm serve、ray start
    commands/*.executed.sh    # 实际执行的 docker run / vllm serve / ray start 命令
    summary.json
  case-logs/<case_key>/       # 入库 case 的日志副本（run 详情页按 case 展示）
    stdout.log  stderr.log
```

| 归档内容 | 来源 | 说明 |
| --- | --- | --- |
| 每个用例的 stdout.log / stderr.log | `case-*/stdout.log`、`case-*/stderr.log` | **含失败用例**；成功用例另有一份 `case-logs/` 副本供 run 详情页展开 |
| 用例命令与定义 | `case-*/command.json`、`case-*/case.json` | 用于复现与核对 |
| 压测摘要 | `summary.json`、`summary.csv`、`config.json`、`requests.jsonl`、`manifest.json` | 测试端原始指标 |
| 服务端启动命令 | profile-service 部署执行 | `commands/*.executed.sh`（容器 `docker run`、`vllm serve`、`ray start`）+ `summary.json` |

服务端启动文件有两种归档方式：

```bash
# A. 直接给部署执行 id（自动从 profile-service 拉取，可重复；多机 Ray 传对端再给一个 --service-url）
python3 $K benchmark-suite ... \
  --deploy-execution exec_71ae8269acbf \
  --deploy-execution exec_34699f710f54 \
  --service-url http://172.31.8.31:18729 \
  --service-url http://172.31.8.32:18729

# B. 已把日志拉到本地目录时直接归档整个目录
python3 $K benchmark-suite ... --deploy-dir ./raw/deploy-exec_71ae8269acbf
```

- `--service-url` 默认取 `$PROFILE_SERVICE_URL`，否则 `http://127.0.0.1:18729`；只取 `commands/*.sh` 与 `summary.json`，其它日志不归档。
- `--execution <id>` 记录本次压测执行 id 到 `archive.json`，便于回溯。
- 不归档原始用例结果时加 `--no-archive-raw`（只上传指标，目录里不会出现 `benchmark-suite/`、`raw-results/`）。
- 带 `--scenario-id` 导入时，未被 case 消费的归档文件挂到**场景**下；不带时挂到本次 **run** 下，均在 run/场景详情页可下载。

## 3. 必填信息

manifest/归一化数据至少要有：

- `model.name`（served name；别名放 `model.aliases`）
- `config.name` + `config.params`（**未声明配置拒绝入库**；参数会归一化后做指纹）
- `suite_key`（suite/case 不存在会按数据自动创建）
- `node_label`（节点 IP/hostname）
- `backend`（本工具统一写 `generic-json`）

建议同时提供 `config.deploy`（`docker_cmd` / `serve_cmd` / `env` / `source`）和 `scenario_id`，用于部署信息展示与场景关联。`scenario_id` 同时限定 run 的 PK 归属：同一 suite 被多个场景复用时，只有本场景导入的 run 参与对比；未带 `scenario_id` 的历史 run 按 suite 共享。

**场景规范（按容器名固化，工作流导入）**：`perf_viewer.scenario_name` / `scenario_id` 都留空时，按部署容器名自动决定场景：

| 容器名 | 场景 |
| --- | --- |
| `workflow_vllm_<model>_perf` | 性能场景 |
| `workflow_vllm_<model>_swe` | SWE 场景 |
| `workflow_vllm_<model>_stage` / `_no_stage` | 单机最佳交付场景 |
| `*tp16*`（如 `workflow-glm-53-tp16`） | TP16 交付场景 |

容器名取自部署记录里的 `docker run --name` 或 `docker exec`。正式导入（`dry_run: false`）时，目标 viewer 上没有该场景会自动创建（连同 suite），已有场景缺 suite 时会补链；预览（`dry_run: true`，模板默认）只查询、不写 viewer，缺场景时在日志里说明“看起来会创建”。命中不了规则的容器不挂场景。显式 `scenario_name` / `scenario_id` 仍然优先。手工 `import_kit.py` 用 `--scenario-id`。

**多机节点**：节点上的工作流只能读到自己节点的 profile-service。tp16 这类双机部署要让 `perf_viewer.service_urls` 带上对端节点地址（逗号分隔，如 `http://172.31.8.31:18729,http://172.31.8.32:18729`），否则 worker 节点的 `ray start --address=...` 不会出现在部署记录里。计划运行时 head-only 的上报步骤会按计划选中的节点自动填充；对端没有该 run 或该工作流的记录会被跳过（例如 worker 不跑压测），有该工作流但无成功部署则直接失败（避免静默丢数据）。手工归档启动命令时把对端地址作为额外的 `--service-url` 传入。

每个数据点（配置 × suite × case）都会在 perf-viewer 中展示「复现命令」：
导入时可用 `--repro-cmd '<命令>'` 记录 run 级命令，或用 `--case-repro <case_key>=<命令>`
记录单个 case 的命令（优先级最高）。未提供时由 perf-viewer 按 backend 和 case 参数生成客户端压测命令；
`backend` 识别不出时回退为 benchmark-suite 命令模板。

**压测复现命令（工作流导入）**：`benchmark-suite` 执行目录里每个 `case-*/command.json` 的 argv 会作为 `config.deploy.extra_cmds` 的一项（标题 `<工具> · <case id>`，如 `evalscope · base-short-1024`）自动上报，perf-viewer 里可直接复制。每个 case 还会在 `cases.<case key>.repro_cmd` 里带一条**客户端复现命令**：直接调用后端（`evalscope perf …` / `vllm-bench serve …` / `guidellm …`），不依赖 suite 脚本和配置；argv 里的 `--outputs-dir`/`--result-dir` 改写成 `./repro/<case key>`，回环的 `--url`/`--base-url`/`--target` 改成节点地址，API key 脱敏后由执行人补。手工导入用 `--bench-cmd 'title::cmd'` 补 `extra_cmds` 的同等内容。工作流导入（`perf_viewer.import`）会先读回 viewer 上同配置已有的 `extra_cmds`，按 `cmd` 去重后与本次命令合并，所以完成一次新压测不会把之前手工补的命令覆盖掉；全手工 curl 没有这一步，除重写 `extra_cmds` 外请把仍需要的命令一起写进 manifest。

### case 日志按测试记录展示（console.log / stdout.log / stderr.log）

run 详情页（`/runs/{id}`）的 case 指标表中，每条测试记录会列出属于该 case 的 `stdout.log`、`stderr.log`、`console.log`，可展开查看并下载；场景页不展示这些文件。推送时按后端选择一种方式：

- **benchmark-suite（evalscope）**：无需额外参数。`import_kit.py benchmark-suite` 会读取 `case-*/case.json` 里的 `id` 映射到 case_key，自动复制每个用例的 `stdout.log`/`stderr.log` 并写入 `stdout_log`/`stderr_log`；同时把全部 case（含失败用例）的日志归档到 `<out>/benchmark-suite/case-*/`（见第 2 节）。
- **guidellm-suite**：无需额外字段，把 `console.log` 放进各 case 目录（与 `meta.json`、`guidellm.json` 同级）即可；适配器会以 `<case_key>/console.log` 自动登记。
- **generic-json / import_kit**：在 normalized.json 的 `cases[]` 增加 `console_log`、`stdout_log`、`stderr_log` 字段，值为**相对导入根目录**的文件路径：

```json
"cases": [{
  "case_key": "B_256",
  "metrics": {"e2e_mean_s": 5.12},
  "console_log": "B_256/console.log",
  "stdout_log": "B_256/stdout.log",
  "stderr_log": "B_256/stderr.log"
}]
```

使用 `import_kit.py` 时也可直接指定 console.log 源文件，脚本会复制到输出目录并写好字段：

```bash
python3 $K benchmark-suite ... \
  --case-console-log long-64k-cold-256=./raw/long-64k-cold-256/console.log
```

- manifest 也支持 `cases.<case_key>.console_log` / `stdout_log` / `stderr_log`。
- 日志路径或标题中的 case_key 决定归属；文件名必须是 `stdout.log`、`stderr.log` 或 `console.log`。
- 文件必须随本次导入一起上传/位于导入目录内；缺失时对应记录显示文件缺失或不展示该文件。
- `benchmark-suite` 的 case 日志已自动归档并上传；手工 `generic-json` 需自行把日志放进导入目录。

### 配置集归一口径

- 服务端配置 = `config.params`（serve 参数 + `image`）+ `config.deploy`。
- `config.params` 由证据推导，不再维护白名单：`vllm serve` 的每个 flag 都进 params（`--max-num-seqs` → `max_num_seqs`，布尔开关 → `true`，`-tp`/`-dcp` 走别名），`docker run` 的每个选项加 `container_` 前缀（`--shm-size` → `container_shm_size`，可重复的 `-v`/`--cap-add`/`--ulimit` 合成有序列表，`--name`/`--label` 因每次不同而丢弃），`docker run -e` 与 serve 导出环境变量进 `env_<NAME>`。凭证（`--api-key`、`*TOKEN*`/`*SECRET*` 等）既不进 params 也不进部署记录。因此以后新增的 vLLM/docker 参数不会漏掉，而是自动成为配置差异。
- `image` 建议只写 `name:tag`；`import_kit.py` 会自动去掉 registry/namespace 与 `_with_ray` 后缀，避免同镜像不同仓库被拆成多套配置。
- 次要参数（如 `compilation_config`、`hf_overrides`、`gpu_memory_utilization`、`enable_log_requests`、`no_scheduler_reserve_full_isl`）在 perf-viewer 的 **PK 聚合**中会被视为变体差异，但它们仍会进入配置指纹，因此同一核心配置的不同变体不会被静默合并。
- perf-viewer 的配置身份包含 `deploy` 里的 serve 命令 flag、模型环境变量与 docker 命令（敏感 flag 除外）：即使 `params` 相同，`--compilation-config` 等只用 serve 命令区分的两次部署也会各成一套配置，且导入**不会**覆盖已有配置集的部署记录。早期版本（指纹只看 params）留下的错配可用 `perf-viewer audit`（或 `GET /api/deploy-audit`）核对，对比页也会给出「部署记录不一致」提示。

## 4. 手工启动 server + 手工压测

### 4.1 手工启动 vLLM（采集部署信息）

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

### 4.2 vllm-bench（benchmark_serving.py）

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

### 4.3 evalscope

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
- 脚本会按 `AverageTTFT/P99TTFT/AverageLatency/P99Latency/AverageTPOT/P99TPOT/AverageITL/P99ITL/AverageTPS/TotalRequests` 等常见字段匹配；若报“指标映射为空”，先查看 summary 的字段名，必要时手工生成 normalized.json（schema 见第 6 节）。

### 4.4 GuideLLM

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

## 5. profile-service 部署记录与 benchmark-suite

使用 `profile-workflow-ops` skill 的 CLI（示例路径）：

```bash
S=$HOME/.agents/skills/profile-workflow-ops/scripts/profile_ops.py
# 若已安装到项目内，则改为 .kilo/skills/profile-workflow-ops/scripts/profile_ops.py
U=http://127.0.0.1:18729
python3 $S health --base-url $U
```

> 部署配置取 **服务端执行记录**；benchmark-suite 的 `config.json` 是压测客户端配置（backend/tokenizer/base_url），不能当服务端配置。

### 5.1 vllm-deploy（单机部署，如 tp8）

1. 找部署执行：`executions --json` 或 `execution-get <id>`，确认 `workflow_name=vllm-deploy`、节点、时间。
2. 直接把执行 id 交给 `import_kit.py`（`--deploy-execution`）即可自动归档启动命令；需要人工核对时再按下面拉取：

```bash
python3 $S logs --execution <exec_id> --base-url $U --list
python3 $S logs --execution <exec_id> --file commands/vllm_deploy.container_ensure.executed.sh --base-url $U
python3 $S logs --execution <exec_id> --file commands/vllm_deploy.model_start.sh --base-url $U
```

从输出提取 `docker run`、`vllm serve`、`-e K=V`、镜像，填入 `--docker-cmd/--serve-cmd/--env/--image`。

### 5.2 tp16-cluster + tp16-serve（多机 Ray）

- `tp16-cluster`：Ray head/worker 启动命令，通常包含 `ray start --head --address ...`。
- `tp16-serve`：head 节点容器内的 `vllm serve` 与 `docker run`，以及 `compilation-config`、`long-prefill-token-threshold` 等关键参数。
- 执行记录里的 `commands/*.executed.sh`（或 `tp16.serve_start.executed.sh` / `tp16.container_ensure.executed.sh` / `tp16.ray_start.executed.sh`）就是实际命令，作为 `config.deploy` 的来源，`source` 写 `profile-service <node> · <workflow> · <execution>`。

### 5.3 benchmark-suite 结果

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

转换（服务端配置来自 5.1/5.2；`--deploy-execution` 会把容器/vLLM/Ray 启动命令归档到 `<out>/server-start/`）：

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
  --execution <bench_exec_id> \
  --deploy-execution <serve_exec_id> --deploy-execution <cluster_exec_id> \
  --service-url http://172.31.8.31:18729 --service-url http://172.31.8.32:18729 \
  --source 'profile-service 172.31.8.31 · benchmark-suite · <exec_id>'
```

说明：脚本只导入 `status=success` 的 case；失败用例不入库，但它们的 `stdout.log`/`stderr.log` 仍随归档上传。
`cache_hit_rate` 仅当提供 `requests.jsonl` 时按 `sum(cached_tokens)/sum(prompt_tokens)` 计算。

## 6. 手工 normalized.json（适配器无法覆盖时）

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
    "description": "并发 64 · 输入 64000 · 输出 1024",
    "repro_cmd": "python3 benchmark_serving.py ...   # 可选：该 case 的复现命令",
    "console_log": "c64/console.log   # 可选：相对导入根目录，run 详情页按 case 展示",
    "stdout_log": "c64/stdout.log    # 可选：evalscope / 客户端标准输出",
    "stderr_log": "c64/stderr.log    # 可选：evalscope / 客户端标准错误"
  }]
}
```

manifest 与 normalized 的 `model/config/suite_key/node` 保持一致；manifest 额外支持 `scenario_id`、
`config.deploy`、run 级 `repro_cmd`、`cases.<case_key>.repro_cmd`（复现命令）与
`cases.<case_key>.console_log` / `stdout_log` / `stderr_log`（该 case 的日志路径）。

全手工流程（从原始产物手算指标、手写 manifest.json + normalized.json、curl 直接调导入接口，不走工作流也不走 `import_kit.py`）另见 profile 仓库的 `docs/perf-viewer-manual-import.md`。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `503` / 写接口失败 | `.env` 未配置 `PERF_VIEWER_TOKEN`，或请求未带 Bearer token |
| `duplicate` | 同配置指纹 + 同 run 特征已存在；确认是否确实重复 |
| `node_label` 缺失 | 手工指定 `--node`；CSV/探针/guidellm 可从产物推导 |
| 指标映射为空 | 检查 summary 字段名；用第 6 节手工 normalized.json |
| 配置被拆成两套 | 检查 `image` 是否带 registry/`_with_ray`；用 `--image vllm_glm:<tag>` |
| 没有归档启动命令 | 加 `--deploy-execution <部署执行 id>`（多机再给对端 `--service-url`），或用 `--deploy-dir` 归档已下载的目录 |
| 归档后 run 详情页看不到失败用例日志 | 失败用例本身不入库，日志在 run 产物 `benchmark-suite/case-*/` 下；要进指标表需加 `--include-failed` |
| 归档文件太大 | 加 `--no-archive-raw` 只上传指标，或精简 case 目录后重跑转换 |
| 失败 case 没导入 | benchmark-suite 默认只导入 success；调试可加 `--include-failed` |
| 测试记录没有日志展开 | benchmark-suite 保留 `case-*/stdout.log`、`stderr.log` 与 `case.json` 后重新转换；generic-json 补 `cases[].stdout_log` / `stderr_log` / `console_log`，确保文件随本次导入上传 |
| 想挂报告/产物 | 传 `--scenario-id`，未消费文件会挂到场景 |
