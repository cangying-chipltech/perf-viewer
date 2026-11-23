# perf-viewer

以**模型**为索引汇总推理压测结果，按「交付场景 → 配置集 → 设备×多次运行 → 测试集合(case)」层级呈现汇总报告。

技术栈：**Python FastAPI + SQLite(WAL) + Jinja2 + htmx + ECharts**，无前端构建步骤。

## 快速开始

```bash
cd /home/cangying/perf-viewer
./start.sh                      # 首次自动建 venv 并安装依赖，监听 0.0.0.0:8600
```

或手动：

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m app.seed    # 目录种子 + 验收数据导入（幂等）
.venv/bin/python -m app.cli serve
```

打开 http://127.0.0.1:8600 。浏览免认证；写操作需在右上角填入 token（见「安全」）。

## 配置（环境变量 / `.env`）

服务启动时自动读取项目根目录 `.env`（`KEY=VALUE`，支持引号与 `export ` 前缀；**已存在的真实环境变量优先于 `.env`**）。`.env` 含写 token，权限应为 600（`start.sh` 会自动收紧）。

| 变量 | 默认 | 说明 |
|---|---|---|
| `PERF_VIEWER_TOKEN` | 空 | 写操作 Bearer token。**为空时所有写 API 返回 503** |
| `PERF_VIEWER_DATA_DIR` | `<项目>/data` | SQLite 与产物根目录 |
| `PERF_VIEWER_HOST` | `0.0.0.0` | 监听地址 |
| `PERF_VIEWER_PORT` | `8600` | 监听端口 |
| `PERF_VIEWER_MAX_UPLOAD_MB` | `512` | 单次导入上传总量上限 |
| `PERF_VIEWER_BASE_URL` | `http://127.0.0.1:8600` | CLI 目标服务地址 |
| `PERF_VIEWER_SKILLS_DIR` | `.kilo/skills` | 技能查询/下载目录 |

数据布局：`data/perf_viewer.sqlite3`（WAL）、`data/artifacts/<batch_id>/<相对路径>`。

## 层级与实体

- **模型** `model`：以 served name 为主键，支持别名（如 `cltech/glm53` ↔ `GLM-5.3-MTP-Mix`）。
- **场景** `scenario`：交付/调优目标，引用若干测试集合；持有最佳配置与评分参数。
- **测试集合** `suite`：命名 case 矩阵，可被多场景复用；`test_case` 含 params/weight。
- **配置集** `config_set`：名称 + 参数 JSON，按**归一化参数指纹(sha256)** 去重自动新建，可人工合并/重命名。另持**部署信息** `deploy`（docker 启动命令 / 模型环境变量 / 模型启动命令 / 来源），不参与指纹。
- **节点** `node`：hostname/IP 标签。
- **运行** `run`：模型×配置×节点×suite 的一次压测；`valid`/`excluded` 控制是否参与对比。
- **case 结果** `case_result`：run 内逐 case 的归一化指标。

**未声明配置的导入拒绝入库**（manifest 必须含 `config{name,params}`）。

### 配置指纹归一化

- serve 命令行字符串（`-tp 8 -dcp 8 --block-size 256 ...`）解析为**与顺序无关**的规范 dict；`--k=v`、布尔开关、位置参数均支持。
- 短 flag 别名归一（`tp→tensor_parallel_size`、`dcp→decode_context_parallel_size` 等），`-`→`_`。
- 数字字符串归一为数字（`"8"`≡`8`）；dict 键排序。
- 因此「仅 `--long-prefill-token-threshold` 不同」的两套配置会正确拆分为两个指纹，而 flag 顺序不同的同一配置不会误拆分。

## 归一化指标

固定种子（`metric_def`）：`e2e_mean_s, e2e_p99_s, ttft_mean_s, ttft_p99_s, tpot_mean_ms, tpot_p99_ms, itl_mean_ms, itl_p99_ms, out_tps`（延迟类 lower、`out_tps` higher），可选 `cache_hit_rate`（higher）。缺失指标 = `null`，不参与对比，页面显示「—」。

- **探针映射**（n=1）：`ttft_s→ttft_mean_s`、`latency_s→e2e_mean_s`、`output_tok_s→out_tps`，p99=mean。`output_tok_s=0.0`（1-token 输出）是合法值，不当缺失。
- **node-csv ITL**：`itl_mean_ms ← itl_tok_mean_ms`（token 级，与报告口径一致），缺失回退 `itl_reqmean_ms`。`in_tok/out_tok` 为测量请求总量。
- **guidellm-suite**：仅统计测量窗口（successful，warmup 已排除）；`out_tps=输出token总量/E2E耗时之和`；`cache_hit_rate` **仅**取自 `server_metrics.json.prefix_cache_hit_rate`，缺失留空（不用成功请求比例代替）。

### 有效运行判定

`valid` = 探针 `ok&&complete` / GuideLLM `request_totals.errored==0&&incomplete==0&&total>0` / CSV `err==0`。代表值与评分仅纳入 `valid=1 且 excluded=0` 的运行。允许部分套件运行（只含 subset 的 case）。

## 对比与评分

对比以**配置**为主轴：对场景引用的每个 suite，逐 case×metric 做配置间两两比较，不要求每套配置跑满全部节点。

- **同节点比较**（高权重）：对每个双方都有数据的节点，比较节点代表值（该节点×配置×case 全部有效运行**均值**）。相对差 ≤ε 记平局 0.5，否则胜 1/负 0；`win_rate_same=得分和/可比节点数`。
- **跨节点校正比较**（无共同节点时的回退，权重 `w_adj`）：用「同一配置 × 同一 case 在不同节点」的运行估计各节点×metric 的 log 倍率（对数空间最小二乘，可复用其他 case/suite 的证据，见 `node_effects`），把双方节点代表值折算到同一基准后再判胜负。校正结果按配对证据量打折：`confidence = min(1, 配对 case 数 / node_evidence_full)`，间接路径证据再按跳数衰减；无证据时该部分权重为 0。`win_rate_adj` 为按置信度加权的校正胜率。
- **最优值比较**（低权重）：比较双方跨全部节点/运行的**历史最好值**（延迟 min、吞吐 max），同样 ε 判平。存在跨节点校正证据时，最优值按 `confidence` 与「校正后最优值」混合，避免只在快节点跑过的配置占便宜。
- `metric_score = w_same*win_rate_same + w_adj*confidence*win_rate_adj + w_best*win_rate_best`。有共同节点时用 same 项；无共同节点且有校正证据时用 adj 项；两者都没有时仅计 best 项。
- `case_score = Σ(metric_score*metric_weight)/Σmetric_weight`（仅双方有数据的 metric）。
- 配置总分 = 对各对手×case 的 case_score 按 case weight 加权平均（仅计有数据且有对手的 case）→ 排名，榜首为「推荐最佳」。

评分参数存于 `scenario.scoring`：`{w_same, w_adj, w_best, epsilon, metric_weights, node_evidence_full}`，默认 `{2,2,1,0.01,{},3}`，可在目录管理页编辑。页面提供「同节点比较 / 跨节点校正 / 最优值比较」三种视图；评分依据中同时展示节点倍率、配对证据量与校正置信度。

**最佳配置** = 自动加权评分推荐 + 人工确认/改判（附理由），展示每层得分依据（胜/负/平、n、范围、来源 run）。改判必填理由，写入 `best_history`。

**数据不足友好提示**：单配置场景显示「待补充对比配置」，无共同节点/无共同 case 时给出说明，不报错。现有 benchmark-suite 数据中两套配置分别只跑过不同节点，且缺少可连通的同配置跨节点证据时，会退化为最优值比较并在页面标注「无校正证据」。

### 部署信息（config.deploy）

最佳配置不仅给参数，还要能直接照着部署。`config_set.deploy` 为 JSON 对象：

```json
{
  "docker_cmd": "docker run -itd --name ... <image>",
  "env": {"VLLM_DLC_INDEX_TOPK": "256", "...": "..."},
  "serve_cmd": "vllm serve /data/models/... --served-model-name ... -tp 16 ...",
  "extra_cmds": [
    {"title": "head 192.168.80.4 · 容器内启动 Ray（--head）", "cmd": "docker exec ... ray start --head ..."},
    {"title": "worker 192.168.80.5 · 容器内加入 Ray 集群", "cmd": "docker exec ... ray start --address=192.168.80.4:6379"}
  ],
  "source": "profile-service <node> · <workflow> · <execution> · 日期"
}
```

- `extra_cmds`：附加启动/编排命令（如 tp16 双机两容器的 `ray start`），按标题逐条展示与复制；每项 `{title, cmd, note?}`，`cmd` 内敏感值同样脱敏。
- 来源：导入 manifest 的 `config.deploy`（`env` 支持对象、`K=V` 列表或多行/空格分隔文本）；目录管理页配置集内可视化编辑；种子数据内置三套配置的已取证命令（取自 profile-service 执行记录的 `commands/*.executed.sh` / `*.sh`，api-key 脱敏为 `***`）。
- 重复导入同指纹配置时：manifest 带 deploy 则更新，不带则保留原值；run 仍按特征判 duplicate。
- 敏感值脱敏：env 的敏感 key、命令内 `--api-key/--token/...` 与 `KEY=VALUE` 赋值均入库前替换为 `***`。
- 场景页「当前最佳配置」直接展示：部署信息（Docker 启动命令 + 模型环境变量 `export` 文本，不单独建表 + 模型启动命令 + 额外启动命令如 tp16 两容器 ray start，均可一键复制）与当前最佳数据；核心参数对比见下钻对比页。未人工确认时展示自动推荐配置的部署信息并明确标注。

## 导入

### manifest（必填字段）

```json
{
  "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
  "config": {"name": "tp8-...", "params": {"tensor_parallel_size": 8, "...": "..."}, "notes": "...",
             "deploy": {"docker_cmd": "...", "env": {"K": "V"}, "serve_cmd": "...", "source": "..."}},
  "suite_key": "glm53-12case",
  "backend": "node-csv",
  "node": "192.168.80.6",
  "scenario_id": 3,
  "case_key": "1m_in_1out",
  "cases": {"A_256": {"params": {...}, "description": "..."}},
  "started_at": "2026-11-17T15:34:09+08:00",
  "ended_at": "..."
}
```

- 必填：`model`、`config{name,params}`、`suite_key`。`backend` 可省略（自动探测）。`config.deploy` 可选（见「部署信息」）。
- `node` 可由产物推导（探针 `base_url`、CSV `node` 列、guidellm `meta.json`）；推导不出时必填。
- `suite` 不存在时按 manifest 自动创建 suite+cases；`cases` 可补充 case params/描述。
- `scenario_id` 存在时，未被适配器消费的额外文件（报告 md/png 等）挂到该场景。
- 时间戳保留时区（产物为 `+08:00` Asia/Shanghai，不做 naive 解析）。
- manifest/产物中的敏感 key（`api_key/token/secret/password/authorization/credential`）入库前脱敏为 `***`。

### 适配器（注册表显式登记，新增后端=新增适配器）

| backend | 输入 | 说明 |
|---|---|---|
| `guidellm-suite` | `meta.json + guidellm.json + server_metrics.json` 的 case 目录 | 完整套件 |
| `probe-json` | chat_probe / mixed_impact JSON | 按字段特征区分 |
| `node-csv` | 表头 `node,case,n_measured,err,e2e_mean_s,...` | 一文件多节点多 case → 每节点一个 run |
| `generic-json` | 直接符合 NormalizedRun schema | 单对象或数组 |
| `ssh-pull` / `profile-workflow` | — | **预留 stub**，仅接口占位，调用即报错 |

### HTTP API

`POST /api/import`（multipart，Bearer token）：`manifest`（manifest.json 文件）或 `manifest_json`（表单字段）+ `files`（若干文件或单个 zip）+ `dry_run`。返回 batch 报告（逐 run imported/duplicate/failed、实体登记、产物落盘）。

**幂等**：同配置指纹 + 同 run 特征（节点/suite/backend/case 数据）→ 报 `duplicate` 而非静默重复。run 特征不含 manifest 注入的时间戳，保证无时间产物（CSV）重复导入幂等。

### 技能 API

项目技能位于 `.kilo/skills/<dir>/SKILL.md`，可通过 API 在线查询与下载：

```bash
GET /api/skill                          # 列出全部技能
GET /api/skill?name=perf-viewer-import  # 查询技能详情（含 SKILL.md 与文件校验和）
GET /api/skill/{name}                   # 同上，路径形式
GET /api/skill/package?name=<name>      # 下载 zip（响应头 X-Checksum-SHA256）
```

`PERF_VIEWER_SKILLS_DIR` 可覆盖技能目录，默认 `.kilo/skills`。


### CLI

```bash
perf-viewer import <dir|zip> --manifest m.json [--dry-run] [--base-url URL] [--token TOKEN]
perf-viewer seed [--assets-dir DIR]
perf-viewer serve [--host HOST] [--port PORT]
```

`import` 将目录打包为 zip 调用 HTTP API；`--base-url/--token` 缺省从 `PERF_VIEWER_BASE_URL/PERF_VIEWER_TOKEN` 读取。

### 远程结果

GuideLLM 完整套件产物在远程 worker（`/data/llm-load-results`）。**v1 不做自动拉取**：先人工 `rsync` 回本机，再 CLI 导入。`ssh-pull`/`profile-workflow` 仅为预留接口。

```bash
rsync -av worker:/data/llm-load-results/ ./local-results/
perf-viewer import ./local-results --manifest manifest.json
```

## Docker

```bash
docker build -t perf-viewer:latest -t perf-viewer:0.1.0 .

docker run -d --name perf-viewer -p 8600:8600 \
  -e PERF_VIEWER_TOKEN='<写操作 token>' \
  -v perf-viewer-data:/app/data \
  perf-viewer:latest
```

- 镜像内含 `app/` 与 `.kilo/skills/`（技能 API 可直接查询/下载）。
- 数据持久化在 `/app/data`；使用 named volume 或把宿主机目录 chown 给 UID `10001`。
- `PERF_VIEWER_TOKEN` 通过环境变量注入，不要把 `.env` 打进镜像。
- 内置 healthcheck 调用 `/healthz`。

## 页面

- `/` 模型列表卡片（场景数、run 数、最近更新、各场景最佳配置摘要）。
- `/models/{name}` 模型页：**只展示交付场景**卡片（场景名/描述/suite/当前最佳配置）。
- `/models/{name}/scenarios/{id}` 场景页：**只展示最佳配置交付信息**——最佳配置标识 + 部署信息（Docker 启动命令 / 模型环境变量 export 文本 / 模型启动命令 / 额外启动命令如 tp16 两容器 ray start，均可一键复制）+ 当前最佳数据（最佳值/范围/n/覆盖节点/来源 run）+ 下钻对比链接（参数与部署对比在对比页）。
- `/models/{name}/scenarios/{id}/compare` 下钻对比页：**参数对比**（配置声明参数 ∪ serve 命令解析参数 ∪ Docker 容器启动参数 ∪ 模型环境变量，逐项一行并标注来源；默认只展示差异项，可切「全部参数」；配置多时可勾选参与对比的配置）、**部署信息对比**（Docker/env/serve/额外命令默认折叠，如 tp16 两容器 ray start）、「为什么是最佳 · 评分依据」逐对手/逐 case/逐指标下钻（同节点逐节点胜负、跨节点校正胜负与置信度、最优值对比、metric_score/weight、来源 run，以及节点倍率/配对证据表）、case×配置对比表（同节点/跨节点校正/最优值三模式 tab + 单元格下钻）、ECharts 柱状图、最佳配置确认/改判与历史、场景报告 md/png 渲染。
- `/runs/{id}` run 详情：case 指标表、manifest、有效性/剔除、产物（md 渲染 HTML、png 内联、csv/log/json 下载）。
- `/catalog` 目录编辑：场景/suite/case/模型/节点增改、**配置集重命名/备注/部署信息编辑与合并**、metric weights、ε、w_same/w_adj/w_best、node_evidence_full。

htmx/ECharts 已 vendor 到 `app/static/vendor/`；若 ECharts 缺失，图表降级为提示，表格不受影响。

## API 摘要

- 读：`GET /api/models`、`/api/models/{m}`、`/api/models/{m}/scenarios`、`/api/scenarios/{id}`、`/api/scenarios/{id}/comparison?suite=&mode=same-node|adjusted|best`、`/api/runs/{id}`、`/api/artifacts/{id}`、`/api/catalog`、`/api/batches/{id}`、`/healthz`
- 写（token）：`POST /api/import`、`POST /api/scenarios/{id}/best`、`POST /api/runs/{id}/exclude`、`POST /api/configs/merge`、目录 CRUD `/api/catalog/...`（`PATCH /api/catalog/configs/{id}` 支持 `deploy` 字段）

交互式文档：`/docs`。

## 安全

- 浏览开放；写操作需 `Authorization: Bearer <PERF_VIEWER_TOKEN>`，用 `hmac.compare_digest` 比较。token 未配置时写 API 返回 503。
- 产物路径限制在 `data/artifacts` 内（防穿越）；zip 解压拒绝绝对路径/`..`，限制成员数与解压总量。
- 敏感 key 脱敏后入库；部署命令内的 `--api-key/--token` 等 flag 值与敏感 `KEY=VALUE` 环境变量同样脱敏为 `***`。
- `.env` 含 token，已加入 `.gitignore`，勿提交。

## 部署

```bash
./start.sh                                   # 前台/脚本启动（自动校验/修复 venv，chmod 600 .env）
# 或 systemd user unit：
mkdir -p ~/.config/systemd/user
cp deploy/perf-viewer.service ~/.config/systemd/user/
loginctl enable-linger "$USER"               # 允许用户服务在未登录/开机时运行
systemctl --user daemon-reload
systemctl --user enable --now perf-viewer
systemctl --user status perf-viewer
```

> user unit 需 `enable-linger` 才会在开机/注销后持续运行；否则随最后一个会话退出。
> 如需开机即起且不受登录影响，也可改用 system unit（`User=cangying`、`WantedBy=multi-user.target`）。

## 测试与验证

```bash
.venv/bin/python -m pytest tests -q        # 适配器/指纹/评分/API 鉴权/路径穿越/导入幂等/部署信息与迁移
.venv/bin/python -m compileall app         # 语法检查
```

端到端验收（已内置种子数据）：启动服务 → `curl /healthz` → CLI 导入 CSV 与探针 JSON → 场景页对比表数值与手工计算一致（1M prefill：threshold 2048 优于 1024，TTFT 2236.4s vs 2411.6s，同节点得分 2.7 vs 0.3）→ 确认最佳配置 → 产物 md/png 正常渲染。

## 范围备注

- v1：推理压测数据（GuideLLM 套件、探针 JSON、节点明细 CSV）的导入、归一化、对比、最佳配置判定、在线呈现。
- 明确不在 v1：agent 基准数据（benchmark-profile-results）、md 报告导出、SSH/profile-service 主动拉取（仅预留适配器接口）。