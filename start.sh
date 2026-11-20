#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

need_install=0
if [ ! -x .venv/bin/python ]; then
  need_install=1
else
  .venv/bin/python -c "import fastapi, uvicorn, jinja2, multipart, markdown, nh3" >/dev/null 2>&1 || need_install=1
fi

if [ "$need_install" = "1" ]; then
  if command -v uv >/dev/null 2>&1; then
    [ -x .venv/bin/python ] || uv venv .venv --python 3.10
    uv pip install --python .venv/bin/python -e .
  else
    [ -x .venv/bin/python ] || python3 -m venv .venv
    .venv/bin/pip install -e .
  fi
fi

if [ -f .env ]; then
  chmod 600 .env
else
  echo "提示：未找到 .env（写操作 token）。浏览可用；写操作需配置 PERF_VIEWER_TOKEN。" >&2
fi

exec .venv/bin/python -m app.cli serve