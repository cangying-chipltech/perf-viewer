"""perf-viewer CLI。

perf-viewer import <dir|zip> --manifest m.json [--dry-run] [--base-url URL] [--token TOKEN]
perf-viewer seed [--assets-dir DIR]
perf-viewer serve [--host HOST] [--port PORT]

base-url/token 缺省从环境变量 PERF_VIEWER_BASE_URL / PERF_VIEWER_TOKEN 读取。
"""

import argparse
import io
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path


def _multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    buf = io.BytesIO()
    for name, value in fields:
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.write(f"{value}\r\n".encode())
    for name, filename, content in files:
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        buf.write(content)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


def _zip_dir(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, str(path.relative_to(directory)))
    return buf.getvalue()


def cmd_import(args) -> int:
    source = Path(args.source)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"manifest 不存在: {manifest_path}", file=sys.stderr)
        return 2
    manifest_bytes = manifest_path.read_bytes()
    json.loads(manifest_bytes)

    if source.is_dir():
        payload = _zip_dir(source)
        filename = source.name + ".zip"
    elif source.is_file() and source.suffix.lower() == ".zip":
        payload = source.read_bytes()
        filename = source.name
    else:
        print(f"导入源必须是目录或 zip: {source}", file=sys.stderr)
        return 2

    base_url = (args.base_url or os.environ.get("PERF_VIEWER_BASE_URL")
                or "http://127.0.0.1:8600").rstrip("/")
    token = args.token or os.environ.get("PERF_VIEWER_TOKEN", "")
    body, content_type = _multipart(
        [("dry_run", "true" if args.dry_run else "false")],
        [("manifest", "manifest.json", manifest_bytes), ("files", filename, payload)],
    )
    req = urllib.request.Request(
        base_url + "/api/import", data=body, method="POST",
        headers={"Content-Type": content_type,
                 **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            report = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"导入失败 HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"无法连接 {base_url}: {exc.reason}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in ("ok", "partial") else 1


def cmd_seed(args) -> int:
    from .seed import main as seed_main
    seed_main(["--assets-dir", args.assets_dir])
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from . import config
    uvicorn.run("app.main:app", host=args.host or config.HOST,
                port=args.port or config.PORT, log_level="info")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="perf-viewer", description="推理压测结果汇总呈现服务")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="导入目录或 zip（打包调用 HTTP API）")
    p_import.add_argument("source")
    p_import.add_argument("--manifest", required=True)
    p_import.add_argument("--dry-run", action="store_true")
    p_import.add_argument("--base-url", default=None)
    p_import.add_argument("--token", default=None)
    p_import.set_defaults(handler=cmd_import)

    p_seed = sub.add_parser("seed", help="初始目录种子 + 验收数据导入")
    p_seed.add_argument("--assets-dir", default="/home/cangying")
    p_seed.set_defaults(handler=cmd_seed)

    p_serve = sub.add_parser("serve", help="启动 Web 服务")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(handler=cmd_serve)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())