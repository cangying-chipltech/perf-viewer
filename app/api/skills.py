"""技能在线查询与下载：读取项目 .kilo/skills 下的技能目录。"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from .. import config

router = APIRouter(prefix="/api")

SKILLS_DIR = Path(os.environ.get("PERF_VIEWER_SKILLS_DIR", str(config.BASE_DIR / ".kilo" / "skills")))

_EXCLUDE_DIRS = {"__pycache__", ".git"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _EXCLUDE_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix in _EXCLUDE_SUFFIXES or path.name == ".DS_Store":
            continue
        yield path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    meta = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        meta[key.strip()] = value
    return meta


def _skill_dirs() -> list[Path]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p for p in SKILLS_DIR.iterdir()
                  if p.is_dir() and (p / "SKILL.md").is_file())


def _skill_manifest(root: Path) -> dict:
    skill_md = root / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    meta = _parse_frontmatter(text)
    name = (meta.get("name") or root.name).strip()
    files = []
    for path in _iter_files(root):
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": len(data),
            "sha256": _sha256_bytes(data),
        })
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "name": name,
        "dir": root.name,
        "version": meta.get("version") or "1.0.0",
        "description": meta.get("description", ""),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "checksum": "sha256:" + digest.hexdigest(),
        "download": f"/api/skill/package?name={quote(name)}",
    }


def _find_skill(name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(400, "非法技能名")
    for root in _skill_dirs():
        manifest = _skill_manifest(root)
        if manifest["name"] == name or root.name == name:
            return root
    raise HTTPException(404, "技能不存在")


def _zip_skill(root: Path, manifest: dict) -> bytes:
    buffer = io.BytesIO()
    base = manifest["name"] or root.name
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in _iter_files(root):
            rel = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{base}/{rel}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return buffer.getvalue()


@router.get("/skill")
def api_skill_list(name: str | None = Query(default=None, description="按技能名查询")):
    if name:
        root = _find_skill(name)
        manifest = _skill_manifest(root)
        manifest["skill_md"] = (root / "SKILL.md").read_text(encoding="utf-8")
        return manifest
    return {"skills": [_skill_manifest(root) for root in _skill_dirs()]}


@router.get("/skill/package")
def api_skill_package(name: str = Query(..., description="技能名")):
    root = _find_skill(name)
    manifest = _skill_manifest(root)
    data = _zip_skill(root, manifest)
    digest = _sha256_bytes(data)
    headers = {
        "Content-Disposition": f'attachment; filename="{manifest["name"]}.zip"',
        "X-Checksum-SHA256": digest,
        "X-Skill-Checksum": manifest["checksum"],
    }
    return Response(content=data, media_type="application/zip", headers=headers)


@router.get("/skill/{name}")
def api_skill_detail(name: str):
    root = _find_skill(name)
    manifest = _skill_manifest(root)
    manifest["skill_md"] = (root / "SKILL.md").read_text(encoding="utf-8")
    return manifest
