import hmac
import re

import nh3
from fastapi import Header, HTTPException

from . import config

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?key|private[_-]?key|secret|passw(?:or)?d|pwd|"
    r"authorization|credential|(?:^|[_-])token$|(?:^|[_-])auth(?:[_-]|$))",
    re.I,
)


def redact(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEY_RE.search(k):
                out[k] = v if v in (None, "") else "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


_CMD_SECRET_FLAG_RE = re.compile(
    r"(--(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|pwd|token|authorization|credential))"
    r"(\s+|=)(?!\*\*\*)(\S+)",
    re.I,
)
_ENV_ASSIGN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(?!\*\*\*)(\S+)")


def redact_cmd(text: str) -> str:
    """命令字符串内的敏感值脱敏：--api-key xxx 与 KEY=VALUE 环境变量赋值。"""
    if not isinstance(text, str):
        return text
    text = _CMD_SECRET_FLAG_RE.sub(r"\1\2***", text)

    def _env_sub(m):
        key = m.group(1)
        return f"{key}=***" if _SECRET_KEY_RE.search(key) else m.group(0)

    return _ENV_ASSIGN_RE.sub(_env_sub, text)


def sanitize_html(html: str) -> str:
    return nh3.clean(html)


def check_token(authorization: str | None = Header(default=None)) -> None:
    if not config.TOKEN:
        raise HTTPException(status_code=503, detail="写操作未启用：服务端未配置 PERF_VIEWER_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="需要 Bearer token")
    supplied = authorization[len("Bearer "):]
    if not hmac.compare_digest(supplied, config.TOKEN):
        raise HTTPException(status_code=403, detail="token 无效")