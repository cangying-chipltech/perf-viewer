"""进程内对比缓存：写操作（非 GET /api 成功响应）后整体失效。

适用于单 worker 部署（start.sh / systemd 默认）。多 worker 时各进程缓存独立，
写操作仅失效处理该请求的进程；如需强一致可扩展为共享存储版本号。
"""

_cache: dict = {}


def bump_generation() -> None:
    _cache.clear()


def cache_get(key):
    return _cache.get(key)


def cache_set(key, value) -> None:
    _cache[key] = value