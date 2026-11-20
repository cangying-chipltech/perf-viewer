import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_ENV_FILE = BASE_DIR / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        if _line.startswith("export "):
            _line = _line[len("export "):].lstrip()
        _key, _, _value = _line.partition("=")
        _key = _key.strip()
        _value = _value.strip()
        if len(_value) >= 2 and _value[0] == _value[-1] and _value[0] in ("'", '"'):
            _value = _value[1:-1]
        os.environ.setdefault(_key, _value)

DATA_DIR = Path(os.environ.get("PERF_VIEWER_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = DATA_DIR / "perf_viewer.sqlite3"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

TOKEN = os.environ.get("PERF_VIEWER_TOKEN", "")
HOST = os.environ.get("PERF_VIEWER_HOST", "0.0.0.0")
PORT = int(os.environ.get("PERF_VIEWER_PORT", "8600"))

MAX_UPLOAD_BYTES = int(os.environ.get("PERF_VIEWER_MAX_UPLOAD_MB", "512")) * 1024 * 1024

DEFAULT_SCORING = {
    "w_same": 2.0,
    "w_adj": 2.0,
    "w_best": 1.0,
    "epsilon": 0.01,
    "metric_weights": {},
    "node_evidence_full": 3.0,
}