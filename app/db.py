import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA_VERSION = 3

MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE run ADD COLUMN signature TEXT",
        "CREATE INDEX IF NOT EXISTS idx_run_signature ON run(model_id, suite_id, signature)",
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
    ],
    3: [
        "ALTER TABLE config_set ADD COLUMN deploy TEXT NOT NULL DEFAULT '{}'",
    ],
}


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def get_db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect(config.DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_db_dryrun():
    """与 get_db 相同，但结束时始终回滚，用于 --dry-run 不落库。"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect(config.DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _apply_schema(conn, schema_sql: str) -> None:
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    conn = _connect(config.DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < SCHEMA_VERSION:
            conn.execute("BEGIN IMMEDIATE")
            if version < 1:
                _apply_schema(conn, schema)
                version = 1
            for v in range(version + 1, SCHEMA_VERSION + 1):
                for stmt in MIGRATIONS[v]:
                    conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("COMMIT")
        from .entities import seed_metric_defs
        seed_metric_defs(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()