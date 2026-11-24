import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA_VERSION = 7

MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE run ADD COLUMN signature TEXT",
        "CREATE INDEX IF NOT EXISTS idx_run_signature ON run(model_id, suite_id, signature)",
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
    ],
    3: [
        "ALTER TABLE config_set ADD COLUMN deploy TEXT NOT NULL DEFAULT '{}'",
    ],
    4: [
        "ALTER TABLE run ADD COLUMN scenario_id INTEGER REFERENCES scenario(id) ON DELETE SET NULL",
        "CREATE INDEX IF NOT EXISTS idx_run_scenario ON run(scenario_id)",
    ],
    5: [],
    6: [
        "ALTER TABLE scenario ADD COLUMN source_scenario_ids TEXT NOT NULL DEFAULT '[]'",
    ],
    7: [
        "ALTER TABLE scenario ADD COLUMN config_ids TEXT NOT NULL DEFAULT '[]'",
    ],
}


def _backfill_run_scenarios(conn) -> None:
    """把历史 run 的 manifest.scenario_id 回填到 run.scenario_id。

    回填后仍为 NULL 的 run 不参与任何场景的对比；若其 suite 只被一个场景引用，
    由 assign_legacy_scenarios 兜底归属，避免唯一归属的数据被隐藏。
    """
    import json

    rows = conn.execute(
        "SELECT id, model_id, manifest FROM run WHERE scenario_id IS NULL").fetchall()
    for row in rows:
        try:
            manifest = json.loads(row["manifest"] or "{}")
            imported = manifest.get("import_manifest") or {}
            scenario_id = imported.get("scenario_id")
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
        if not isinstance(scenario_id, int):
            continue
        scenario = conn.execute(
            "SELECT model_id FROM scenario WHERE id = ?", (scenario_id,)).fetchone()
        if scenario is None or scenario["model_id"] != row["model_id"]:
            continue
        conn.execute("UPDATE run SET scenario_id = ? WHERE id = ?",
                     (scenario_id, row["id"]))


def assign_legacy_scenarios(conn) -> None:
    """把未归属 run 归到唯一引用其 suite 的场景；多场景共用 suite 时保持未归属。"""
    rows = conn.execute(
        "SELECT r.id,"
        " (SELECT COUNT(DISTINCT ss.scenario_id) FROM scenario_suite ss"
        "  JOIN scenario s ON s.id = ss.scenario_id"
        "  WHERE ss.suite_id = r.suite_id AND s.model_id = r.model_id) AS n,"
        " (SELECT MIN(ss.scenario_id) FROM scenario_suite ss"
        "  JOIN scenario s ON s.id = ss.scenario_id"
        "  WHERE ss.suite_id = r.suite_id AND s.model_id = r.model_id) AS scenario_id"
        " FROM run r WHERE r.scenario_id IS NULL").fetchall()
    for row in rows:
        if row["n"] == 1 and row["scenario_id"] is not None:
            conn.execute("UPDATE run SET scenario_id = ? WHERE id = ?",
                         (row["scenario_id"], row["id"]))


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
            _backfill_run_scenarios(conn)
            assign_legacy_scenarios(conn)
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