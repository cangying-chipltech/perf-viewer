CREATE TABLE IF NOT EXISTS model (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    aliases TEXT NOT NULL DEFAULT '[]',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS suite (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS config_set (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES model(id),
    name TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    fingerprint TEXT NOT NULL,
    notes TEXT,
    UNIQUE(model_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS scenario (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES model(id),
    name TEXT NOT NULL,
    description TEXT,
    best_config_id INTEGER REFERENCES config_set(id),
    best_status TEXT CHECK(best_status IN ('recommended','confirmed','overridden')),
    best_reason TEXT,
    best_updated_at TEXT,
    scoring TEXT NOT NULL DEFAULT '{}',
    UNIQUE(model_id, name)
);

CREATE TABLE IF NOT EXISTS scenario_suite (
    scenario_id INTEGER NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
    suite_id INTEGER NOT NULL REFERENCES suite(id) ON DELETE CASCADE,
    PRIMARY KEY (scenario_id, suite_id)
);

CREATE TABLE IF NOT EXISTS test_case (
    id INTEGER PRIMARY KEY,
    suite_id INTEGER NOT NULL REFERENCES suite(id) ON DELETE CASCADE,
    case_key TEXT NOT NULL,
    description TEXT,
    params TEXT NOT NULL DEFAULT '{}',
    weight REAL NOT NULL DEFAULT 1,
    sort_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE(suite_id, case_key)
);

CREATE TABLE IF NOT EXISTS metric_def (
    key TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    unit TEXT,
    direction TEXT NOT NULL CHECK(direction IN ('lower','higher'))
);

CREATE TABLE IF NOT EXISTS node (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS import_batch (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT,
    manifest TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    report TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES model(id),
    config_id INTEGER NOT NULL REFERENCES config_set(id),
    node_id INTEGER NOT NULL REFERENCES node(id),
    suite_id INTEGER NOT NULL REFERENCES suite(id),
    backend TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    worker_label TEXT,
    valid INTEGER NOT NULL DEFAULT 1,
    excluded INTEGER NOT NULL DEFAULT 0,
    manifest TEXT NOT NULL DEFAULT '{}',
    artifacts_dir TEXT,
    batch_id INTEGER REFERENCES import_batch(id)
);

CREATE TABLE IF NOT EXISTS case_result (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    case_id INTEGER NOT NULL REFERENCES test_case(id),
    n_measured INTEGER,
    err_count INTEGER,
    metrics TEXT NOT NULL DEFAULT '{}',
    in_tok INTEGER,
    out_tok INTEGER,
    UNIQUE(run_id, case_id)
);

CREATE TABLE IF NOT EXISTS artifact (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES run(id) ON DELETE CASCADE,
    scenario_id INTEGER REFERENCES scenario(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('md','png','csv','log','json')),
    rel_path TEXT NOT NULL,
    title TEXT
);

CREATE TABLE IF NOT EXISTS best_history (
    id INTEGER PRIMARY KEY,
    scenario_id INTEGER NOT NULL REFERENCES scenario(id) ON DELETE CASCADE,
    prev_config_id INTEGER,
    new_config_id INTEGER,
    status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_model_suite ON run(model_id, suite_id);
CREATE INDEX IF NOT EXISTS idx_run_config ON run(config_id);
CREATE INDEX IF NOT EXISTS idx_run_node ON run(node_id);
CREATE INDEX IF NOT EXISTS idx_case_result_case ON case_result(case_id);
CREATE INDEX IF NOT EXISTS idx_case_result_run ON case_result(run_id);
CREATE INDEX IF NOT EXISTS idx_artifact_run ON artifact(run_id);
CREATE INDEX IF NOT EXISTS idx_artifact_scenario ON artifact(scenario_id);
CREATE INDEX IF NOT EXISTS idx_scenario_model ON scenario(model_id);
CREATE INDEX IF NOT EXISTS idx_config_model ON config_set(model_id);