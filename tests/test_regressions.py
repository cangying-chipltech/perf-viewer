import json
import shutil

import pytest

from app import config
from app.db import get_db
from app.entities import config_fingerprint
from app.importer import import_runs
from app.scoring import load_comparison
from app.security import redact, sanitize_html
from app.web import render_markdown
from conftest import FIXTURES, make_catalog, make_run, make_scenario

CSV_MANIFEST = {
    "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
    "config": {"name": "tp8", "params": {"tp": 8}},
    "suite_key": "glm53-12case",
    "backend": "node-csv",
}


def stage(tmp_path, *names):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(FIXTURES / name, root / name)
    return root


class TestRedaction:
    def test_masks_credential_keys(self):
        data = {
            "api_key": "s", "password": "s", "passwd": "s", "pwd": "s",
            "access_key": "s", "private_key": "s", "auth_token": "s",
            "extra_token": "s", "token": "s", "authorization": "s",
            "credential": "s", "client_secret": "s", "auth": "s",
            "numeric_secret": 1, "api_key_num": 12345,
        }
        out = redact(data)
        for k in ("api_key", "password", "passwd", "pwd", "access_key", "private_key",
                  "auth_token", "extra_token", "token", "authorization", "credential",
                  "client_secret", "auth"):
            assert out[k] == "***", k
        assert out["api_key_num"] == "***"

    def test_preserves_perf_keys(self):
        data = {
            "long_prefill_token_threshold": 2048,
            "num_speculative_tokens": 5,
            "prompt_tokens": 1000026,
            "input_target_tokens": 1000000,
            "output_limit_tokens": 1,
            "cached_tokens": 0,
            "max_num_batched_tokens": 4096,
            "completion_tokens": 1,
        }
        out = redact(data)
        assert out == data

    def test_threshold_configs_still_split(self):
        base = {"num_gpu_blocks_override": 1024, "block_size": 256}
        a = redact(dict(base, long_prefill_token_threshold=1024))
        b = redact(dict(base, long_prefill_token_threshold=2048))
        assert a["long_prefill_token_threshold"] == 1024
        assert b["long_prefill_token_threshold"] == 2048
        assert config_fingerprint(a) != config_fingerprint(b)


class TestSanitize:
    def test_strips_script_and_handlers(self):
        html = sanitize_html('<p>ok</p><script>alert(1)</script>'
                             '<img src=x onerror="alert(2)">'
                             '<a href="javascript:alert(3)">x</a>')
        assert "<script>" not in html
        assert "onerror" not in html
        assert "javascript:" not in html
        assert "<p>ok</p>" in html

    def test_render_markdown_sanitizes(self):
        out = render_markdown("# T\n\n<script>alert(1)</script>\n\n| a | b |\n|---|---|\n| 1 | 2 |", [])
        assert "<script>" not in out
        assert "<table>" in out


class TestFingerprint:
    def test_int_float_string_equal(self):
        assert config_fingerprint({"tp": 8}) == config_fingerprint({"tp": 8.0})
        assert config_fingerprint({"tp": 8}) == config_fingerprint({"tp": "8"})
        assert config_fingerprint({"tp": "8.0"}) == config_fingerprint({"tp": 8})

    def test_negative_flag_value(self):
        assert config_fingerprint("--foo -1") == config_fingerprint("--foo=-1")
        assert config_fingerprint("--foo -1.5") == config_fingerprint("--foo=-1.5")


class TestScoringGuards:
    def test_case_weight_zero_excluded(self, conn):
        model, suite, cases = make_catalog(conn, case_keys=("c1", "c2"))
        sid = make_scenario(conn, model, suite)
        conn.execute("UPDATE test_case SET weight = 0 WHERE id = ?", (cases[1]["id"],))
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1",
                 [{"ttft_mean_s": 10}, {"ttft_mean_s": 30}])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1",
                 [{"ttft_mean_s": 20}, {"ttft_mean_s": 20}])
        comp = load_comparison(conn, sid)
        a = next(r for r in comp["suites"][0]["ranking"] if r["name"] == "A")
        assert a["total_score"] == 3.0

    def test_metric_weight_zero_no_crash(self, conn):
        model, suite, cases = make_catalog(conn, case_keys=("c1",))
        sid = make_scenario(conn, model, suite)
        conn.execute("UPDATE scenario SET scoring = ? WHERE id = ?",
                     (json.dumps({"metric_weights": {"ttft_mean_s": 0}}), sid))
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [{"ttft_mean_s": 10}])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [{"ttft_mean_s": 20}])
        comp = load_comparison(conn, sid)
        assert comp["suites"][0]["cases"][0]["pairwise"] == []


class TestImportRegressions:
    def test_alias_reimport_idempotent(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        import_runs(root, CSV_MANIFEST)
        alias_manifest = json.loads(json.dumps(CSV_MANIFEST))
        alias_manifest["model"] = {"name": "GLM-5.3-MTP-Mix"}
        report = import_runs(root, alias_manifest)
        assert all(r["status"] == "duplicate" for r in report["runs"])
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 4

    def test_duplicate_csv_rows_isolated(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "x.csv").write_text(
            "node,case,n_measured,err,e2e_mean_s,out_tps\n"
            "n1,c1,2,0,1,5\n"
            "n1,c1,2,0,1,5\n"
            "n2,c1,2,0,2,6\n", encoding="utf-8")
        report = import_runs(root, dict(CSV_MANIFEST, suite_key="dup-suite"))
        statuses = {r["node"]: r["status"] for r in report["runs"]}
        assert statuses["n1"] == "failed"
        assert statuses["n2"] == "imported"
        assert report["status"] == "partial"
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 1

    def test_reimport_no_duplicate_artifacts(self, tmp_path):
        with get_db() as conn:
            from app.entities import ensure_suite, resolve_model
            model = resolve_model(conn, "cltech/glm53")
            suite = ensure_suite(conn, "glm53-12case")
            sid = make_scenario(conn, model, suite)
        root = stage(tmp_path, "node_details.csv")
        shutil.copyfile(FIXTURES / "probe_threshold1024.json", root / "extra.json")
        manifest = dict(CSV_MANIFEST, scenario_id=sid)
        import_runs(root, manifest)
        with get_db() as conn:
            before = conn.execute("SELECT COUNT(*) n FROM artifact").fetchone()["n"]
        report = import_runs(root, manifest)
        assert all(r["status"] == "duplicate" for r in report["runs"])
        with get_db() as conn:
            after = conn.execute("SELECT COUNT(*) n FROM artifact").fetchone()["n"]
        assert after == before

    def test_generic_json_config_secret_redacted(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        data = {
            "model_name": "cltech/glm53", "config_name": "gcfg",
            "config_params": {"tp": 8, "api_key": "sk-leak"},
            "node_label": "n1", "suite_key": "g-suite",
            "cases": [{"case_key": "c1", "metrics": {"ttft_mean_s": 1.0},
                       "params": {"password": "p", "input_tokens": 100}}],
        }
        (root / "run.json").write_text(json.dumps(data), encoding="utf-8")
        manifest = {"model": {"name": "cltech/glm53"},
                    "config": {"name": "m", "params": {"x": 1}},
                    "suite_key": "g-suite", "backend": "generic-json"}
        report = import_runs(root, manifest)
        assert report["status"] == "ok"
        with get_db() as conn:
            cfg = conn.execute("SELECT * FROM config_set WHERE name = 'gcfg'").fetchone()
            assert json.loads(cfg["params"])["api_key"] == "***"
            case = conn.execute("SELECT * FROM test_case WHERE case_key = 'c1'").fetchone()
            cp = json.loads(case["params"])
            assert cp["password"] == "***"
            assert cp["input_tokens"] == 100


class TestApiRegressions:
    def test_create_model_alias_conflict(self, client):
        headers = {"Authorization": "Bearer test-token"}
        client.post("/api/catalog/models", json={"name": "m1", "aliases": ["a1"]},
                    headers=headers)
        resp = client.post("/api/catalog/models", json={"name": "a1"}, headers=headers)
        assert resp.status_code == 409

    def test_svg_served_as_attachment(self, client):
        art_dir = config.ARTIFACTS_DIR / "999"
        art_dir.mkdir(parents=True)
        (art_dir / "x.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg">'
                                       '<script>alert(1)</script></svg>', encoding="utf-8")
        with get_db() as conn:
            cur = conn.execute("INSERT INTO artifact(run_id, kind, rel_path, title)"
                               " VALUES(NULL, 'png', '999/x.svg', 'x.svg')")
            art_id = cur.lastrowid
        resp = client.get(f"/api/artifacts/{art_id}")
        assert resp.status_code == 200
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_last_updated_utc(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(conn)
            make_run(conn, model, suite, cases, "A", {"v": 1}, "n1",
                     [{"ttft_mean_s": 1}], started_at="2026-11-17T16:42:12+08:00")
            make_run(conn, model, suite, cases, "B", {"v": 2}, "n1",
                     [{"ttft_mean_s": 2}], started_at="2026-11-17T09:00:00+00:00")
        models = client.get("/api/models").json()["models"]
        assert models[0]["last_updated"] == "2026-11-17T09:00:00+00:00"