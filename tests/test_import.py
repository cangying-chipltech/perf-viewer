import json
import shutil
import zipfile

import pytest

from app import config
from app.db import get_db
from app.importer import ImportError_, extract_zip, import_runs
from app.scoring import load_comparison
from conftest import FIXTURES, make_scenario


def stage(tmp_path, *names):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(FIXTURES / name, root / name)
    return root


CSV_MANIFEST = {
    "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
    "config": {"name": "tp8", "params": "-tp 8 -dcp 8 --block-size 256"},
    "suite_key": "glm53-12case",
    "backend": "node-csv",
}


class TestCsvImport:
    def test_import_creates_runs(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        report = import_runs(root, CSV_MANIFEST, source="test")
        assert report["status"] == "ok"
        assert len(report["runs"]) == 4
        assert all(r["status"] == "imported" for r in report["runs"])
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 4
            assert conn.execute("SELECT COUNT(*) n FROM case_result").fetchone()["n"] == 48
            assert conn.execute("SELECT COUNT(*) n FROM node").fetchone()["n"] == 4
            assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 1
            arts = conn.execute("SELECT * FROM artifact").fetchall()
            assert len(arts) == 4
            for art in arts:
                assert (config.ARTIFACTS_DIR / art["rel_path"]).is_file()

    def test_reimport_duplicate(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        import_runs(root, CSV_MANIFEST, source="test")
        report2 = import_runs(root, dict(CSV_MANIFEST, started_at="2026-11-18T00:00:00+08:00"),
                              source="test")
        assert all(r["status"] == "duplicate" for r in report2["runs"])
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 4

    def test_dry_run_no_persist(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        report = import_runs(root, CSV_MANIFEST, source="test", dry_run=True)
        assert report["dry_run"] is True
        assert all(r["status"] == "would-import" for r in report["runs"])
        assert report["batch_id"] is None
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 0
            assert conn.execute("SELECT COUNT(*) n FROM model").fetchone()["n"] == 0
            assert conn.execute("SELECT COUNT(*) n FROM import_batch").fetchone()["n"] == 0

    def test_missing_config_rejected(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = dict(CSV_MANIFEST)
        del manifest["config"]
        with pytest.raises(ImportError_, match="config"):
            import_runs(root, manifest)

    def test_missing_suite_key_rejected(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = {k: v for k, v in CSV_MANIFEST.items() if k != "suite_key"}
        with pytest.raises(ImportError_, match="suite_key"):
            import_runs(root, manifest)

    def test_secret_redaction(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = json.loads(json.dumps(CSV_MANIFEST))
        manifest["config"]["params"] = {"tp": 8, "api_key": "sk-secret-123"}
        manifest["extra_token"] = "abc"
        import_runs(root, manifest)
        with get_db() as conn:
            cfg = conn.execute("SELECT * FROM config_set").fetchone()
            assert json.loads(cfg["params"])["api_key"] == "***"
            batch = conn.execute("SELECT * FROM import_batch").fetchone()
            stored = json.loads(batch["manifest"])
            assert stored["extra_token"] == "***"
            assert "sk-secret-123" not in batch["manifest"]

    def test_auto_create_suite_cases(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = dict(CSV_MANIFEST, suite_key="brand-new-suite")
        report = import_runs(root, manifest)
        assert report["status"] == "ok"
        assert len(report["entities"]["cases_created"]) == 12
        with get_db() as conn:
            suite = conn.execute("SELECT * FROM suite WHERE key = 'brand-new-suite'").fetchone()
            assert suite is not None
            n = conn.execute("SELECT COUNT(*) n FROM test_case WHERE suite_id = ?",
                             (suite["id"],)).fetchone()["n"]
            assert n == 12


class TestProbeImportAndScoring:
    def _import_probes(self, tmp_path):
        scenario_id = None
        with get_db() as conn:
            from app.entities import ensure_suite, resolve_model
            model = resolve_model(conn, "cltech/glm53", ["GLM-5.3-MTP-Mix"])
            suite = ensure_suite(conn, "1m-prefill")
            scenario_id = make_scenario(conn, model, suite, name="1m交付配置")
        reports = []
        for threshold in (1024, 2048):
            root = stage(tmp_path, f"probe_threshold{threshold}.json")
            manifest = {
                "model": {"name": "cltech/glm53"},
                "config": {"name": f"lpt{threshold}",
                           "params": {"num_gpu_blocks_override": 1024,
                                      "long_prefill_token_threshold": threshold}},
                "suite_key": "1m-prefill",
                "case_key": "1m_in_1out",
                "backend": "probe-json",
                "scenario_id": scenario_id,
            }
            reports.append(import_runs(root, manifest))
        return scenario_id, reports

    def test_two_configs_same_node(self, tmp_path):
        scenario_id, reports = self._import_probes(tmp_path)
        assert all(r["status"] == "ok" for r in reports)
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 2
            comp = load_comparison(conn, scenario_id)
        suite_out = comp["suites"][0]
        assert len(suite_out["config_ids"]) == 2
        assert suite_out["insufficient"] == []
        ranking = suite_out["ranking"]
        assert ranking[0]["name"] == "lpt2048"
        assert ranking[0]["total_score"] == pytest.approx(2.7)
        assert ranking[1]["total_score"] == pytest.approx(0.3)
        assert suite_out["recommended_config_id"] == ranking[0]["config_id"]
        pw = suite_out["cases"][0]["pairwise"][0]
        assert pw["metrics"]["ttft_mean_s"]["win_rate_same"] in (0.0, 1.0)
        assert pw["metrics"]["out_tps"]["metric_score"] == 1.5

    def test_timestamps_preserved(self, tmp_path):
        self._import_probes(tmp_path)
        with get_db() as conn:
            runs = conn.execute("SELECT * FROM run").fetchall()
            for r in runs:
                assert r["started_at"].endswith("+08:00")
                assert r["ended_at"].endswith("+08:00")


class TestZipSafety:
    def test_zip_slip_rejected(self, tmp_path):
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../evil.txt", "pwned")
        with pytest.raises(ImportError_, match="穿越"):
            extract_zip(zip_path, tmp_path / "out")
        assert not (tmp_path / "evil.txt").exists()

    def test_absolute_path_rejected(self, tmp_path):
        zip_path = tmp_path / "abs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/tmp/evil.txt", "pwned")
        with pytest.raises(ImportError_, match="穿越"):
            extract_zip(zip_path, tmp_path / "out")

    def test_normal_zip(self, tmp_path):
        zip_path = tmp_path / "ok.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sub/data.csv", "node,case\nn1,c1\n")
        extract_zip(zip_path, tmp_path / "out")
        assert (tmp_path / "out" / "sub" / "data.csv").is_file()


class TestScenarioArtifacts:
    def test_extra_files_attach_to_scenario(self, tmp_path):
        with get_db() as conn:
            from app.entities import ensure_suite, resolve_model
            model = resolve_model(conn, "cltech/glm53")
            suite = ensure_suite(conn, "glm53-12case")
            sid = make_scenario(conn, model, suite)
        root = stage(tmp_path, "node_details.csv")
        shutil.copyfile(FIXTURES / "probe_threshold1024.json", root / "report.json")
        manifest = dict(CSV_MANIFEST, scenario_id=sid)
        report = import_runs(root, manifest)
        assert report["status"] == "ok"
        with get_db() as conn:
            scen_arts = conn.execute("SELECT * FROM artifact WHERE scenario_id = ?",
                                     (sid,)).fetchall()
            assert len(scen_arts) == 1
            assert scen_arts[0]["rel_path"].endswith("report.json")


class TestCaseConsoleLogImport:
    def _root(self, tmp_path, case_decl):
        root = tmp_path / "root"
        (root / "console-logs" / "c1").mkdir(parents=True)
        (root / "console-logs" / "c1" / "console.log").write_text("case log", encoding="utf-8")
        (root / "console-logs" / "c1" / "stdout.log").write_text("stdout", encoding="utf-8")
        (root / "console-logs" / "c1" / "stderr.log").write_text("stderr", encoding="utf-8")
        data = {
            "model_name": "cltech/glm53", "config_name": "cfg",
            "config_params": {"tp": 8}, "node_label": "n1",
            "suite_key": "console-suite",
            "cases": [{"case_key": "c1", "metrics": {"ttft_mean_s": 1.0},
                       "console_log": "console-logs/c1/console.log",
                       "stdout_log": "console-logs/c1/stdout.log",
                       "stderr_log": "console-logs/c1/stderr.log", **case_decl}],
        }
        (root / "run.json").write_text(json.dumps(data), encoding="utf-8")
        return root

    def _manifest(self, sid, cases=None):
        manifest = {
            "model": {"name": "cltech/glm53"},
            "config": {"name": "cfg", "params": {"tp": 8}},
            "suite_key": "console-suite", "node": "n1", "scenario_id": sid,
        }
        if cases:
            manifest["cases"] = cases
        return manifest

    def _scenario(self, conn):
        from app.entities import ensure_suite, resolve_model
        model = resolve_model(conn, "cltech/glm53")
        suite = ensure_suite(conn, "console-suite")
        return make_scenario(conn, model, suite)

    def test_normalized_case_console_log_artifact(self, tmp_path):
        with get_db() as conn:
            sid = self._scenario(conn)
        root = self._root(tmp_path, {})
        report = import_runs(root, self._manifest(sid))
        assert report["status"] == "ok"
        with get_db() as conn:
            art = conn.execute(
                "SELECT * FROM artifact WHERE title = 'c1/console.log'").fetchone()
            assert art is not None and art["run_id"] is not None
            assert (config.ARTIFACTS_DIR / art["rel_path"]).read_text(
                encoding="utf-8") == "case log"
            for title, content in (("c1/stdout.log", "stdout"),
                                   ("c1/stderr.log", "stderr")):
                row = conn.execute(
                    "SELECT * FROM artifact WHERE title = ?", (title,)).fetchone()
                assert row is not None and row["run_id"] is not None
                assert (config.ARTIFACTS_DIR / row["rel_path"]).read_text(
                    encoding="utf-8") == content

    def test_manifest_case_console_log_artifact(self, tmp_path):
        with get_db() as conn:
            sid = self._scenario(conn)
        root = self._root(tmp_path, {"console_log": None})
        manifest = self._manifest(sid, {"c1": {"console_log": "console-logs/c1/console.log"}})
        report = import_runs(root, manifest)
        assert report["status"] == "ok"
        with get_db() as conn:
            assert conn.execute(
                "SELECT COUNT(*) n FROM artifact WHERE title = 'c1/console.log'"
            ).fetchone()["n"] == 1