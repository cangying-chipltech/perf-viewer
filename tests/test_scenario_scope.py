import json

from app.db import get_db
from app.importer import import_runs
from app.scoring import load_comparison
from conftest import make_catalog, make_run, make_scenario


def _active_configs(comp):
    return {comp["configs"][cid]["name"]
            for suite in comp["suites"] for cid in suite["config_ids"]}


class TestScenarioScopedPK:
    def test_runs_are_scoped_to_their_scenario(self):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            s1 = make_scenario(conn, model, suite, name="场景1")
            s2 = make_scenario(conn, model, suite, name="场景2")
            metric = [{"e2e_mean_s": 1.0}]
            make_run(conn, model, suite, cases, "cfg1", {"tp": 8}, "n1", metric,
                     scenario_id=s1)
            make_run(conn, model, suite, cases, "cfg2", {"tp": 16}, "n1", metric,
                     scenario_id=s2)
            unassigned, _ = make_run(conn, model, suite, cases, "unassigned",
                                     {"tp": 24}, "n1", metric)
            assert conn.execute("SELECT scenario_id FROM run WHERE id = ?",
                                (unassigned,)).fetchone()["scenario_id"] is None
        with get_db() as conn:
            assert _active_configs(load_comparison(conn, s1)) == {"cfg1"}
            assert _active_configs(load_comparison(conn, s2)) == {"cfg2"}

    def test_same_data_can_be_imported_per_scenario(self, tmp_path):
        with get_db() as conn:
            model, suite, _ = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=())
            s1 = make_scenario(conn, model, suite, name="场景1")
            s2 = make_scenario(conn, model, suite, name="场景2")
        root = tmp_path / "run"
        root.mkdir()
        normalized = {
            "model": {"name": "cltech/glm53"},
            "config": {"name": "cfg1", "params": {"tensor_parallel_size": 8}},
            "suite_key": "s1",
            "node_label": "n1",
            "backend": "generic-json",
            "cases": [{"case_key": "c1", "metrics": {"e2e_mean_s": 1.0}}],
        }
        (root / "normalized.json").write_text(json.dumps(normalized), encoding="utf-8")

        def manifest(scenario_id):
            return {
                "model": {"name": "cltech/glm53"},
                "config": {"name": "cfg1", "params": {"tensor_parallel_size": 8}},
                "suite_key": "s1",
                "backend": "generic-json",
                "node": "n1",
                "scenario_id": scenario_id,
            }

        first = import_runs(root, manifest(s1))
        second = import_runs(root, manifest(s2))
        assert first["runs"][0]["status"] == "imported"
        assert second["runs"][0]["status"] == "imported"
        with get_db() as conn:
            rows = conn.execute("SELECT scenario_id FROM run ORDER BY id").fetchall()
        assert [r["scenario_id"] for r in rows] == [s1, s2]

    def test_assign_runs_endpoint(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            s1 = make_scenario(conn, model, suite, name="场景1")
            s2 = make_scenario(conn, model, suite, name="场景2")
            run_id, _ = make_run(conn, model, suite, cases, "cfg1", {"tp": 8}, "n1",
                                 [{"e2e_mean_s": 1.0}])
        headers = {"Authorization": "Bearer test-token"}
        resp = client.post(f"/api/scenarios/{s1}/runs", json={"run_ids": [run_id]},
                           headers=headers)
        assert resp.status_code == 200
        with get_db() as conn:
            row = conn.execute("SELECT scenario_id FROM run WHERE id = ?",
                               (run_id,)).fetchone()
            assert row["scenario_id"] == s1
        resp = client.post(f"/api/runs/{run_id}/scenario", json={"scenario_id": s2},
                           headers=headers)
        assert resp.status_code == 200
        with get_db() as conn:
            row = conn.execute("SELECT scenario_id FROM run WHERE id = ?",
                               (run_id,)).fetchone()
            assert row["scenario_id"] == s2
        resp = client.post(f"/api/runs/{run_id}/scenario", json={"scenario_id": None},
                           headers=headers)
        assert resp.status_code == 200
        with get_db() as conn:
            row = conn.execute("SELECT scenario_id FROM run WHERE id = ?",
                               (run_id,)).fetchone()
            assert row["scenario_id"] is None

    def test_legacy_run_backfill_requires_unique_scenario(self):
        from app.db import assign_legacy_scenarios

        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            run_id, _ = make_run(conn, model, suite, cases, "cfg1", {"tp": 8}, "n1",
                                 [{"e2e_mean_s": 1.0}])
            sid = make_scenario(conn, model, suite, name="唯一场景")
            conn.execute("UPDATE run SET scenario_id = NULL WHERE id = ?", (run_id,))
            assign_legacy_scenarios(conn)
            row = conn.execute("SELECT scenario_id FROM run WHERE id = ?",
                               (run_id,)).fetchone()
            assert row["scenario_id"] == sid

    def test_legacy_run_stays_unassigned_when_suite_is_shared(self):
        from app.db import assign_legacy_scenarios

        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            run_id, _ = make_run(conn, model, suite, cases, "cfg1", {"tp": 8}, "n1",
                                 [{"e2e_mean_s": 1.0}])
            make_scenario(conn, model, suite, name="场景1")
            make_scenario(conn, model, suite, name="场景2")
            conn.execute("UPDATE run SET scenario_id = NULL WHERE id = ?", (run_id,))
            assign_legacy_scenarios(conn)
            row = conn.execute("SELECT scenario_id FROM run WHERE id = ?",
                               (run_id,)).fetchone()
            assert row["scenario_id"] is None

    def test_deleting_scenario_keeps_runs_shared(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            sid = make_scenario(conn, model, suite, name="场景1")
            make_run(conn, model, suite, cases, "cfg1", {"tp": 8}, "n1",
                     [{"e2e_mean_s": 1.0}], scenario_id=sid)
        resp = client.delete(f"/api/catalog/scenarios/{sid}",
                             headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        with get_db() as conn:
            row = conn.execute("SELECT scenario_id FROM run").fetchone()
        assert row["scenario_id"] is None


class TestCustomComparisonScenario:
    def test_combines_runs_from_source_scenarios(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            s1 = make_scenario(conn, model, suite, name="场景1")
            s2 = make_scenario(conn, model, suite, name="场景2")
            make_run(conn, model, suite, cases, "cfgA", {"tp": 8}, "n1",
                     [{"e2e_mean_s": 1.0}], scenario_id=s1)
            make_run(conn, model, suite, cases, "cfgB", {"tp": 16}, "n1",
                     [{"e2e_mean_s": 2.0}], scenario_id=s2)

        headers = {"Authorization": "Bearer test-token"}
        resp = client.post("/api/catalog/scenarios", json={
            "model_name": "cltech/glm53", "name": "跨场景对比",
            "description": "合并两个场景", "source_scenario_ids": [s1, s2],
        }, headers=headers)
        assert resp.status_code == 200
        custom = resp.json()["id"]

        detail = client.get(f"/api/scenarios/{custom}").json()
        assert detail["source_scenario_ids"] == [s1, s2]
        assert {s["id"] for s in detail["source_scenarios"]} == {s1, s2}
        assert detail["suites"][0]["key"] == "s1"

        comp = client.get(f"/api/scenarios/{custom}/comparison").json()
        configs = [comp["configs"][str(cid)]["name"]
                   for cid in comp["suites"][0]["config_ids"]]
        assert set(configs) == {"cfgA", "cfgB"}
        assert comp["suites"][0]["cases"][0]["case"]["case_key"] == "c1"

        with get_db() as conn:
            owners = {r["name"]: r["scenario_id"] for r in conn.execute(
                "SELECT c.name, r.scenario_id FROM run r"
                " JOIN config_set c ON c.id = r.config_id")}
        assert owners == {"cfgA": s1, "cfgB": s2}

        resp = client.patch(f"/api/catalog/scenarios/{custom}",
                            json={"name": "跨场景对比-改名"}, headers=headers)
        assert resp.status_code == 200
        detail = client.get(f"/api/scenarios/{custom}").json()
        assert detail["source_scenario_ids"] == [s1, s2]

    def test_custom_scenario_compares_only_selected_configs(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, model_name="cltech/glm53", suite_key="s1", case_keys=("c1",))
            s1 = make_scenario(conn, model, suite, name="场景1")
            s2 = make_scenario(conn, model, suite, name="场景2")
            _, cfg_a = make_run(conn, model, suite, cases, "cfgA", {"tp": 8}, "n1",
                                [{"e2e_mean_s": 1.0}], scenario_id=s1)
            _, cfg_b = make_run(conn, model, suite, cases, "cfgB", {"tp": 16}, "n1",
                                [{"e2e_mean_s": 2.0}], scenario_id=s2)
            _, cfg_c = make_run(conn, model, suite, cases, "cfgC", {"tp": 32}, "n1",
                                [{"e2e_mean_s": 3.0}], scenario_id=s2)

        headers = {"Authorization": "Bearer test-token"}
        resp = client.post("/api/catalog/scenarios", json={
            "model_name": "cltech/glm53", "name": "自选配置对比",
            "source_scenario_ids": [s1, s2], "config_ids": [cfg_a["id"], cfg_c["id"]],
        }, headers=headers)
        assert resp.status_code == 200
        custom = resp.json()["id"]

        detail = client.get(f"/api/scenarios/{custom}").json()
        assert detail["config_ids"] == [cfg_a["id"], cfg_c["id"]]
        comp = client.get(f"/api/scenarios/{custom}/comparison").json()
        names = {comp["configs"][str(cid)]["name"]
                 for cid in comp["suites"][0]["config_ids"]}
        assert names == {"cfgA", "cfgC"}

        resp = client.patch(f"/api/catalog/scenarios/{custom}",
                            json={"name": "自选配置对比-改名"}, headers=headers)
        assert resp.status_code == 200
        assert client.get(f"/api/scenarios/{custom}").json()["config_ids"] == [
            cfg_a["id"], cfg_c["id"]]

        resp = client.patch(f"/api/catalog/scenarios/{custom}",
                            json={"name": "自选配置对比-字符串",
                                  "config_ids": str(cfg_b["id"]),
                                  "source_scenario_ids": f"{s1},{s2}"},
                            headers=headers)
        assert resp.status_code == 200
        detail = client.get(f"/api/scenarios/{custom}").json()
        assert detail["config_ids"] == [cfg_b["id"]]
        assert detail["source_scenario_ids"] == [s1, s2]

        resp = client.patch(f"/api/catalog/scenarios/{custom}",
                            json={"name": "自选配置对比-全部", "config_ids": []},
                            headers=headers)
        assert resp.status_code == 200
        comp = client.get(f"/api/scenarios/{custom}/comparison").json()
        names = {comp["configs"][str(cid)]["name"]
                 for cid in comp["suites"][0]["config_ids"]}
        assert names == {"cfgA", "cfgB", "cfgC"}

    def test_source_scenario_must_share_model(self, client):
        with get_db() as conn:
            model, suite, _ = make_catalog(conn, model_name="cltech/glm53",
                                           suite_key="s1", case_keys=())
            sid = make_scenario(conn, model, suite, name="场景1")
            other, other_suite, _ = make_catalog(conn, model_name="other/model",
                                                 suite_key="s2", case_keys=())
            other_sid = make_scenario(conn, other, other_suite, name="其他场景")

        headers = {"Authorization": "Bearer test-token"}
        resp = client.post("/api/catalog/scenarios", json={
            "model_name": "cltech/glm53", "name": "坏对比",
            "source_scenario_ids": [sid, other_sid],
        }, headers=headers)
        assert resp.status_code == 400
