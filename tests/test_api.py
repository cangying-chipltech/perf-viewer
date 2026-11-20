import io
import json
import zipfile

import pytest

from app import config
from app.db import get_db
from conftest import FIXTURES, make_catalog, make_run, make_scenario

CSV_MANIFEST = {
    "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
    "config": {"name": "tp8", "params": {"tp": 8}},
    "suite_key": "glm53-12case",
}


def import_csv(client, manifest=None, dry_run=False):
    csv_bytes = (FIXTURES / "node_details.csv").read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("node_details.csv", csv_bytes)
    buf.seek(0)
    files = [
        ("manifest", ("manifest.json", json.dumps(manifest or CSV_MANIFEST).encode(),
                      "application/json")),
        ("files", ("data.zip", buf.getvalue(), "application/zip")),
    ]
    return client.post("/api/import", files=files,
                       data={"dry_run": "true" if dry_run else "false"},
                       headers={"Authorization": "Bearer test-token"})


class TestHealthAndRead:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_models_empty(self, client):
        assert client.get("/api/models").json() == {"models": []}

    def test_model_404(self, client):
        assert client.get("/api/models/nope").status_code == 404

    def test_pages_render(self, client):
        assert client.get("/").status_code == 200
        assert client.get("/catalog").status_code == 200

    def test_import_then_read(self, client):
        resp = import_csv(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        models = client.get("/api/models").json()["models"]
        assert len(models) == 1
        assert models[0]["name"] == "cltech/glm53"
        assert models[0]["run_count"] == 4
        detail = client.get("/api/models/cltech/glm53").json()
        assert detail["name"] == "cltech/glm53"
        detail_encoded = client.get("/api/models/cltech%2Fglm53").json()
        assert detail_encoded["name"] == "cltech/glm53"
        runs = client.get("/api/runs/1").json()
        assert len(runs["case_results"]) == 12
        assert client.get("/").status_code == 200
        assert client.get("/models/cltech/glm53").status_code == 200
        assert client.get("/runs/1").status_code == 200

    def test_slash_model_subroutes(self, client):
        import_csv(client)
        with get_db() as conn:
            model = conn.execute("SELECT * FROM model").fetchone()
            suite = conn.execute("SELECT * FROM suite").fetchone()
            sid = make_scenario(conn, model, suite)
        assert client.get("/api/models/cltech/glm53/scenarios").status_code == 200
        assert client.get(f"/models/cltech/glm53/scenarios/{sid}").status_code == 200
        assert client.get(f"/models/cltech/glm53/scenarios/{sid}?mode=best").status_code == 200


class TestAuth:
    def test_no_token(self, client):
        resp = client.post("/api/runs/1/exclude", json={"excluded": True})
        assert resp.status_code == 401

    def test_wrong_token(self, client):
        resp = client.post("/api/runs/1/exclude", json={"excluded": True},
                           headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_token_unset(self, client, monkeypatch):
        monkeypatch.setattr(config, "TOKEN", "")
        resp = client.post("/api/runs/1/exclude", json={"excluded": True},
                           headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 503

    def test_reads_open(self, client):
        assert client.get("/api/models").status_code == 200
        assert client.get("/api/catalog").status_code == 200


class TestImportApi:
    def test_missing_manifest(self, client):
        resp = client.post("/api/import", files=[("files", ("a.csv", b"x", "text/csv"))],
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 400

    def test_missing_config_rejected(self, client):
        manifest = {k: v for k, v in CSV_MANIFEST.items() if k != "config"}
        resp = import_csv(client, manifest=manifest)
        assert resp.status_code == 400
        assert "config" in resp.json()["detail"]

    def test_dry_run(self, client):
        resp = import_csv(client, dry_run=True)
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 0

    def test_duplicate_import(self, client):
        assert import_csv(client).json()["status"] == "ok"
        resp = import_csv(client)
        assert all(r["status"] == "duplicate" for r in resp.json()["runs"])

    def test_evil_zip_rejected(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "pwned")
        resp = client.post(
            "/api/import",
            files=[("manifest", ("manifest.json", json.dumps(CSV_MANIFEST).encode(),
                                "application/json")),
                   ("files", ("evil.zip", buf.getvalue(), "application/zip"))],
            headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 400


class TestBestDecision:
    def _setup_scenario(self, client):
        import_csv(client)
        with get_db() as conn:
            model = conn.execute("SELECT * FROM model").fetchone()
            suite = conn.execute("SELECT * FROM suite").fetchone()
            sid = make_scenario(conn, model, suite)
            cfg = conn.execute("SELECT * FROM config_set").fetchone()
        return sid, cfg["id"]

    def test_confirm(self, client):
        sid, cfg_id = self._setup_scenario(client)
        resp = client.post(f"/api/scenarios/{sid}/best",
                           json={"config_id": cfg_id, "status": "confirmed"},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        detail = client.get(f"/api/scenarios/{sid}").json()
        assert detail["best_status"] == "confirmed"
        assert detail["best_config"]["id"] == cfg_id
        assert len(detail["history"]) == 1

    def test_override_requires_reason(self, client):
        sid, cfg_id = self._setup_scenario(client)
        resp = client.post(f"/api/scenarios/{sid}/best",
                           json={"config_id": cfg_id, "status": "overridden", "reason": "  "},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 400
        resp = client.post(f"/api/scenarios/{sid}/best",
                           json={"config_id": cfg_id, "status": "overridden",
                                 "reason": "人工经验判断"},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        detail = client.get(f"/api/scenarios/{sid}").json()
        assert detail["best_status"] == "overridden"
        assert detail["best_reason"] == "人工经验判断"

    def test_foreign_config_rejected(self, client):
        sid, _ = self._setup_scenario(client)
        with get_db() as conn:
            from app.entities import register_config, resolve_model
            other = resolve_model(conn, "other/model")
            cfg, _ = register_config(conn, other["id"], "x", {"a": 1})
        resp = client.post(f"/api/scenarios/{sid}/best",
                           json={"config_id": cfg["id"], "status": "confirmed"},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 400


class TestExclude:
    def test_exclude_changes_comparison(self, client):
        import_csv(client)
        with get_db() as conn:
            model = conn.execute("SELECT * FROM model").fetchone()
            suite = conn.execute("SELECT * FROM suite").fetchone()
            sid = make_scenario(conn, model, suite)
        comp = client.get(f"/api/scenarios/{sid}/comparison").json()
        assert comp["suites"][0]["ranking"][0]["n_runs"] == 4
        resp = client.post("/api/runs/1/exclude", json={"excluded": True, "reason": "异常"},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        comp = client.get(f"/api/scenarios/{sid}/comparison").json()
        assert comp["suites"][0]["ranking"][0]["n_runs"] == 3
        run = client.get("/api/runs/1").json()
        assert run["excluded"] == 1
        assert run["manifest"]["exclude_reason"] == "异常"
        client.post("/api/runs/1/exclude", json={"excluded": False},
                    headers={"Authorization": "Bearer test-token"})
        comp = client.get(f"/api/scenarios/{sid}/comparison").json()
        assert comp["suites"][0]["ranking"][0]["n_runs"] == 4


class TestCatalogApi:
    def test_suite_case_scenario_crud(self, client):
        headers = {"Authorization": "Bearer test-token"}
        resp = client.post("/api/catalog/models", json={"name": "m1", "aliases": ["a1"]},
                           headers=headers)
        assert resp.status_code == 200
        resp = client.post("/api/catalog/suites", json={
            "key": "s1", "name": "套件1",
            "cases": [{"case_key": "c1", "weight": 2.0, "params": {"x": 1}}]},
            headers=headers)
        assert resp.status_code == 200
        suite_id = resp.json()["id"]
        resp = client.post(f"/api/catalog/suites/{suite_id}/cases",
                           json={"case_key": "c2", "weight": 1.0}, headers=headers)
        case_id = resp.json()["id"]
        resp = client.patch(f"/api/catalog/cases/{case_id}",
                            json={"case_key": "c2", "weight": 3.0, "params": {},
                                  "description": "d", "sort_order": 1}, headers=headers)
        assert resp.status_code == 200
        resp = client.post("/api/catalog/scenarios",
                           json={"model_name": "m1", "name": "sc1", "suite_keys": ["s1"],
                                 "scoring": {"epsilon": 0.02}}, headers=headers)
        sid = resp.json()["id"]
        detail = client.get(f"/api/scenarios/{sid}").json()
        assert detail["scoring"]["epsilon"] == 0.02
        assert detail["suites"][0]["key"] == "s1"
        resp = client.patch(f"/api/catalog/scenarios/{sid}",
                            json={"name": "sc1-renamed", "suite_keys": []}, headers=headers)
        assert resp.status_code == 200
        assert client.get(f"/api/scenarios/{sid}").json()["suites"] == []
        assert client.delete(f"/api/catalog/scenarios/{sid}", headers=headers).status_code == 200
        assert client.delete(f"/api/catalog/cases/{case_id}", headers=headers).status_code == 200

    def test_delete_case_with_results_blocked(self, client):
        import_csv(client)
        with get_db() as conn:
            case = conn.execute("SELECT * FROM test_case LIMIT 1").fetchone()
        resp = client.delete(f"/api/catalog/cases/{case['id']}",
                             headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 409

    def test_duplicate_suite_rejected(self, client):
        headers = {"Authorization": "Bearer test-token"}
        client.post("/api/catalog/suites", json={"key": "dup"}, headers=headers)
        resp = client.post("/api/catalog/suites", json={"key": "dup"}, headers=headers)
        assert resp.status_code == 409


class TestMerge:
    def test_merge_moves_runs(self, client):
        import_csv(client)
        with get_db() as conn:
            from app.entities import register_config
            model = conn.execute("SELECT * FROM model").fetchone()
            cfg2, _ = register_config(conn, model["id"], "tp8-variant", {"tp": "8"})
            cfg1 = conn.execute("SELECT * FROM config_set WHERE name = 'tp8'").fetchone()
        assert cfg2["id"] == cfg1["id"]
        with get_db() as conn:
            cfg3, _ = register_config(conn, model["id"], "other", {"tp": 16})
        resp = client.post("/api/configs/merge",
                           json={"source_id": cfg1["id"], "target_id": cfg3["id"]},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        assert resp.json()["moved_runs"] == 4
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 1
            assert conn.execute("SELECT COUNT(*) n FROM run WHERE config_id = ?",
                                (cfg3["id"],)).fetchone()["n"] == 4


class TestArtifactSafety:
    def test_traversal_rel_path_rejected(self, client):
        import_csv(client)
        with get_db() as conn:
            conn.execute("INSERT INTO artifact(run_id, kind, rel_path, title)"
                         " VALUES(1, 'log', '../outside.txt', 'evil')")
            art_id = conn.execute("SELECT MAX(id) id FROM artifact").fetchone()["id"]
        resp = client.get(f"/api/artifacts/{art_id}")
        assert resp.status_code == 404

    def test_artifact_download(self, client):
        import_csv(client)
        art = client.get("/api/artifacts/1")
        assert art.status_code == 200
        assert "node,case" in art.text

    def test_comparison_mode_validation(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(conn)
            sid = make_scenario(conn, model, suite)
        assert client.get(f"/api/scenarios/{sid}/comparison?mode=bogus").status_code == 400
        assert client.get(f"/api/scenarios/{sid}/comparison?mode=best").status_code == 200
        assert client.get(f"/api/scenarios/{sid}/comparison?mode=adjusted").status_code == 200
        assert client.get("/api/scenarios/999/comparison").status_code == 404

class TestSkillAPI:
    def test_list_detail_and_package(self, client):
        listing = client.get("/api/skill")
        assert listing.status_code == 200
        skills = listing.json()["skills"]
        assert skills
        name = next(s["name"] for s in skills if s["name"] == "perf-viewer-import")

        detail = client.get(f"/api/skill?name={name}")
        assert detail.status_code == 200
        body = detail.json()
        assert "SKILL.md" in [f["path"] for f in body["files"]]
        assert "perf-viewer-import" in body["skill_md"]

        pkg = client.get(f"/api/skill/package?name={name}")
        assert pkg.status_code == 200
        assert pkg.headers["content-type"] == "application/zip"
        assert pkg.headers.get("X-Checksum-SHA256")
        with zipfile.ZipFile(io.BytesIO(pkg.content)) as zf:
            names = zf.namelist()
        assert f"{name}/SKILL.md" in names
        assert f"{name}/scripts/import_kit.py" in names

    def test_missing_and_invalid_name(self, client):
        assert client.get("/api/skill?name=not-exists").status_code == 404
        assert client.get("/api/skill/package?name=../etc").status_code == 400
