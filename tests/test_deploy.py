import json
import shutil

import pytest

from app.db import get_db, init_db
from app.entities import normalize_deploy, register_config, resolve_model
from app.importer import ImportError_, import_runs
from app.security import redact_cmd
from conftest import FIXTURES, make_scenario

DEPLOY = {
    "docker_cmd": "docker run -itd --name c --net=host image:tag",
    "env": {"VLLM_DLC_INDEX_TOPK": "256", "API_KEY": "sk-secret"},
    "serve_cmd": "vllm serve /data/models/m --served-model-name m --api-key sk-secret -tp 8",
    "extra_cmds": [
        {"title": "head · ray start --head", "cmd": "docker exec c ray start --head --port=6379"},
        {"cmd": "docker exec c ray start --address=10.0.0.1:6379 --token sk-secret"},
    ],
    "source": "exec_test",
}


def stage(tmp_path, *names):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    for name in names:
        shutil.copyfile(FIXTURES / name, root / name)
    return root


CSV_MANIFEST = {
    "model": {"name": "cltech/glm53"},
    "config": {"name": "tp8", "params": "-tp 8 -dcp 8 --block-size 256"},
    "suite_key": "glm53-12case",
    "backend": "node-csv",
}


class TestNormalizeDeploy:
    def test_env_text_lines_and_spaces(self):
        out = normalize_deploy({"env": "A=1 B=2\n# comment\nC=3 D=4"})
        assert out["env"] == {"A": "1", "B": "2", "C": "3", "D": "4"}

    def test_env_list_and_dict(self):
        assert normalize_deploy({"env": ["A=1", "B=2"]})["env"] == {"A": "1", "B": "2"}
        assert normalize_deploy({"env": {"A": 1}})["env"] == {"A": "1"}

    def test_env_invalid(self):
        with pytest.raises(ValueError):
            normalize_deploy({"env": "NOEQUALS"})
        with pytest.raises(ValueError):
            normalize_deploy({"env": 42})
        with pytest.raises(ValueError):
            normalize_deploy({"docker_cmd": 42})
        with pytest.raises(ValueError):
            normalize_deploy(["not", "dict"])

    def test_redaction(self):
        out = normalize_deploy(DEPLOY)
        assert out["env"]["API_KEY"] == "***"
        assert out["env"]["VLLM_DLC_INDEX_TOPK"] == "256"
        assert "--api-key ***" in out["serve_cmd"]
        assert "sk-secret" not in json.dumps(out)
        assert len(out["extra_cmds"]) == 2
        assert out["extra_cmds"][0]["title"] == "head · ray start --head"

    def test_extra_cmds_validation(self):
        with pytest.raises(ValueError):
            normalize_deploy({"extra_cmds": "not-list"})
        with pytest.raises(ValueError):
            normalize_deploy({"extra_cmds": [{"title": "no cmd"}]})
        with pytest.raises(ValueError):
            normalize_deploy({"extra_cmds": [{"cmd": 42}]})
        out = normalize_deploy({"extra_cmds": [{"cmd": " ray start --head "}]})
        assert out["extra_cmds"] == [{"cmd": "ray start --head"}]

    def test_redact_cmd_env_assign(self):
        assert redact_cmd("export TOKEN=abc OK=1") == "export TOKEN=*** OK=1"
        assert redact_cmd("cmd --api-key=abc") == "cmd --api-key=***"
        assert redact_cmd("cmd --api-key ***") == "cmd --api-key ***"
        assert redact_cmd("VLLM_TORCH_PROFILER_TOKENS=50") == "VLLM_TORCH_PROFILER_TOKENS=50"

    def test_none_and_empty(self):
        assert normalize_deploy(None) == {}
        assert normalize_deploy({}) == {}
        assert normalize_deploy({"docker_cmd": "  "}) == {}


class TestDeployImport:
    def test_import_stores_deploy(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = json.loads(json.dumps(CSV_MANIFEST))
        manifest["config"]["deploy"] = DEPLOY
        report = import_runs(root, manifest)
        assert report["status"] == "ok"
        with get_db() as conn:
            cfg = conn.execute("SELECT * FROM config_set").fetchone()
            deploy = json.loads(cfg["deploy"])
            assert deploy["docker_cmd"] == DEPLOY["docker_cmd"]
            assert deploy["env"]["API_KEY"] == "***"
            assert len(deploy["extra_cmds"]) == 2
            assert "sk-secret" not in cfg["deploy"]

    def test_deploy_not_in_fingerprint(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        m1 = json.loads(json.dumps(CSV_MANIFEST))
        m1["config"]["deploy"] = DEPLOY
        import_runs(root, m1)
        m2 = json.loads(json.dumps(CSV_MANIFEST))
        m2["config"]["deploy"] = {"docker_cmd": "docker run other"}
        report = import_runs(root, m2)
        assert all(r["status"] == "duplicate" for r in report["runs"])
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 1
            cfg = conn.execute("SELECT * FROM config_set").fetchone()
            assert json.loads(cfg["deploy"])["docker_cmd"] == "docker run other"

    def test_reimport_without_deploy_preserves(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        m1 = json.loads(json.dumps(CSV_MANIFEST))
        m1["config"]["deploy"] = DEPLOY
        import_runs(root, m1)
        import_runs(root, CSV_MANIFEST)
        with get_db() as conn:
            cfg = conn.execute("SELECT * FROM config_set").fetchone()
            assert json.loads(cfg["deploy"])["docker_cmd"] == DEPLOY["docker_cmd"]

    def test_invalid_deploy_rejected(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = json.loads(json.dumps(CSV_MANIFEST))
        manifest["config"]["deploy"] = {"env": "BROKEN"}
        with pytest.raises(ImportError_, match="deploy"):
            import_runs(root, manifest)

    def test_register_config_deploy_update(self, conn):
        model = resolve_model(conn, "m/x")
        cfg, created = register_config(conn, model["id"], "c", {"tp": 8},
                                       deploy={"serve_cmd": "vllm serve a"})
        assert created
        cfg2, created2 = register_config(conn, model["id"], "c", {"tp": 8},
                                         deploy={"serve_cmd": "vllm serve b"})
        assert not created2
        assert cfg2["id"] == cfg["id"]
        assert json.loads(cfg2["deploy"])["serve_cmd"] == "vllm serve b"
        cfg3, _ = register_config(conn, model["id"], "c", {"tp": 8})
        assert json.loads(cfg3["deploy"])["serve_cmd"] == "vllm serve b"


class TestDeployApi:
    def test_patch_config_deploy(self, client, conn):
        model = resolve_model(conn, "m/x")
        cfg, _ = register_config(conn, model["id"], "c", {"tp": 8})
        conn.commit()
        resp = client.patch(f"/api/catalog/configs/{cfg['id']}",
                            json={"name": "c", "notes": "", "deploy": DEPLOY},
                            headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        detail = client.get("/api/models/m/x").json()
        deploy = detail["configs"][0]["deploy"]
        assert deploy["env"]["API_KEY"] == "***"
        assert deploy["docker_cmd"] == DEPLOY["docker_cmd"]

    def test_patch_config_deploy_invalid(self, client, conn):
        model = resolve_model(conn, "m/x")
        cfg, _ = register_config(conn, model["id"], "c", {"tp": 8})
        conn.commit()
        resp = client.patch(f"/api/catalog/configs/{cfg['id']}",
                            json={"name": "c", "deploy": {"env": "BROKEN"}},
                            headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 400

    def test_patch_requires_token(self, client, conn):
        model = resolve_model(conn, "m/x")
        cfg, _ = register_config(conn, model["id"], "c", {"tp": 8})
        conn.commit()
        resp = client.patch(f"/api/catalog/configs/{cfg['id']}",
                            json={"name": "c", "deploy": DEPLOY})
        assert resp.status_code == 401


class TestScenarioPageDeploy:
    def _setup(self, tmp_path, with_deploy=True):
        with get_db() as conn:
            model = resolve_model(conn, "cltech/glm53")
            from app.entities import ensure_suite
            suite = ensure_suite(conn, "1m-prefill")
            sid = make_scenario(conn, model, suite, name="1m交付配置")
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
                "scenario_id": sid,
            }
            if with_deploy:
                manifest["config"]["deploy"] = {
                    "docker_cmd": (f"docker run -itd --name c --net=host --shm-size 256G"
                                   f" image{threshold}"),
                    "env": {"VLLM_DLC_INDEX_TOPK": "256"},
                    "serve_cmd": (f"vllm serve m --long-prefill-token-threshold {threshold}"
                                  " --block-size 256 --disable-log-requests"),
                    "extra_cmds": [{
                        "title": f"head · ray start {threshold}",
                        "cmd": f"docker exec c ray start --head --port=6379",
                    }],
                }
            import_runs(root, manifest)
        return sid

    def test_page_shows_recommended_deploy_and_best_data(self, client, tmp_path):
        sid = self._setup(tmp_path)
        resp = client.get(f"/models/cltech/glm53/scenarios/{sid}")
        assert resp.status_code == 200
        html = resp.text
        assert "当前最佳配置" in html
        assert "自动推荐" in html
        assert "核心参数" not in html
        assert "部署信息" in html
        assert "image2048" in html
        assert "VLLM_DLC_INDEX_TOPK" in html
        assert "--long-prefill-token-threshold 2048" in html
        assert "当前最佳数据" in html
        assert "最佳值" in html
        assert "来源 run" in html
        assert "模型环境变量" in html
        assert "export VLLM_DLC_INDEX_TOPK=256" in html
        assert "额外启动命令" in html and "ray start --head" in html
        assert 'class="cmd-details"' in html
        assert 'class="kv env-table"' not in html
        assert f"/scenarios/{sid}/compare" in html
        for moved in ("为什么是最佳", "评分依据", "确认 / 改判", "最佳配置历史"):
            assert moved not in html

    def test_compare_page_basis_and_deploy_compare(self, client, tmp_path):
        sid = self._setup(tmp_path)
        resp = client.get(f"/models/cltech/glm53/scenarios/{sid}/compare")
        assert resp.status_code == 200
        html = resp.text
        assert "参数对比" in html
        assert "部署信息对比" in html
        assert "只展示差异项" in html
        assert "image1024" in html and "image2048" in html
        assert "lpt1024" in html
        assert "VLLM_DLC_INDEX_TOPK=256" in html
        assert "额外启动命令" in html
        assert 'class="cmd-details"' in html
        assert "确认 / 改判" in html
        for removed in ("为什么是最佳", "为什么是候选", "评分依据", "场景报告与产物"):
            assert removed not in html

    def test_page_confirmed_best(self, client, tmp_path):
        sid = self._setup(tmp_path)
        with get_db() as conn:
            cfg = conn.execute("SELECT * FROM config_set WHERE name = 'lpt2048'").fetchone()
        resp = client.post(f"/api/scenarios/{sid}/best",
                           json={"config_id": cfg["id"], "status": "confirmed"},
                           headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}").text
        assert "人工确认" in html
        assert "image2048" in html
        detail = client.get(f"/api/scenarios/{sid}").json()
        assert "image2048" in detail["best_config"]["deploy"]["docker_cmd"]

    def test_page_missing_deploy_notice(self, client, tmp_path):
        sid = self._setup(tmp_path, with_deploy=False)
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}").text
        assert "部署信息" in html
        assert "未录入" in html

    def test_single_config_kept_simple(self, client, tmp_path):
        with get_db() as conn:
            model = resolve_model(conn, "cltech/glm53")
            from app.entities import ensure_suite
            suite = ensure_suite(conn, "1m-prefill")
            sid = make_scenario(conn, model, suite, name="single")
        root = stage(tmp_path, "probe_threshold1024.json")
        import_runs(root, {
            "model": {"name": "cltech/glm53"},
            "config": {"name": "only", "params": {"long_prefill_token_threshold": 1024},
                       "deploy": {"serve_cmd": "vllm serve only"}},
            "suite_key": "1m-prefill", "case_key": "1m_in_1out",
            "backend": "probe-json", "scenario_id": sid,
        })
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}").text
        assert "仅一套配置" in html
        assert "vllm serve only" in html
        assert "当前最佳数据" in html
        compare = client.get(f"/models/cltech/glm53/scenarios/{sid}/compare").text
        assert "参数对比" in compare
        assert "无对比对手" not in compare

    def test_compare_params_diff_filter(self, client, tmp_path):
        sid = self._setup(tmp_path)
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}/compare").text
        sec = html[html.index("params-matrix"):html.index("部署信息对比")]
        assert "long_prefill_token_threshold" in sec
        assert "2048" in sec
        assert "block_size" not in sec
        assert "disable_log_requests" not in sec
        assert "num_gpu_blocks_override" not in sec
        assert "shm_size" not in sec
        assert "VLLM_DLC_INDEX_TOPK" not in sec
        assert "image" in sec and "image1024" in sec

        html_all = client.get(f"/models/cltech/glm53/scenarios/{sid}/compare?diff=0").text
        sec_all = html_all[html_all.index("params-matrix"):html_all.index("部署信息对比")]
        assert "block_size" in sec_all
        assert "disable_log_requests" in sec_all
        assert "num_gpu_blocks_override" in sec_all
        assert "long_prefill_token_threshold" in sec_all
        assert "shm_size" in sec_all and "256G" in sec_all
        assert "network" in sec_all and "container_name" in sec_all
        assert "VLLM_DLC_INDEX_TOPK" in sec_all
        for source in ("配置参数", "容器启动", "环境变量"):
            assert source in sec_all

    def test_compare_config_selection(self, client, tmp_path):
        sid = self._setup(tmp_path)
        with get_db() as conn:
            ids = {r["name"]: r["id"] for r in
                   conn.execute("SELECT id, name FROM config_set").fetchall()}
        only = ids["lpt1024"]
        html = client.get(
            f"/models/cltech/glm53/scenarios/{sid}/compare?configs={only}&diff=0").text
        sec = html[html.index("params-matrix"):html.index("部署信息对比")]
        assert "lpt1024" in sec
        assert "lpt2048" not in sec
        deploy_sec = html[html.index("部署信息对比"):html.index("最佳配置确认")]
        assert "image1024" in deploy_sec
        assert "image2048" not in deploy_sec

    def test_compare_scoped_to_scenario_configs(self, client, tmp_path):
        sid = self._setup(tmp_path)
        with get_db() as conn:
            model = resolve_model(conn, "cltech/glm53")
            from app.entities import ensure_suite
            suite = ensure_suite(conn, "glm53-12case")
            other = make_scenario(conn, model, suite, name="other")
        root = stage(tmp_path, "node_details.csv")
        import_runs(root, {
            "model": {"name": "cltech/glm53"},
            "config": {"name": "tp8-other", "params": {"tp": 8},
                       "deploy": {"docker_cmd": "docker run tp8-image"}},
            "suite_key": "glm53-12case", "backend": "node-csv", "scenario_id": other,
        })
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}/compare").text
        assert "image1024" in html and "image2048" in html
        assert "tp8-other" not in html and "docker run tp8-image" not in html

    def test_model_page_only_scenarios_and_catalog_edits(self, client, tmp_path):
        self._setup(tmp_path)
        model_html = client.get("/models/cltech/glm53").text
        assert "交付场景" in model_html
        assert "1m交付配置" in model_html
        assert '<a class="card" href="/models/cltech%2Fglm53/scenarios/' in model_html
        assert "配置集" not in model_html
        assert "合并配置" not in model_html
        catalog = client.get("/catalog").text
        assert "配置集" in catalog
        assert "docker_cmd" in catalog
        assert "env_text" in catalog
        assert "serve_cmd" in catalog
        assert "合并配置" in catalog

    def test_empty_scenario_page_explains(self, client):
        with get_db() as conn:
            model = resolve_model(conn, "cltech/glm53")
            cur = conn.execute(
                "INSERT INTO scenario(model_id, name, scoring) VALUES(?,?,?)",
                (model["id"], "空壳场景", "{}"))
            sid = cur.lastrowid
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}").text
        assert "尚未关联测试集合" in html
        assert f"/scenarios/{sid}/compare" in html


class TestSeedDeploy:
    def test_tp16_has_two_container_ray_commands(self):
        from app.seed import tp16_deploy
        deploy = tp16_deploy(2048, "wf_x", "exec_x")
        assert len(deploy["extra_cmds"]) == 2
        titles = [item["title"] for item in deploy["extra_cmds"]]
        assert any("head 192.168.80.4" in t for t in titles)
        assert any("worker 192.168.80.5" in t for t in titles)
        cmds = [item["cmd"] for item in deploy["extra_cmds"]]
        assert any("ray start --head --port=6379 --node-ip-address=192.168.80.4" in c
                   for c in cmds)
        assert any("ray start --address=192.168.80.4:6379" in c for c in cmds)
        assert "--long-prefill-token-threshold 2048" in deploy["serve_cmd"]
        assert "sk-glm53" not in json.dumps(deploy)


class TestMigration:
    def test_v2_to_v3_adds_deploy(self, tmp_path, monkeypatch):
        from pathlib import Path

        from app import config as app_config
        from app.db import _connect
        db_path = tmp_path / "mig" / "perf_viewer.sqlite3"
        db_path.parent.mkdir(parents=True)
        conn = _connect(db_path)
        schema_sql = Path(__file__).parent.parent / "app" / "schema.sql"
        sql = schema_sql.read_text(encoding="utf-8")
        for stmt in sql.split(";"):
            if stmt.strip():
                conn.execute(stmt)
        from app.db import MIGRATIONS
        for stmt in MIGRATIONS[2]:
            conn.execute(stmt)
        conn.execute("PRAGMA user_version = 2")
        conn.execute("INSERT INTO model(name) VALUES('m/x')")
        conn.execute("INSERT INTO config_set(model_id, name, params, fingerprint)"
                     " VALUES(1, 'c', '{}', 'fp')")
        conn.commit()
        conn.close()
        monkeypatch.setattr(app_config, "DATA_DIR", db_path.parent)
        monkeypatch.setattr(app_config, "DB_PATH", db_path)
        monkeypatch.setattr(app_config, "ARTIFACTS_DIR", db_path.parent / "artifacts")
        init_db()
        conn = _connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
            row = conn.execute("SELECT deploy FROM config_set").fetchone()
            assert row["deploy"] == "{}"
        finally:
            conn.close()