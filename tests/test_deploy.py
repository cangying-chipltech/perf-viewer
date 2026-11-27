import json
import re
import shutil
from pathlib import Path

import pytest

from app import config
from app.db import get_db, init_db
from app.entities import normalize_deploy, register_config, resolve_model
from app.importer import ImportError_, import_runs
from app.security import redact_cmd
from conftest import FIXTURES, make_catalog, make_run, make_scenario

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

    def test_deploy_variant_gets_its_own_config(self, tmp_path):
        # The deployment record is part of the configuration identity: a different
        # docker/serve command must not relabel the runs imported before it.
        root = stage(tmp_path, "node_details.csv")
        m1 = json.loads(json.dumps(CSV_MANIFEST))
        m1["config"]["deploy"] = DEPLOY
        import_runs(root, m1)
        m2 = json.loads(json.dumps(CSV_MANIFEST))
        m2["config"]["deploy"] = {"docker_cmd": "docker run other"}
        report = import_runs(root, m2)
        assert all(r["status"] == "imported" for r in report["runs"])
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 2
            deploys = [json.loads(r["deploy"])["docker_cmd"]
                       for r in conn.execute("SELECT deploy FROM config_set ORDER BY id")]
            assert DEPLOY["docker_cmd"] in deploys
            assert "docker run other" in deploys

    def test_reimport_without_deploy_preserves(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        m1 = json.loads(json.dumps(CSV_MANIFEST))
        m1["config"]["deploy"] = DEPLOY
        import_runs(root, m1)
        report = import_runs(root, CSV_MANIFEST)
        assert all(r["status"] == "duplicate" for r in report["runs"])
        with get_db() as conn:
            assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 1
            assert conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 4
            cfg = conn.execute("SELECT * FROM config_set").fetchone()
            assert json.loads(cfg["deploy"])["docker_cmd"] == DEPLOY["docker_cmd"]

    def test_invalid_deploy_rejected(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        manifest = json.loads(json.dumps(CSV_MANIFEST))
        manifest["config"]["deploy"] = {"env": "BROKEN"}
        with pytest.raises(ImportError_, match="deploy"):
            import_runs(root, manifest)

    def test_register_config_deploy_conflict_creates_variant(self, conn):
        # Different serve command with the same declared params must not rewrite the
        # deployment record of the runs that used the previous command.
        model = resolve_model(conn, "m/x")
        cfg, created = register_config(conn, model["id"], "c", {"tp": 8},
                                       deploy={"serve_cmd": "vllm serve a"})
        assert created
        cfg2, created2 = register_config(conn, model["id"], "c", {"tp": 8},
                                         deploy={"serve_cmd": "vllm serve b"})
        assert created2
        assert cfg2["id"] != cfg["id"]
        stored = conn.execute("SELECT * FROM config_set WHERE id = ?", (cfg["id"],)).fetchone()
        assert json.loads(stored["deploy"])["serve_cmd"] == "vllm serve a"
        assert json.loads(cfg2["deploy"])["serve_cmd"] == "vllm serve b"

    def test_register_config_same_deploy_is_idempotent(self, conn):
        model = resolve_model(conn, "m/x")
        cfg, _ = register_config(conn, model["id"], "c", {"tp": 8},
                                 deploy={"serve_cmd": "vllm serve a"})
        cfg2, created2 = register_config(conn, model["id"], "c", {"tp": 8},
                                         deploy={"serve_cmd": "vllm serve a"})
        assert not created2
        assert cfg2["id"] == cfg["id"]

    def test_register_config_reordered_flags_is_idempotent(self, conn):
        model = resolve_model(conn, "m/x")
        cfg, _ = register_config(
            conn, model["id"], "c", {"tp": 16, "block_size": 256},
            deploy={"serve_cmd": "vllm serve /m -tp 16 --block-size 256"})
        cfg2, created2 = register_config(
            conn, model["id"], "c", {"tp": 16, "block_size": 256},
            deploy={"serve_cmd": "vllm serve /m --block-size 256 -tp 16"})
        assert not created2
        assert cfg2["id"] == cfg["id"]

    def test_register_config_keeps_deploy_without_incoming(self, conn):
        model = resolve_model(conn, "m/x")
        cfg, _ = register_config(conn, model["id"], "c", {"tp": 8},
                                 deploy={"serve_cmd": "vllm serve a"})
        cfg2, _ = register_config(conn, model["id"], "c", {"tp": 8})
        assert cfg2["id"] == cfg["id"]
        assert json.loads(cfg2["deploy"])["serve_cmd"] == "vllm serve a"


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
        assert "节点执行数据" in html
        assert 'class="case-runs-link"' in html
        assert "weight=" not in html
        assert "data-case-filter" in html
        assert 'href="/runs/' not in html
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

    def test_case_runs_partial_shows_node_records(self, client, tmp_path):
        sid = self._setup(tmp_path)
        with get_db() as conn:
            case = conn.execute(
                "SELECT * FROM test_case WHERE case_key = '1m_in_1out'").fetchone()
            cfg = conn.execute(
                "SELECT * FROM config_set WHERE name = 'lpt2048'").fetchone()
            run_count = conn.execute(
                "SELECT COUNT(*) n FROM run r JOIN case_result cr ON cr.run_id = r.id"
                " WHERE r.config_id = ? AND cr.case_id = ?",
                (cfg["id"], case["id"])).fetchone()["n"]
        resp = client.get(
            f"/partials/scenarios/{sid}/case_runs"
            f"?case={case['id']}&config={cfg['id']}")
        assert resp.status_code == 200
        html = resp.text
        assert "节点执行数据" in html
        assert 'class="node-runs"' in html
        assert 'href="/runs/' in html
        if run_count > 1:
            assert f"展开 {run_count} 条记录" in html
            assert 'class="run-record-table"' in html

    def test_case_runs_partial_expands_multiple_node_records(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, suite_key="multi-runs", case_keys=("c1",))
            sid = make_scenario(conn, model, suite, name="multi")
            for value in (1.0, 1.2):
                make_run(conn, model, suite, cases, "cfg", {"x": 1}, "node-a",
                         [{"e2e_mean_s": value}], started_at="2026-01-01T00:00:00")
        resp = client.get(
            f"/partials/scenarios/{sid}/case_runs?case={cases[0]['id']}")
        assert resp.status_code == 200
        html = resp.text
        assert "node-a" in html
        assert "展开 2 条记录" in html
        assert 'class="run-record-table"' in html

    def test_run_page_expands_case_logs(self, client):
        with get_db() as conn:
            model, suite, cases = make_catalog(
                conn, suite_key="run-case-logs", case_keys=("c1", "c2"))
            sid = make_scenario(conn, model, suite, name="run-logs")
            run_id, _cfg = make_run(conn, model, suite, cases, "cfg", {"x": 1},
                                    "node-a",
                                    [{"e2e_mean_s": 1.0}, {"e2e_mean_s": 2.0}])
            for case_key, name, content in (
                    ("c1", "stdout.log", "C1 OUT"), ("c2", "stderr.log", "C2 ERR")):
                rel = Path("logs") / case_key / name
                path = config.ARTIFACTS_DIR / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                conn.execute(
                    "INSERT INTO artifact(run_id, scenario_id, kind, rel_path, title)"
                    " VALUES(?,?,?,?,?)",
                    (run_id, None, "log", str(rel), f"{case_key}/{name}"))

        html = client.get(f"/runs/{run_id}").text
        assert "日志文件" in html
        assert html.count('class="case-log-file"') == 2
        assert html.count('class="case-log-toggle"') == 2
        log_rows = re.findall(r'<tr class="case-log-row".*?</tr>', html, re.S)
        assert len(log_rows) == 2
        assert "stdout.log" in log_rows[0] and "C1 OUT" in log_rows[0]
        assert "C2 ERR" not in log_rows[0]
        assert "stderr.log" in log_rows[1] and "C2 ERR" in log_rows[1]
        assert "C1 OUT" not in log_rows[1]

        scenario = client.get(f"/models/{model['name']}/scenarios/{sid}").text
        assert "case-console-log-link" not in scenario
        assert "C1 OUT" not in scenario and "C2 ERR" not in scenario

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
            from app.db import SCHEMA_VERSION
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            row = conn.execute("SELECT deploy FROM config_set").fetchone()
            assert row["deploy"] == "{}"
        finally:
            conn.close()