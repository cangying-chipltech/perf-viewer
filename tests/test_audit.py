import json

from app.audit import deploy_audit, recompute_config_identity
from app.entities import config_identity, register_config
from tests.conftest import make_catalog, make_scenario

BASE_SERVE = (
    "vllm serve /data/models/GLM-5.3-MTP-Mix --served-model-name cltech/glm53 "
    "-tp 16 --block-size 256 --max-num-seqs 4 --max-num-batched-tokens 4096 "
    "--compilation-config '{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' "
    "--speculative-config '{\"num_speculative_tokens\":5,\"method\":\"deepseek_mtp\"}'"
)
NONE_SERVE = BASE_SERVE.replace("FULL_DECODE_ONLY", "NONE")
PARAMS = {"tensor_parallel_size": 16, "block_size": 256, "max_num_seqs": 4,
          "max_num_batched_tokens": 4096}


class TestConfigIdentity:
    def test_serve_flag_splits_identity(self):
        a, _ = config_identity(PARAMS, {"serve_cmd": BASE_SERVE})
        b, _ = config_identity(PARAMS, {"serve_cmd": NONE_SERVE})
        assert a != b

    def test_declared_params_do_not_change_identity_when_serve_disagrees(self):
        # The serve command is what actually ran, so declaring the flag in params
        # (or not) must not move the identity.
        with_flag = dict(PARAMS, compilation_config='{"mode":0,"cudagraph_mode":"NONE"}')
        a, _ = config_identity(with_flag, {"serve_cmd": NONE_SERVE})
        b, _ = config_identity(PARAMS, {"serve_cmd": NONE_SERVE})
        assert a == b

    def test_env_splits_identity(self):
        a, _ = config_identity(PARAMS, {"serve_cmd": BASE_SERVE, "env": {"X": "1"}})
        b, _ = config_identity(PARAMS, {"serve_cmd": BASE_SERVE, "env": {"X": "2"}})
        assert a != b

    def test_secret_flags_ignored(self):
        a, _ = config_identity(PARAMS, {"serve_cmd": BASE_SERVE + " --api-key one"})
        b, _ = config_identity(PARAMS, {"serve_cmd": BASE_SERVE + " --api-key two"})
        assert a == b

    def test_redacted_env_values_are_unknown(self):
        a, _ = config_identity(PARAMS, {"serve_cmd": NONE_SERVE,
                                        "env": {"API_KEY": "***", "X": "1"}})
        b, _ = config_identity(PARAMS, {"serve_cmd": NONE_SERVE, "env": {"X": "1"}})
        assert a == b

    def test_identity_matches_params_only_without_deploy(self):
        fp, stored = config_identity(PARAMS)
        assert stored == PARAMS
        from app.entities import config_fingerprint
        assert fp == config_fingerprint(PARAMS)


def _insert_run(conn, model, suite, cases, cfg_id, metrics, node="n1",
                deploy=None, scenario_id=None):
    from app.entities import register_node
    node_row = register_node(conn, node)
    manifest = {"import_manifest": {"config": {"deploy": deploy or {}}}}
    cur = conn.execute(
        "INSERT INTO run(model_id, config_id, node_id, suite_id, backend, valid, excluded,"
        " manifest, scenario_id) VALUES(?,?,?,?,?,?,?,?,?)",
        (model["id"], cfg_id, node_row["id"], suite["id"], "test", 1, 0,
         json.dumps(manifest), scenario_id),
    )
    run_id = cur.lastrowid
    for case, values in zip(cases, metrics):
        conn.execute(
            "INSERT INTO case_result(run_id, case_id, n_measured, err_count, metrics)"
            " VALUES(?,?,?,?,?)", (run_id, case["id"], 1, 0, json.dumps(values)))
    return run_id


class TestDeployAudit:
    def _setup(self, conn):
        model, suite, cases = make_catalog(conn)
        cfg, _ = register_config(conn, model["id"], "c", PARAMS,
                                 deploy={"serve_cmd": NONE_SERVE})
        return model, suite, cases, cfg

    def test_flags_run_whose_recorded_deploy_differs(self, conn):
        model, suite, cases, cfg = self._setup(conn)
        bad = _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 1}],
                          deploy={"serve_cmd": BASE_SERVE})
        good = _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 2}],
                           deploy={"serve_cmd": NONE_SERVE})
        report = deploy_audit(conn)
        assert not report["ok"]
        assert [m["run"] for m in report["mismatched_runs"]] == [bad]
        assert "compilation_config" in report["mismatched_runs"][0]["serve_flags_differ"]
        assert good not in [m["run"] for m in report["mismatched_runs"]]

    def test_redacted_env_values_are_not_mismatches(self, conn):
        model, suite, cases, cfg = self._setup(conn)
        cfg_deploy = json.loads(
            conn.execute("SELECT deploy FROM config_set WHERE id = ?",
                         (cfg["id"],)).fetchone()["deploy"])
        cfg_deploy["serve_cmd"] = NONE_SERVE
        cfg_deploy["env"] = {"VLLM_TORCH_PROFILER_TOKENS": "***"}
        conn.execute("UPDATE config_set SET deploy = ? WHERE id = ?",
                     (json.dumps(cfg_deploy), cfg["id"]))
        _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 1}],
                    deploy={"serve_cmd": NONE_SERVE,
                            "env": {"VLLM_TORCH_PROFILER_TOKENS": "50"}})
        assert deploy_audit(conn)["ok"]

    def test_runs_without_deploy_evidence_are_skipped(self, conn):
        model, suite, cases, cfg = self._setup(conn)
        _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 1}])
        assert deploy_audit(conn)["ok"]

    def test_env_text_and_list_forms_are_compared(self, conn):
        model, suite, cases, cfg = self._setup(conn)
        cfg_deploy = json.loads(
            conn.execute("SELECT deploy FROM config_set WHERE id = ?",
                         (cfg["id"],)).fetchone()["deploy"])
        cfg_deploy["env"] = {"A": "1"}
        conn.execute("UPDATE config_set SET deploy = ? WHERE id = ?",
                     (json.dumps(cfg_deploy), cfg["id"]))
        text_run = _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 1}],
                               deploy={"serve_cmd": NONE_SERVE, "env": "A=2"})
        list_run = _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 2}],
                               deploy={"serve_cmd": NONE_SERVE, "env": ["A=2"]})
        report = deploy_audit(conn)
        assert not report["ok"]
        assert {m["run"] for m in report["mismatched_runs"]} == {text_run, list_run}
        assert all("A" in m["env_keys_differ"] for m in report["mismatched_runs"])

    def test_merge_rewrites_scenario_config_ids_and_history(self, conn):
        model, suite, cases = make_catalog(conn)
        sid = make_scenario(conn, model, suite, name="场景")
        first, _ = register_config(conn, model["id"], "c", PARAMS,
                                   deploy={"serve_cmd": NONE_SERVE})
        conn.execute(
            "INSERT INTO config_set(model_id, name, params, fingerprint, deploy)"
            " VALUES(?,?,?,?,?)",
            (model["id"], "c-legacy", json.dumps(PARAMS), "legacy",
             json.dumps({"serve_cmd": NONE_SERVE})),
        )
        legacy_id = conn.execute(
            "SELECT id FROM config_set WHERE fingerprint = 'legacy'").fetchone()["id"]
        conn.execute("UPDATE scenario SET config_ids = ? WHERE id = ?",
                     (json.dumps([legacy_id]), sid))
        conn.execute(
            "INSERT INTO best_history(scenario_id, prev_config_id, new_config_id,"
            " status, reason, created_at) VALUES(?,?,?,?,?,?)",
            (sid, legacy_id, legacy_id, "confirmed", "", "2026-01-01"),
        )
        _insert_run(conn, model, suite, cases, legacy_id, [{"x": 1}],
                    deploy={"serve_cmd": NONE_SERVE}, scenario_id=sid)

        recompute_config_identity(conn)

        scenario = conn.execute("SELECT config_ids FROM scenario WHERE id = ?",
                                (sid,)).fetchone()
        assert json.loads(scenario["config_ids"]) == [first["id"]]
        history = conn.execute("SELECT prev_config_id, new_config_id FROM best_history"
                               " WHERE scenario_id = ?", (sid,)).fetchone()
        assert history["prev_config_id"] == first["id"]
        assert history["new_config_id"] == first["id"]

    def test_merges_duplicate_configs_from_old_fingerprint(self, conn):
        model, suite, cases = make_catalog(conn)
        # Two rows with the legacy params-only fingerprint, same params and deploy.
        first, _ = register_config(conn, model["id"], "c", PARAMS,
                                   deploy={"serve_cmd": NONE_SERVE})
        conn.execute(
            "INSERT INTO config_set(model_id, name, params, fingerprint, deploy)"
            " VALUES(?,?,?,?,?)",
            (model["id"], "c-legacy", json.dumps(PARAMS), "legacy",
             json.dumps({"serve_cmd": NONE_SERVE})),
        )
        legacy_id = conn.execute(
            "SELECT id FROM config_set WHERE fingerprint = 'legacy'").fetchone()["id"]
        run_id = _insert_run(conn, model, suite, cases, legacy_id, [{"x": 1}],
                             deploy={"serve_cmd": NONE_SERVE})
        report = recompute_config_identity(conn)
        assert report["merged_configs"]
        assert conn.execute("SELECT config_id FROM run WHERE id = ?",
                            (run_id,)).fetchone()["config_id"] == first["id"]
        assert conn.execute("SELECT COUNT(*) n FROM config_set").fetchone()["n"] == 1

    def test_audit_api(self, client, conn):
        model, suite, cases, cfg = self._setup(conn)
        _insert_run(conn, model, suite, cases, cfg["id"], [{"x": 1}],
                    deploy={"serve_cmd": BASE_SERVE})
        conn.commit()
        resp = client.get("/api/deploy-audit")
        assert resp.status_code == 200
        assert resp.json()["ok"] is False

    def test_compare_page_warns_about_drift(self, client, conn):
        model, suite, cases = make_catalog(conn)
        sid = make_scenario(conn, model, suite, name="场景")
        cfg, _ = register_config(conn, model["id"], "c", PARAMS,
                                 deploy={"serve_cmd": NONE_SERVE})
        _insert_run(conn, model, suite, cases, cfg["id"],
                    [{"e2e_mean_s": 10.0}] * len(cases),
                    deploy={"serve_cmd": BASE_SERVE}, scenario_id=sid)
        conn.commit()
        resp = client.get(f"/models/{model['name']}/scenarios/{sid}/compare")
        assert resp.status_code == 200
        assert "部署记录不一致" in resp.text
