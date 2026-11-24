import json

from app.db import get_db
from app.importer import import_runs
from app.repro import resolve_repro
from conftest import make_catalog, make_scenario


class TestResolveRepro:
    def test_explicit_case_command_wins(self):
        run = {"backend": "generic-json",
               "manifest": {"case_repro": {"c1": "echo reproduce c1"}}}
        cfg = {"name": "cfg"}
        assert resolve_repro("m", cfg, run, {"case_key": "c1"}) == "echo reproduce c1"

    def test_imported_case_command_from_manifest_cases(self):
        # profile-service uploads the client backend command per case.
        run = {"backend": "generic-json", "manifest": {"import_manifest": {
            "cases": {"c1": {"repro_cmd": "evalscope perf --model m --parallel 1"}}}}}
        assert resolve_repro("m", {}, run, {"case_key": "c1"}) == \
            "evalscope perf --model m --parallel 1"

    def test_manifest_level_command_fallback(self):
        run = {"backend": "generic-json", "manifest": {"repro_cmd": "echo all"}}
        assert resolve_repro("m", {}, run, {"case_key": "c1"}) == "echo all"

    def test_generated_evalscope_command(self):
        run = {
            "backend": "generic-json",
            "manifest": {
                "benchmark_backend": "evalscope",
                "benchmark_client_config": {"base_url": "http://10.0.0.1:8000/v1"},
            },
        }
        cfg = {"name": "tp8", "deploy": {"serve_cmd": "vllm serve m -tp 8"}}
        case = {"case_key": "c1",
                "params": {"input_tokens": 1024, "output_tokens": 512,
                           "concurrency": 4, "requests": 8}}
        cmd = resolve_repro("cltech/glm53", cfg, run, case, suite_key="s1")
        assert "evalscope perf" in cmd
        assert "--parallel 4" in cmd
        assert "--number 8" in cmd
        assert "--max-tokens 512" in cmd
        assert "--tokenize-prompt" in cmd
        assert "s1/c1" in cmd
        assert "vllm serve" not in cmd
        assert "docker run" not in cmd

    def test_generated_vllm_bench_command(self):
        run = {"backend": "vllm-bench", "manifest": {}}
        case = {"case_key": "c64",
                "params": {"input_tokens": 64000, "output_tokens": 1024,
                           "concurrency": 64, "requests": 100}}
        cmd = resolve_repro("cltech/glm53", {}, run, case)
        assert "benchmark_serving.py" in cmd
        assert "--random-input-len 64000" in cmd
        assert "--max-concurrency 64" in cmd


def _write_run(root, repro_cmd=None):
    root.mkdir(parents=True, exist_ok=True)
    case = {
        "case_key": "base-short-1024",
        "metrics": {"e2e_mean_s": 1.0, "out_tps": 10.0},
        "params": {"input_tokens": 1024, "output_tokens": 1024,
                   "concurrency": 1, "requests": 10},
    }
    if repro_cmd:
        case["repro_cmd"] = repro_cmd
    normalized = {
        "model": {"name": "cltech/glm53"},
        "config": {"name": "tp8", "params": {"tensor_parallel_size": 8},
                   "deploy": {"serve_cmd": "vllm serve m -tp 8"}},
        "suite_key": "evalscope",
        "node_label": "10.0.0.1",
        "backend": "generic-json",
        "cases": [case],
    }
    manifest = {
        "model": {"name": "cltech/glm53"},
        "config": {"name": "tp8", "params": {"tensor_parallel_size": 8},
                   "deploy": {"serve_cmd": "vllm serve m -tp 8"}},
        "suite_key": "evalscope",
        "backend": "generic-json",
        "node": "10.0.0.1",
    }
    (root / "normalized.json").write_text(json.dumps(normalized), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class TestReproInPages:
    def test_run_page_shows_explicit_repro(self, client, tmp_path):
        manifest = _write_run(tmp_path / "run", repro_cmd="bash reproduce.sh c1")
        report = import_runs(tmp_path / "run", manifest)
        run_id = report["runs"][0]["run_id"]
        html = client.get(f"/runs/{run_id}").text
        assert "复现命令" in html
        assert "bash reproduce.sh c1" in html

    def test_suite_compare_has_repro_row(self, client, tmp_path):
        import_runs(tmp_path / "run", _write_run(tmp_path / "run"))
        with get_db() as conn:
            model, suite, _ = make_catalog(conn, model_name="cltech/glm53", suite_key="evalscope", case_keys=())
            sid = make_scenario(conn, model, suite, name="场景")
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}/suites/{suite['id']}").text
        assert "复现命令" in html
        assert "未记录压测客户端命令" in html

    def test_scenario_best_data_cells_have_repro(self, client, tmp_path):
        import_runs(tmp_path / "run", _write_run(tmp_path / "run"))
        with get_db() as conn:
            model, suite, _ = make_catalog(conn, model_name="cltech/glm53", suite_key="evalscope", case_keys=())
            sid = make_scenario(conn, model, suite, name="场景")
        html = client.get(f"/models/cltech/glm53/scenarios/{sid}").text
        assert "repro-best-" in html
        assert "复现" in html
