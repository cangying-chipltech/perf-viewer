import json
import shutil

import pytest

from app.adapters import REGISTRY, AdapterError, detect_adapter
from app.adapters.guidellm_suite import GuidellmSuiteAdapter
from app.adapters.node_csv import NodeCsvAdapter
from app.adapters.probe_json import ProbeJsonAdapter
from app.adapters.generic_json import GenericJsonAdapter
from conftest import FIXTURES


def stage(tmp_path, *sources):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    for src in sources:
        src = FIXTURES / src
        if src.is_dir():
            shutil.copytree(src, root / src.name)
        else:
            shutil.copyfile(src, root / src.name)
    return root


MANIFEST = {
    "model": {"name": "cltech/glm53", "aliases": ["GLM-5.3-MTP-Mix"]},
    "config": {"name": "cfg", "params": {"tp": 8}},
    "suite_key": "glm53-12case",
}


class TestNodeCsv:
    def test_parse_real_csv(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        runs = NodeCsvAdapter().parse(root, MANIFEST)
        assert len(runs) == 4
        assert sorted(r.node_label for r in runs) == [
            "192.168.80.10", "192.168.80.6", "192.168.80.8", "192.168.80.9"]
        for run in runs:
            assert len(run.cases) == 12
            assert run.valid is True
            assert run.suite_key == "glm53-12case"
        run6 = next(r for r in runs if r.node_label == "192.168.80.6")
        b256 = next(c for c in run6.cases if c.case_key == "B_256")
        assert b256.n_measured == 8
        assert b256.err_count == 0
        assert b256.metrics["e2e_mean_s"] == 42.18
        assert b256.metrics["ttft_mean_s"] == 31.997
        assert b256.metrics["tpot_mean_ms"] == 164.8
        assert b256.metrics["itl_mean_ms"] == 39.9
        assert b256.metrics["itl_p99_ms"] == 43.9
        assert b256.metrics["out_tps"] == 6.1
        assert b256.metrics["cache_hit_rate"] is None
        assert b256.in_tok == 200013
        assert b256.out_tok == 2048

    def test_itl_fallback_to_reqmean(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "x.csv").write_text(
            "node,case,n_measured,err,e2e_mean_s,e2e_p99_s,ttft_mean_s,ttft_p99_s,"
            "tpot_mean_ms,tpot_p99_ms,itl_reqmean_ms,itl_p99_ms,out_tps,in_tok,out_tok\n"
            "n1,c1,2,0,1,1,0.5,0.5,10,11,40.5,42,5,100,10\n", encoding="utf-8")
        runs = NodeCsvAdapter().parse(root, MANIFEST)
        assert runs[0].cases[0].metrics["itl_mean_ms"] == 40.5

    def test_err_marks_invalid(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "x.csv").write_text(
            "node,case,n_measured,err,e2e_mean_s,out_tps\nn1,c1,2,1,1,5\n", encoding="utf-8")
        runs = NodeCsvAdapter().parse(root, MANIFEST)
        assert runs[0].valid is False

    def test_detect(self, tmp_path):
        root = stage(tmp_path, "node_details.csv")
        files = [p for p in root.rglob("*") if p.is_file()]
        assert NodeCsvAdapter().detect(files, {}) is True
        assert detect_adapter(files, {}).name == "node-csv"


class TestProbeJson:
    def test_parse_real_probes(self, tmp_path):
        root = stage(tmp_path, "probe_threshold1024.json", "probe_threshold2048.json")
        manifest = dict(MANIFEST, suite_key="1m-prefill", case_key="1m_in_1out")
        runs = ProbeJsonAdapter().parse(root, manifest)
        assert len(runs) == 2
        by_ttft = {round(r.cases[0].metrics["ttft_mean_s"], 1): r for r in runs}
        r1024 = by_ttft[2411.6]
        r2048 = by_ttft[2236.4]
        for run in runs:
            assert run.node_label == "192.168.80.4"
            assert run.worker_label == "wenling-5"
            assert run.valid is True
            assert run.started_at.endswith("+08:00")
            case = run.cases[0]
            assert case.case_key == "1m_in_1out"
            assert case.n_measured == 1
            assert case.in_tok == 1000026
            assert case.out_tok == 1
            assert case.metrics["out_tps"] == 0.0
            assert case.metrics["e2e_mean_s"] == case.metrics["e2e_p99_s"]
            assert case.metrics["ttft_mean_s"] == case.metrics["ttft_p99_s"]
            assert case.metrics["tpot_mean_ms"] is None
        assert r1024.cases[0].metrics["e2e_mean_s"] == 2411.621
        assert r2048.cases[0].metrics["e2e_mean_s"] == 2236.47
        assert r2048.meta["response_id"] == "chatcmpl-a1cf63a0ae63a848"

    def test_failed_probe_invalid(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        data = json.loads((FIXTURES / "probe_threshold1024.json").read_text())
        data["ok"] = False
        data["complete"] = False
        data["error"] = "timeout"
        (root / "p.json").write_text(json.dumps(data), encoding="utf-8")
        runs = ProbeJsonAdapter().parse(root, dict(MANIFEST, suite_key="1m-prefill"))
        assert runs[0].valid is False
        assert runs[0].cases[0].err_count == 1

    def test_mixed_impact(self, tmp_path):
        root = stage(tmp_path, "mixed_impact.json")
        manifest = dict(MANIFEST, suite_key="mixed", node="n1")
        runs = ProbeJsonAdapter().parse(root, manifest)
        run = runs[0]
        keys = {c.case_key for c in run.cases}
        assert keys == {"mixed_baseline", "mixed_long", "mixed_probe_5",
                        "mixed_probe_60", "mixed_probe_120"}
        assert run.valid is False
        long = next(c for c in run.cases if c.case_key == "mixed_long")
        assert long.metrics["e2e_mean_s"] == 1842.1
        assert long.in_tok == 1008579

    def test_detect(self, tmp_path):
        root = stage(tmp_path, "probe_threshold1024.json")
        files = [p for p in root.rglob("*") if p.is_file()]
        assert detect_adapter(files, {}).name == "probe-json"


class TestGuidellmSuite:
    def test_parse_suite(self, tmp_path):
        root = stage(tmp_path, "guidellm_suite")
        manifest = dict(MANIFEST, node="192.168.80.6")
        run = GuidellmSuiteAdapter().parse(root, manifest)
        assert run.node_label == "192.168.80.6"
        assert run.worker_label == "node-6"
        assert run.valid is True
        assert len(run.cases) == 1
        case = run.cases[0]
        assert case.case_key == "B_256"
        assert case.n_measured == 8
        assert case.err_count == 0
        assert case.metrics["e2e_mean_s"] == 42.18
        assert case.metrics["e2e_p99_s"] == 43.14
        assert case.metrics["ttft_mean_s"] == pytest.approx(31.997)
        assert case.metrics["ttft_p99_s"] == pytest.approx(32.071)
        assert case.metrics["tpot_mean_ms"] == 164.8
        assert case.metrics["itl_mean_ms"] == 39.9
        assert case.metrics["out_tps"] == pytest.approx(2048 / 337.44)
        assert case.metrics["cache_hit_rate"] == 0.8098
        assert case.in_tok == 200013
        assert case.out_tok == 2048
        assert case.params["warmup"] == 1
        assert run.started_at == "2026-11-17T15:40:00+08:00"
        assert run.ended_at is not None
        kinds = {a.kind for a in run.artifacts}
        assert "json" in kinds and "csv" in kinds

    def test_failed_case_invalid(self, tmp_path):
        root = stage(tmp_path, "guidellm_suite_fail")
        manifest = dict(MANIFEST, node="n1")
        run = GuidellmSuiteAdapter().parse(root, manifest)
        assert run.valid is False
        assert run.cases[0].err_count == 1
        assert run.cases[0].metrics["cache_hit_rate"] is None

    def test_node_required(self, tmp_path):
        root = stage(tmp_path, "guidellm_suite_fail")
        with pytest.raises(AdapterError):
            GuidellmSuiteAdapter().parse(root, MANIFEST)

    def test_detect(self, tmp_path):
        root = stage(tmp_path, "guidellm_suite")
        files = [p for p in root.rglob("*") if p.is_file()]
        assert detect_adapter(files, {}).name == "guidellm-suite"


class TestGenericJson:
    def test_roundtrip(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        data = {
            "model_name": "m", "config_name": "c", "config_params": {"a": 1},
            "node_label": "n1", "suite_key": "s",
            "cases": [{"case_key": "k1", "metrics": {"ttft_mean_s": 1.5, "bogus": 9},
                       "n_measured": 2, "console_log": "k1/console.log",
                       "stdout_log": "k1/stdout.log", "stderr_log": "k1/stderr.log"}],
        }
        (root / "run.json").write_text(json.dumps(data), encoding="utf-8")
        runs = GenericJsonAdapter().parse(root, {})
        assert len(runs) == 1
        assert runs[0].cases[0].metrics["ttft_mean_s"] == 1.5
        assert "bogus" not in runs[0].cases[0].metrics
        assert runs[0].cases[0].console_log == "k1/console.log"
        assert runs[0].cases[0].stdout_log == "k1/stdout.log"
        assert runs[0].cases[0].stderr_log == "k1/stderr.log"

    def test_missing_config_rejected(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        data = {"model_name": "m", "node_label": "n", "suite_key": "s", "cases": []}
        (root / "run.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(AdapterError):
            GenericJsonAdapter().parse(root, {})


class TestRegistry:
    def test_registry_names(self):
        assert set(REGISTRY) == {"guidellm-suite", "probe-json", "node-csv", "generic-json"}

    def test_stub_rejected(self, tmp_path):
        with pytest.raises(AdapterError, match="预留"):
            detect_adapter([], {"backend": "ssh-pull"})
        with pytest.raises(AdapterError, match="预留"):
            detect_adapter([], {"backend": "profile-workflow"})

    def test_unknown_backend(self):
        with pytest.raises(AdapterError, match="未知"):
            detect_adapter([], {"backend": "nope"})