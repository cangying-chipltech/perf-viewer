import json
from pathlib import Path

from app.import_config import (
    _auto_config_name,
    _case_groups,
    _image_from_docker,
    _merge_parts,
    _parse_deploy_dir,
    _parse_serve_params,
)

SERVE = (
    "nohup vllm serve /data/models/GLM-5.3-MTP-Mix --served-model-name cltech/glm53 "
    "--host 0.0.0.0 --port 8000 -tp 8 -dcp 8 --enable-prefix-caching "
    "--max-model-len 1048576 --block-size 256 --max-num-seqs 4 "
    "--max-num-batched-tokens 4096 --moe-ep-size 2 "
    '--speculative-config \'{"num_speculative_tokens":5,"method":"deepseek_mtp"}\' &'
)
DOCKER = (
    "docker run -itd --name workflow_vllm_glm53_stage --net=host "
    "--shm-size 256G -v /data:/data "
    "hangzhou-office-harbor.infraai.top/qingting/vllm_glm:20261111_base1102"
)


def test_parse_serve_params():
    params = _parse_serve_params(SERVE)
    assert params["tensor_parallel_size"] == 8
    assert params["decode_context_parallel_size"] == 8
    assert params["block_size"] == 256
    assert params["max_num_seqs"] == 4
    assert params["max_num_batched_tokens"] == 4096
    assert params["moe_ep_size"] == 2
    assert params["num_speculative_tokens"] == 5
    assert params["speculative_method"] == "deepseek_mtp"
    assert params["enable_prefix_caching"] is True
    assert "host" not in params and "port" not in params


def test_image_and_config_name():
    params = _parse_serve_params(SERVE)
    assert _image_from_docker(DOCKER) == "hangzhou-office-harbor.infraai.top/qingting/vllm_glm:20261111_base1102"
    assert _auto_config_name("cltech/glm53", DOCKER, params, "evalscope") == "glm53-tp8-dcp8-mtp5-stage"


def test_parse_deploy_dir(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    (tmp_path / "env.sh").write_text("export VLLM_USE_V2_MODEL_RUNNER=1\n", encoding="utf-8")
    (commands / "01-docker.sh").write_text(DOCKER + "\n", encoding="utf-8")
    (commands / "02-vllm.sh").write_text(SERVE + "\n", encoding="utf-8")
    (commands / "03-ray.sh").write_text(
        "docker exec x ray start --head --port=6379\n", encoding="utf-8"
    )
    deploy = _parse_deploy_dir(tmp_path)
    assert deploy["docker_cmd"] == DOCKER
    assert "vllm serve" in deploy["serve_cmd"]
    assert deploy["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert deploy["extra_cmds"][0]["title"] == "Ray head"


def test_case_groups_skip_warmup(tmp_path):
    (tmp_path / "case1").mkdir()
    (tmp_path / "case1" / "benchmark_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "warmup").mkdir()
    (tmp_path / "warmup" / "benchmark_summary.json").write_text("{}", encoding="utf-8")
    groups = _case_groups({"tool": "evalscope"}, tmp_path)
    assert [g["case_key"] for g in groups] == ["case1"]


def _part(root: Path, case_key: str, started: str, ended: str) -> dict:
    part = root / case_key
    part.mkdir(parents=True)
    normalized = {
        "model": {"name": "cltech/glm53"},
        "config": {"name": "glm53-tp8", "params": {}},
        "suite_key": "evalscope",
        "node_label": "192.168.80.10",
        "backend": "generic-json",
        "valid": True,
        "started_at": started,
        "ended_at": ended,
        "cases": [{"case_key": case_key, "metrics": {"out_tps": 1.0}}],
    }
    manifest = {
        "model": {"name": "cltech/glm53"},
        "config": {"name": "glm53-tp8", "params": {}},
        "suite_key": "evalscope",
        "backend": "generic-json",
        "node": "192.168.80.10",
    }
    (part / "normalized.json").write_text(json.dumps(normalized), encoding="utf-8")
    (part / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw = part / "raw-results"
    raw.mkdir()
    (raw / "benchmark_summary.json").write_text("{}", encoding="utf-8")
    return {"case_key": case_key, "input": f"/raw/{case_key}", "out": str(part)}


def test_merge_parts(tmp_path):
    parts = [
        _part(tmp_path / "parts", "case1", "2026-11-26T10:00:00+08:00", "2026-11-26T10:05:00+08:00"),
        _part(tmp_path / "parts", "case2", "2026-11-26T10:06:00+08:00", "2026-11-26T10:10:00+08:00"),
    ]
    warmup = tmp_path / "warmup.json"
    warmup.write_text("{}", encoding="utf-8")
    parts[0]["warmup"] = str(warmup)
    parts[0]["warmup_command"] = "evalscope perf --warmup-num 1"
    out = tmp_path / "out"
    manifest, normalized = _merge_parts(parts, out, {}, "evalscope")
    assert [c["case_key"] for c in normalized["cases"]] == ["case1", "case2"]
    assert normalized["started_at"] == "2026-11-26T10:00:00+08:00"
    assert normalized["ended_at"] == "2026-11-26T10:10:00+08:00"
    assert (out / "raw-results" / "case1" / "benchmark_summary.json").is_file()
    assert (out / "raw-results" / "case2" / "benchmark_summary.json").is_file()
    assert normalized["cases"][0]["warmup_log"] == "case-logs/case1/warmup.json"
    assert manifest["config"]["deploy"]["extra_cmds"][0]["title"] == "case1 · warmup"
