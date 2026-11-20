import json

from app.entities import (config_fingerprint, normalize_params, parse_serve_flags,
                          register_config, resolve_model)


class TestFingerprint:
    def test_flag_order_insensitive(self):
        a = "-tp 8 -dcp 8 --block-size 256 --max-num-seqs 4 --max-num-batched-tokens 4096 --moe-ep-size 2"
        b = "--moe-ep-size 2 --max-num-batched-tokens 4096 -dcp 8 --block-size 256 --max-num-seqs 4 -tp 8"
        assert config_fingerprint(a) == config_fingerprint(b)

    def test_string_equals_dict(self):
        s = "-tp 8 -dcp 8 --block-size 256"
        d = {"tensor_parallel_size": 8, "decode_context_parallel_size": 8, "block_size": 256}
        assert config_fingerprint(s) == config_fingerprint(d)

    def test_threshold_difference_splits(self):
        base = {"num_gpu_blocks_override": 1024, "block_size": 256, "max_model_len": "1020k",
                "decode_context_parallel_size": 4, "moe_ep_size": 2, "num_speculative_tokens": 5}
        a = dict(base, long_prefill_token_threshold=1024)
        b = dict(base, long_prefill_token_threshold=2048)
        assert config_fingerprint(a) != config_fingerprint(b)

    def test_dict_key_order_insensitive(self):
        assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})

    def test_numeric_string_normalization(self):
        assert config_fingerprint({"tp": "8"}) == config_fingerprint({"tp": 8})

    def test_equals_form(self):
        assert config_fingerprint("--block-size=256 -tp=8") == config_fingerprint({"block_size": 256, "tp": 8})

    def test_boolean_flag(self):
        flags = parse_serve_flags("--enable-prefix-caching -tp 8")
        assert flags["enable_prefix_caching"] is True
        assert flags["tensor_parallel_size"] == 8

    def test_positional_args(self):
        flags = parse_serve_flags("vllm serve /data/models/GLM-5.3-MTP-Mix -tp 8")
        assert flags["_args"] == ["vllm", "serve", "/data/models/GLM-5.3-MTP-Mix"]
        assert flags["tensor_parallel_size"] == 8

    def test_normalize_nested(self):
        norm = normalize_params({"z": {"b": 2, "a": 1}, "a": [1, "2"]})
        assert list(norm) == ["a", "z"]
        assert norm["a"] == [1, 2]


class TestModelAlias:
    def test_alias_resolution(self, conn):
        m1 = resolve_model(conn, "cltech/glm53", ["GLM-5.3-MTP-Mix"])
        m2 = resolve_model(conn, "GLM-5.3-MTP-Mix")
        assert m1["id"] == m2["id"]
        assert m2["name"] == "cltech/glm53"

    def test_new_alias_merged(self, conn):
        m1 = resolve_model(conn, "cltech/glm53", ["GLM-5.3-MTP-Mix"])
        m2 = resolve_model(conn, "cltech/glm53", ["/data/models/GLM-5.3-MTP-Mix"])
        assert m1["id"] == m2["id"]
        assert "/data/models/GLM-5.3-MTP-Mix" in json.loads(m2["aliases"])

    def test_distinct_model(self, conn):
        m1 = resolve_model(conn, "a")
        m2 = resolve_model(conn, "b")
        assert m1["id"] != m2["id"]


class TestConfigDedup:
    def test_same_fingerprint_reused(self, conn):
        model = resolve_model(conn, "m")
        c1, created1 = register_config(conn, model["id"], "name-1", {"tp": 8})
        c2, created2 = register_config(conn, model["id"], "name-2", "-tp 8")
        assert created1 is True
        assert created2 is False
        assert c1["id"] == c2["id"]
        assert c2["name"] == "name-1"

    def test_different_params_new(self, conn):
        model = resolve_model(conn, "m")
        c1, _ = register_config(conn, model["id"], "a", {"tp": 8})
        c2, _ = register_config(conn, model["id"], "b", {"tp": 16})
        assert c1["id"] != c2["id"]