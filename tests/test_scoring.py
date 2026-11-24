import json

import pytest

from app.scoring import load_comparison
from conftest import make_catalog, make_run, make_scenario


def setup(conn, case_keys=("c1",)):
    model, suite, cases = make_catalog(conn, case_keys=case_keys)
    sid = make_scenario(conn, model, suite)
    return model, suite, cases, sid


def m(ttft=None, tps=None):
    return {"ttft_mean_s": ttft, "out_tps": tps}


class TestRepresentatives:
    def test_mean_and_best(self, conn):
        model, suite, cases, sid = setup(conn)
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(10, 100)])
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(20, 80)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [m(30, 50)])
        comp = load_comparison(conn, sid)
        case_out = comp["suites"][0]["cases"][0]
        cfg_ids = case_out["configs_with_data"]
        a_id = next(cid for cid in cfg_ids if comp["configs"][cid]["name"] == "A")
        cell = case_out["cells"][a_id]["ttft_mean_s"]
        assert cell["same"]["n1"]["value"] == 15.0
        assert cell["same"]["n1"]["n_runs"] == 2
        assert cell["best"]["value"] == 10.0
        tps_cell = case_out["cells"][a_id]["out_tps"]
        assert tps_cell["best"]["value"] == 100.0

    def test_excluded_and_invalid_ignored(self, conn):
        model, suite, cases, sid = setup(conn)
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(10, 100)])
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(1, 999)], excluded=True)
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(2, 999)], valid=False)
        comp = load_comparison(conn, sid)
        case_out = comp["suites"][0]["cases"][0]
        a_id = case_out["configs_with_data"][0]
        cell = case_out["cells"][a_id]["ttft_mean_s"]
        assert cell["best"]["value"] == 10.0
        assert cell["same"]["n1"]["n_runs"] == 1


class TestPairwise:
    def _two_config(self, conn, a_nodes, b_nodes):
        model, suite, cases, sid = setup(conn)
        for node, metrics in a_nodes:
            make_run(conn, model, suite, cases, "A", {"v": 1}, node, [metrics])
        for node, metrics in b_nodes:
            make_run(conn, model, suite, cases, "B", {"v": 2}, node, [metrics])
        comp = load_comparison(conn, sid)
        return comp, comp["suites"][0]

    def test_clean_win(self, conn):
        comp, suite_out = self._two_config(
            conn, [("n1", m(10, 100)), ("n2", m(12, 90))],
            [("n1", m(20, 50)), ("n2", m(24, 45))])
        pw = suite_out["cases"][0]["pairwise"][0]
        for metric in ("ttft_mean_s", "out_tps"):
            d = pw["metrics"][metric]
            assert d["win_rate_same"] == 1.0
            assert d["win_rate_best"] == 1.0
            assert d["metric_score"] == 3.0
            assert d["nodes_compared"] == 2
        assert pw["case_score"] == 3.0
        ranking = suite_out["ranking"]
        assert ranking[0]["name"] == "A"
        assert ranking[0]["total_score"] == 3.0
        assert ranking[1]["total_score"] == 0.0
        assert suite_out["recommended_config_id"] == ranking[0]["config_id"]

    def test_epsilon_tie(self, conn):
        comp, suite_out = self._two_config(
            conn, [("n1", m(100.0, None))], [("n1", m(100.5, None))])
        pw = suite_out["cases"][0]["pairwise"][0]
        d = pw["metrics"]["ttft_mean_s"]
        assert d["win_rate_same"] == 0.5
        assert d["win_rate_best"] == 0.5
        assert d["metric_score"] == 1.5
        assert pw["case_score"] == 1.5

    def test_epsilon_boundary_outside(self, conn):
        comp, suite_out = self._two_config(
            conn, [("n1", m(100.0, None))], [("n1", m(102.0, None))])
        d = suite_out["cases"][0]["pairwise"][0]["metrics"]["ttft_mean_s"]
        assert d["win_rate_same"] == 1.0

    def test_split_nodes(self, conn):
        comp, suite_out = self._two_config(
            conn, [("n1", m(10, None)), ("n2", m(30, None))],
            [("n1", m(20, None)), ("n2", m(20, None))])
        d = suite_out["cases"][0]["pairwise"][0]["metrics"]["ttft_mean_s"]
        assert d["win_rate_same"] == 0.5
        assert d["win_rate_best"] == 1.0
        assert d["metric_score"] == 2.0
        assert {r["node"]: r["score"] for r in d["node_results"]} == {"n1": 1.0, "n2": 0.0}

    def test_no_common_nodes(self, conn):
        comp, suite_out = self._two_config(
            conn, [("n1", m(10, None))], [("n2", m(20, None))])
        d = suite_out["cases"][0]["pairwise"][0]["metrics"]["ttft_mean_s"]
        assert d["win_rate_same"] is None
        assert d["nodes_compared"] == 0
        assert d["metric_score"] == 1.0

    def test_missing_metric_skipped(self, conn):
        comp, suite_out = self._two_config(
            conn, [("n1", m(10, 100))], [("n1", m(None, 50))])
        pw = suite_out["cases"][0]["pairwise"][0]
        assert "ttft_mean_s" not in pw["metrics"]
        assert pw["metrics"]["out_tps"]["metric_score"] == 3.0
        assert pw["case_score"] == 3.0

    def test_no_data_case_excluded_from_total(self, conn):
        model, suite, cases, sid = setup(conn, case_keys=("c1", "c2"))
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(10, None), m(None, None)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [m(20, None), m(None, None)])
        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        assert suite_out["ranking"][0]["cases_compared"] == 1
        assert suite_out["ranking"][0]["total_score"] == 3.0


class TestWeights:
    def test_case_weight(self, conn):
        model, suite, cases, sid = setup(conn, case_keys=("c1", "c2"))
        conn.execute("UPDATE test_case SET weight = 3 WHERE id = ?", (cases[1]["id"],))
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(10, None), m(30, None)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [m(20, None), m(20, None)])
        comp = load_comparison(conn, sid)
        ranking = comp["suites"][0]["ranking"]
        a = next(r for r in ranking if r["name"] == "A")
        assert a["total_score"] == pytest.approx((1 * 3.0 + 3 * 0.0) / 4)

    def test_metric_weight(self, conn):
        model, suite, cases, sid = setup(conn)
        conn.execute("UPDATE scenario SET scoring = ? WHERE id = ?",
                     (json.dumps({"metric_weights": {"ttft_mean_s": 3, "out_tps": 1}}), sid))
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(10, 50)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [m(20, 100)])
        comp = load_comparison(conn, sid)
        pw = comp["suites"][0]["cases"][0]["pairwise"][0]
        assert pw["case_score"] == pytest.approx((3 * 3.0 + 1 * 0.0) / 4)

    def test_custom_weights_and_epsilon(self, conn):
        model, suite, cases, sid = setup(conn)
        conn.execute("UPDATE scenario SET scoring = ? WHERE id = ?",
                     (json.dumps({"w_same": 1, "w_best": 1, "epsilon": 0.05}), sid))
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(100, None)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [m(103, None)])
        comp = load_comparison(conn, sid)
        d = comp["suites"][0]["cases"][0]["pairwise"][0]["metrics"]["ttft_mean_s"]
        assert d["metric_score"] == 1.0


class TestInsufficient:
    def test_single_config(self, conn):
        model, suite, cases, sid = setup(conn)
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(10, 100)])
        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        assert suite_out["recommended_config_id"] is None
        assert any("待补充对比配置" in s for s in suite_out["insufficient"])
        assert suite_out["ranking"][0]["total_score"] is None

    def test_empty_suite(self, conn):
        model, suite, cases, sid = setup(conn)
        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        assert suite_out["config_ids"] == []
        assert suite_out["insufficient"]

    def test_partial_suite_runs_allowed(self, conn):
        model, suite, cases, sid = setup(conn, case_keys=("c1", "c2", "c3"))
        make_run(conn, model, suite, cases[:1], "A", {"v": 1}, "n1", [m(10, None)])
        make_run(conn, model, suite, cases[:1], "B", {"v": 2}, "n1", [m(20, None)])
        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        assert suite_out["ranking"][0]["total_score"] == 3.0
        assert len(suite_out["cases"]) == 3

class TestNodeAdjustment:
    """配置只跑过不同节点时，用同配置跨节点用例估计的节点倍率做校正比较。"""

    def _pair(self, conn, a_value, b_value, evidence_cases=3, reverse=False):
        case_keys = ("c1",) + tuple(f"e{i}" for i in range(evidence_cases))
        model, suite, cases = make_catalog(conn, case_keys=case_keys)
        sid = make_scenario(conn, model, suite)
        # 配置 C 在 c2.. 上同时跑 n1/n2，n2 的 ttft 是 n1 的 2 倍。
        for case in cases[1:]:
            make_run(conn, model, suite, [case], "C", {"v": 3}, "n1",
                     [m(10 if not reverse else 20, None)])
            make_run(conn, model, suite, [case], "C", {"v": 3}, "n2",
                     [m(20 if not reverse else 10, None)])
        make_run(conn, model, suite, [cases[0]], "A", {"v": 1}, "n1",
                 [m(a_value, None)])
        make_run(conn, model, suite, [cases[0]], "B", {"v": 2}, "n2",
                 [m(b_value, None)])
        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        cfg_ids = {comp["configs"][cid]["name"]: cid for cid in comp["configs"]}
        pw = next(p for p in suite_out["cases"][0]["pairwise"]
                  if {p["a"], p["b"]} == {cfg_ids["A"], cfg_ids["B"]})
        detail = pw["metrics"]["ttft_mean_s"]
        return comp, detail, cfg_ids

    def test_disjoint_nodes_tie_after_adjustment(self, conn):
        comp, detail, _ = self._pair(conn, 10, 20)
        assert detail["node_mode"] == "adjusted"
        assert detail["win_rate_same"] is None
        assert detail["win_rate_adj"] == pytest.approx(0.5)
        assert detail["node_confidence"] == pytest.approx(1.0)
        assert detail["win_rate_best"] == pytest.approx(0.5)
        assert detail["metric_score"] == pytest.approx(1.5)
        effect = comp["suites"][0]["node_effects"]["ttft_mean_s"]
        assert effect["factor"]["n2"] == pytest.approx(2.0)

    def test_node_effects_are_scoped_per_suite(self, conn):
        from app.entities import ensure_case, ensure_suite

        model, suite_a, cases_a = make_catalog(conn, suite_key="sa", case_keys=("a1",))
        sid = make_scenario(conn, model, suite_a)
        suite_b = ensure_suite(conn, "sb")
        cases_b = [ensure_case(conn, suite_b["id"], "b1", sort_order=0)]
        conn.execute("INSERT INTO scenario_suite(scenario_id, suite_id) VALUES(?,?)",
                     (sid, suite_b["id"]))

        # suite B 有同配置跨节点证据：n2 的 ttft 是 n1 的 2 倍
        make_run(conn, model, suite_b, cases_b, "C", {"v": 3}, "n1", [m(10, None)])
        make_run(conn, model, suite_b, cases_b, "C", {"v": 3}, "n2", [m(20, None)])

        # suite A 的配置只跑过不同节点，不应借用 suite B 的节点倍率
        make_run(conn, model, suite_a, cases_a, "A", {"v": 1}, "n1", [m(10, None)])
        make_run(conn, model, suite_a, cases_a, "B", {"v": 2}, "n2", [m(20, None)])

        comp = load_comparison(conn, sid)
        out_a = next(s for s in comp["suites"] if s["suite"]["key"] == "sa")
        out_b = next(s for s in comp["suites"] if s["suite"]["key"] == "sb")
        assert out_b["node_effects"]["ttft_mean_s"]["factor"]["n2"] == pytest.approx(2.0)
        assert "n2" not in out_a["node_effects"].get("ttft_mean_s", {}).get("known", [])
        detail = out_a["cases"][0]["pairwise"][0]["metrics"]["ttft_mean_s"]
        assert detail["node_mode"] is None

    def test_disjoint_nodes_win_after_adjustment(self, conn):
        comp, detail, _ = self._pair(conn, 10, 30)
        assert detail["node_mode"] == "adjusted"
        assert detail["win_rate_adj"] == pytest.approx(1.0)
        assert detail["win_rate_best"] == pytest.approx(1.0)
        assert detail["metric_score"] == pytest.approx(3.0)

    def test_adjustment_discounted_by_evidence(self, conn):
        comp, detail, _ = self._pair(conn, 10, 20, evidence_cases=1)
        assert detail["node_mode"] == "adjusted"
        assert detail["node_confidence"] == pytest.approx(1 / 3)
        assert detail["win_rate_adj"] == pytest.approx(0.5)
        assert detail["win_rate_best"] == pytest.approx(0.5 / 3 + 2 / 3)
        assert detail["metric_score"] == pytest.approx(2 * (1 / 3) * 0.5 + (0.5 / 3 + 2 / 3))


class TestPartialCoverageAndFallback:
    def test_rank_on_shared_cases_only_and_fallback_source(self, conn):
        from app.web import _case_decision_notes, best_data_view

        model, suite, cases = make_catalog(conn, case_keys=("c1", "c2"))
        sid = make_scenario(conn, model, suite)
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1",
                 [m(10, None), m(None, None)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1",
                 [m(20, None), m(30, None)])

        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        assert suite_out["compared_cases"] == ["c1"]
        assert suite_out["ranking"][0]["name"] == "A"
        assert suite_out["ranking"][0]["cases_compared"] == 1

        c2_out = next(c for c in suite_out["cases"] if c["case"]["case_key"] == "c2")
        assert any("不参与横向对比" in n for n in _case_decision_notes(c2_out))

        view = best_data_view(comp, suite_out["ranking"][0]["config_id"])
        c2 = next(c for c in view[0]["cases"] if c["case"]["case_key"] == "c2")
        metric = c2["metrics"][0]
        assert metric["fallback"] is True
        assert metric["source_config_name"] == "B"
        assert metric["source_rank"] == 2
        assert metric["best"]["value"] == 30

    def test_best_data_hides_cases_with_missing_metrics(self, conn):
        from app.web import best_data_view

        model, suite, cases = make_catalog(conn, case_keys=("c1", "c2"))
        sid = make_scenario(conn, model, suite)
        make_run(conn, model, suite, [cases[0]], "A", {"v": 1}, "n1", [m(10, None)])

        comp = load_comparison(conn, sid)
        suite_out = comp["suites"][0]
        view = best_data_view(comp, suite_out["ranking"][0]["config_id"])
        assert [c["case"]["case_key"] for c in view[0]["cases"]] == ["c1"]


class TestPkView:
    def test_disjoint_coverage_has_no_pk(self, conn):
        from app.web import build_pk_view

        model, suite, cases = make_catalog(conn, case_keys=("c1", "c2"))
        sid = make_scenario(conn, model, suite)
        make_run(conn, model, suite, [cases[0]], "A", {"v": 1}, "n1", [m(10, None)])
        make_run(conn, model, suite, [cases[0]], "B", {"v": 2}, "n1", [m(20, None)])
        make_run(conn, model, suite, [cases[1]], "C", {"v": 3}, "n1", [m(30, None)])

        comp = load_comparison(conn, sid)
        ids = {comp["configs"][cid]["name"]: cid for cid in comp["configs"]}
        pk = build_pk_view(comp, ids["A"], ids["C"])
        assert pk["compared_count"] == 0
        assert pk["coverage_a"] == [("s1", 1)]
        assert pk["coverage_b"] == [("s1", 1)]

        pk2 = build_pk_view(comp, ids["A"], ids["B"])
        assert pk2["compared_count"] == 1
        assert pk2["wins"] == 1
        assert pk2["comparisons"][0]["status"] == "win"


class TestPkGaps:
    def test_config_without_shared_case_is_reported(self, conn):
        from app.entities import ensure_case, ensure_suite
        from app.queries import pk_gap_configs

        model, suite_a, cases_a = make_catalog(conn, suite_key="sa", case_keys=("a1",))
        sid = make_scenario(conn, model, suite_a)
        suite_b = ensure_suite(conn, "sb")
        cases_b = [ensure_case(conn, suite_b["id"], "b1", sort_order=0)]
        conn.execute("INSERT INTO scenario_suite(scenario_id, suite_id) VALUES(?,?)",
                     (sid, suite_b["id"]))
        make_run(conn, model, suite_a, cases_a, "A", {"v": 1}, "n1", [m(10, None)])
        make_run(conn, model, suite_b, cases_b, "B", {"v": 2}, "n1", [m(20, None)])
        cfg_a = conn.execute("SELECT id FROM config_set WHERE name='A'").fetchone()["id"]
        gaps = pk_gap_configs(conn, sid, cfg_a)
        assert [g["name"] for g in gaps] == ["B"]


class TestAggregateGroupPk:
    def test_inner_top_variant_is_used_for_cross_group_pk(self, conn):
        from app.web import build_config_groups, build_group_inner, build_group_pk

        model, suite, cases = make_catalog(conn, case_keys=("c1", "c2"))
        sid = make_scenario(conn, model, suite)
        core = {"tensor_parallel_size": 16, "decode_context_parallel_size": 4,
                "long_prefill_token_threshold": 1024, "image": "vllm_glm:20261102_base1102"}
        make_run(conn, model, suite, [cases[0]], "A1",
                 {**core, "compilation_config": '{"cudagraph_mode":"FULL_DECODE_ONLY"}'},
                 "n1", [m(10, None)])
        make_run(conn, model, suite, cases, "A2",
                 {**core, "compilation_config": '{"cudagraph_mode":"NONE"}'},
                 "n1", [m(20, None), m(10, None)])
        make_run(conn, model, suite, cases, "B",
                 {"tensor_parallel_size": 8, "decode_context_parallel_size": 8,
                  "long_prefill_token_threshold": 1024,
                  "image": "vllm_glm:20261102_base1102"}, "n1",
                 [m(30, None), m(30, None)])

        comp = load_comparison(conn, sid)
        groups = build_config_groups(comp)
        assert len(groups) == 2
        group_a = next(g for g in groups if any(c["name"] == "A1" for c in g["configs"]))
        group_b = next(g for g in groups if any(c["name"] == "B" for c in g["configs"]))
        assert [c["name"] for c in group_a["configs"]] == ["A1", "A2"]

        inner = build_group_inner(comp, group_a)
        assert inner["top"]["name"] == "A1"
        assert inner["top_score"] == 3.0
        assert inner["top_stats"]["wins"] == 1
        assert inner["top_stats"]["losses"] == 0

        gp = build_group_pk(comp, group_a, group_b)
        assert gp["top_a"]["name"] == "A1"
        assert gp["top_b"]["name"] == "B"
        assert gp["compared_count"] == 1
        assert gp["wins"] == 1
        assert "variant_pairs" not in gp

    def test_pk_view_marks_equal_case_scores_as_tie(self, conn):
        from app.web import build_pk_view

        model, suite, cases = make_catalog(conn, case_keys=("c1",))
        sid = make_scenario(conn, model, suite)
        make_run(conn, model, suite, cases, "A", {"v": 1}, "n1", [m(100.0, None)])
        make_run(conn, model, suite, cases, "B", {"v": 2}, "n1", [m(100.5, None)])
        comp = load_comparison(conn, sid)
        ids = {comp["configs"][cid]["name"]: cid for cid in comp["configs"]}
        pk = build_pk_view(comp, ids["A"], ids["B"])
        assert pk["comparisons"][0]["status"] == "tie"
        assert pk["ties"] == 1
        assert pk["wins"] == pk["losses"] == 0


class TestGroupComparePage:
    def test_single_group_renders_inner_variant_pk(self, client, conn):
        model, suite, cases = make_catalog(
            conn, model_name="single/model", case_keys=("c1", "c2"))
        sid = make_scenario(conn, model, suite)
        core = {"tensor_parallel_size": 16, "decode_context_parallel_size": 4,
                "long_prefill_token_threshold": 1024,
                "image": "vllm_glm:20261102_base1102"}
        make_run(conn, model, suite, [cases[0]], "A1",
                 {**core, "compilation_config": '{"cudagraph_mode":"FULL_DECODE_ONLY"}'},
                 "n1", [m(10, None)])
        make_run(conn, model, suite, cases, "A2",
                 {**core, "compilation_config": '{"cudagraph_mode":"NONE"}'},
                 "n1", [m(20, None), m(10, None)])
        conn.commit()

        resp = client.get(f"/models/single%2Fmodel/scenarios/{sid}/compare")
        assert resp.status_code == 200
        assert "组内变体 PK 矩阵（仅 1 个聚合配置组）" in resp.text
        assert "组内 Top" in resp.text
        assert "A1" in resp.text
