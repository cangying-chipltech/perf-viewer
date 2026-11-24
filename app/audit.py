"""配置集一致性：部署信息审计与配置身份重算。

历史问题：perf-viewer 过去用 params 的哈希作为配置集指纹，而部署记录（serve 命令、
镜像、环境变量）不在其中。两次 serve 命令不同（例如 cudagraph_mode 从
FULL_DECODE_ONLY 改成 NONE）但声明参数相同的导入会命中同一个配置集，后一次导入再把
配置集上的 deploy 覆盖掉，于是早先那些 run 在页面上被展示成一条它们从未执行过的命令。

本模块提供两件事：
- ``deploy_audit``：逐 run 比较它导入时记录的部署与所属配置集当前的 deploy，暴露不一致；
- ``recompute_config_identity``：用 params ∪ serve 命令 ∪ 环境变量重算配置身份，
  合并因旧算法产生的重复配置集。
"""
import json

from .entities import _merge_deploy, config_identity, normalize_deploy, serve_params

REDACTED = "***"


def _run_deploy(manifest_json: str) -> dict:
    try:
        manifest = json.loads(manifest_json or "{}")
    except json.JSONDecodeError:
        return {}
    deploy = (((manifest.get("import_manifest") or {}).get("config") or {})
              .get("deploy")) or {}
    return deploy if isinstance(deploy, dict) else {}


def _flag_map(serve_cmd: str) -> dict:
    return serve_params(serve_cmd)


def _env_differs(left: dict, right: dict) -> list[str]:
    keys = set(left) | set(right)
    out = []
    for key in sorted(keys):
        a, b = str(left.get(key, "")), str(right.get(key, ""))
        if a == b or a == REDACTED or b == REDACTED:
            continue
        out.append(key)
    return out


def _compare(config_deploy: dict, run_deploy: dict) -> dict:
    try:
        config_deploy = normalize_deploy(config_deploy) if config_deploy else {}
        run_deploy = normalize_deploy(run_deploy) if run_deploy else {}
    except ValueError:
        return {"evidence": True, "serve_flags_differ": [], "env_keys_differ": [],
                "docker_differs": False, "invalid": True, "ok": False}
    config_serve = str(config_deploy.get("serve_cmd") or "")
    run_serve = str(run_deploy.get("serve_cmd") or "")
    result = {"evidence": bool(run_serve or run_deploy.get("docker_cmd")
                               or run_deploy.get("env")), "invalid": False}
    cfg_flags = _flag_map(config_serve)
    run_flags = _flag_map(run_serve)
    if config_serve and run_serve:
        result["serve_flags_differ"] = sorted(
            key for key in set(cfg_flags) | set(run_flags)
            if cfg_flags.get(key) != run_flags.get(key)
        )
    else:
        result["serve_flags_differ"] = [] if config_serve == run_serve else ["serve_cmd"]
    result["env_keys_differ"] = _env_differs(
        config_deploy.get("env") or {}, run_deploy.get("env") or {})
    config_docker = " ".join(str(config_deploy.get("docker_cmd") or "").split())
    run_docker = " ".join(str(run_deploy.get("docker_cmd") or "").split())
    result["docker_differs"] = bool(
        config_docker and run_docker and config_docker != run_docker)
    result["ok"] = not (result["serve_flags_differ"] or result["env_keys_differ"]
                        or result["docker_differs"])
    return result


def deploy_audit(conn, config_ids: list[int] | None = None) -> dict:
    """Per-config list of runs whose recorded deployment disagrees with the config row.

    ``config_ids`` narrows both the config rows and the run query, so page paths can
    audit only the selected configurations instead of scanning all history.
    """
    all_configs = {row["id"]: dict(row) for row in conn.execute("SELECT * FROM config_set")}
    if config_ids is None:
        configs = all_configs
        runs = conn.execute(
            "SELECT r.id, r.config_id, r.excluded, r.valid, r.manifest, n.label AS node"
            " FROM run r JOIN node n ON n.id = r.node_id ORDER BY r.id").fetchall()
    else:
        configs = {cid: all_configs[cid] for cid in config_ids if cid in all_configs}
        if not configs:
            return {"ok": True, "configs": [], "mismatched_runs": []}
        placeholders = ",".join("?" * len(configs))
        runs = conn.execute(
            "SELECT r.id, r.config_id, r.excluded, r.valid, r.manifest, n.label AS node"
            f" FROM run r JOIN node n ON n.id = r.node_id"
            f" WHERE r.config_id IN ({placeholders}) ORDER BY r.id",
            tuple(configs)).fetchall()
    out: dict = {"ok": True, "configs": [], "mismatched_runs": []}
    per_config: dict[int, dict] = {}
    for run in runs:
        cfg = configs.get(run["config_id"])
        if cfg is None:
            continue
        entry = per_config.setdefault(run["config_id"], {
            "config_id": run["config_id"],
            "config_name": cfg["name"],
            "deploy_compilation": _flag_map(
                (json.loads(cfg["deploy"] or "{}")).get("serve_cmd", "")).get(
                    "compilation_config"),
            "runs": [],
        })
        comparison = _compare(json.loads(cfg["deploy"] or "{}"), _run_deploy(run["manifest"]))
        item = {"run": run["id"], "node": run["node"], "excluded": run["excluded"],
                "valid": run["valid"], **comparison}
        entry["runs"].append(item)
        if comparison["evidence"] and not comparison["ok"]:
            out["ok"] = False
            out["mismatched_runs"].append({
                "run": run["id"], "config_id": run["config_id"],
                "config_name": cfg["name"], "node": run["node"],
                "excluded": run["excluded"],
                "serve_flags_differ": comparison["serve_flags_differ"],
                "env_keys_differ": comparison["env_keys_differ"],
                "docker_differs": comparison["docker_differs"],
                "invalid": comparison["invalid"],
            })
    out["configs"] = list(per_config.values())
    return out


def _rewrite_scenario_config_ids(conn, dropped: int, kept: int) -> None:
    """Replace a dropped config id in scenario whitelists and best-history rows."""
    for row in conn.execute("SELECT id, config_ids FROM scenario").fetchall():
        try:
            values = json.loads(row["config_ids"] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(values, list):
            continue
        updated: list = []
        changed = False
        for raw in values:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                updated.append(raw)
                continue
            if value == dropped:
                value = kept
                changed = True
            if value in updated:
                changed = True
                continue
            updated.append(value)
        if changed:
            conn.execute("UPDATE scenario SET config_ids = ? WHERE id = ?",
                         (json.dumps(updated, ensure_ascii=False), row["id"]))
    conn.execute("UPDATE best_history SET prev_config_id = ? WHERE prev_config_id = ?",
                 (kept, dropped))
    conn.execute("UPDATE best_history SET new_config_id = ? WHERE new_config_id = ?",
                 (kept, dropped))


def recompute_config_identity(conn) -> dict:
    """Re-fingerprint every configuration from params + deploy and merge duplicates.

    Must run after the deployment record of each configuration is corrected: the
    fingerprint follows the deploy, so a wrong deploy would produce a wrong
    identity.
    """
    rows = conn.execute("SELECT * FROM config_set ORDER BY id").fetchall()
    groups: dict[tuple, list] = {}
    for row in rows:
        params = json.loads(row["params"] or "{}")
        deploy = json.loads(row["deploy"] or "{}")
        fp, merged = config_identity(params, deploy)
        groups.setdefault((row["model_id"], fp), []).append((dict(row), merged))
    report = {"updated": 0, "merged_configs": [], "runs_moved": 0}
    for (model_id, fp), items in groups.items():
        items.sort(key=lambda item: item[0]["id"])
        keep, keep_params = items[0]
        if keep["fingerprint"] != fp or json.loads(keep["params"] or "{}") != keep_params:
            conn.execute("UPDATE config_set SET fingerprint = ?, params = ? WHERE id = ?",
                         (fp, json.dumps(keep_params, ensure_ascii=False), keep["id"]))
            report["updated"] += 1
        keep_deploy = json.loads(keep["deploy"] or "{}")
        for row, _params in items[1:]:
            keep_deploy = _merge_deploy(keep_deploy, json.loads(row["deploy"] or "{}"))
            moved = conn.execute("UPDATE run SET config_id = ? WHERE config_id = ?",
                                 (keep["id"], row["id"])).rowcount
            report["runs_moved"] += max(moved, 0)
            conn.execute("UPDATE scenario SET best_config_id = ? WHERE best_config_id = ?",
                         (keep["id"], row["id"]))
            _rewrite_scenario_config_ids(conn, row["id"], keep["id"])
            conn.execute("DELETE FROM config_set WHERE id = ?", (row["id"],))
            report["merged_configs"].append({"dropped": row["id"], "kept": keep["id"],
                                             "name": row["name"]})
        if items[1:]:
            conn.execute("UPDATE config_set SET deploy = ? WHERE id = ?",
                         (json.dumps(keep_deploy, ensure_ascii=False), keep["id"]))
    return report
