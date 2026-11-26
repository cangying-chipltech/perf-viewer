(function () {
  "use strict";

  function getToken() { return localStorage.getItem("pv_token") || ""; }
  function authHeaders() {
    const t = getToken();
    return t ? { Authorization: "Bearer " + t } : {};
  }

  document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("pv-token");
    const state = document.getElementById("pv-token-state");
    if (input) {
      input.value = getToken();
      if (getToken()) state.textContent = "已设置";
      document.getElementById("pv-token-save").addEventListener("click", function () {
        localStorage.setItem("pv_token", input.value.trim());
        state.textContent = input.value.trim() ? "已设置" : "已清除";
      });
    }
    pvInitChart();
    pvInitPkChart();
  });

  document.addEventListener("click", async function (e) {
    const btn = e.target.closest("button[data-pk-toggle]");
    if (!btn) return;
    e.preventDefault();
    const target = document.querySelector(btn.getAttribute("data-pk-target"));
    if (btn.dataset.open === "1") {
      if (target) target.innerHTML = "";
      btn.dataset.open = "0";
      btn.textContent = "展示";
      return;
    }
    try {
      const resp = await fetch(btn.getAttribute("data-pk-url"), { headers: authHeaders() });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      if (target) {
        target.innerHTML = await resp.text();
        if (window.htmx && window.htmx.process) window.htmx.process(target);
      }
      btn.dataset.open = "1";
      btn.textContent = "收起";
    } catch (err) {
      window.alert("加载详情失败: " + err.message);
    }
  });

  document.addEventListener("click", async function (e) {
    const btn = e.target.closest("button[data-case-runs-toggle]");
    if (!btn) return;
    e.preventDefault();
    const target = document.querySelector(btn.getAttribute("data-target"));
    if (!target) return;
    if (btn.getAttribute("aria-expanded") === "true") {
      target.innerHTML = "";
      btn.setAttribute("aria-expanded", "false");
      btn.textContent = "节点执行数据";
      return;
    }
    try {
      const resp = await fetch(btn.getAttribute("data-url"), { headers: authHeaders() });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      target.innerHTML = await resp.text();
      if (window.htmx && window.htmx.process) window.htmx.process(target);
      btn.setAttribute("aria-expanded", "true");
      btn.textContent = "收起节点执行数据";
      target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } catch (err) {
      window.alert("加载节点执行数据失败: " + err.message);
    }
  });

  document.addEventListener("click", function (e) {
    const btn = e.target.closest("button[data-log-target]");
    if (!btn) return;
    const target = document.querySelector(btn.getAttribute("data-log-target"));
    if (!target) return;
    if (target.hasAttribute("hidden")) {
      target.removeAttribute("hidden");
      btn.textContent = "收起日志";
      target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } else {
      target.setAttribute("hidden", "");
      btn.textContent = "日志（" + (btn.dataset.count || "") + "）";
    }
  });

  document.addEventListener("DOMContentLoaded", function () {
    const targets = new Set();
    document.querySelectorAll("input[data-case-filter]").forEach(function (box) {
      targets.add(box.getAttribute("data-target"));
    });
    targets.forEach(applyCaseFilter);
  });

  document.addEventListener("htmx:configRequest", function (e) {
    const t = getToken();
    if (t) e.detail.headers["Authorization"] = "Bearer " + t;
  });

  document.addEventListener("htmx:afterSwap", function (e) {
    pvInitChart();
    pvInitPkChart();
    const target = e.detail && e.detail.target;
    if (target && target.id && (target.id.indexOf("case-drilldown-") === 0 ||
        target.id.indexOf("case-runs-") === 0)) {
      target.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });

  document.addEventListener("click", function (e) {
    const btn = e.target.closest("button[data-copy]");
    if (!btn) return;
    const el = document.querySelector(btn.getAttribute("data-copy"));
    if (!el) return;
    const text = el.textContent;
    const label = btn.getAttribute("data-label") || "复制";
    function done(ok) {
      btn.textContent = ok ? "已复制" : "复制失败";
      setTimeout(function () { btn.textContent = label; }, 1500);
    }
    function fallback() {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
      document.body.removeChild(ta);
      done(ok);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, fallback);
    } else {
      fallback();
    }
  });

  function applyCaseFilter(target) {
    const table = document.querySelector(target);
    if (!table) return;
    const boxes = Array.from(document.querySelectorAll(
      'input[data-case-filter][data-target="' + target + '"]'));
    const selected = new Set(boxes.filter(function (b) { return b.checked; })
      .map(function (b) { return b.value; }));
    let visible = 0;
    table.querySelectorAll("tr[data-case-key]").forEach(function (tr) {
      const show = selected.has(tr.getAttribute("data-case-key"));
      tr.style.display = show ? "" : "none";
      if (show && tr.hasAttribute("data-case-main")) visible += 1;
    });
    document.querySelectorAll('[data-case-count][data-target="' + target + '"]')
      .forEach(function (el) { el.textContent = "显示 " + visible + " / " + boxes.length; });
  }

  document.addEventListener("change", function (e) {
    const box = e.target.closest("input[data-case-filter]");
    if (box) applyCaseFilter(box.getAttribute("data-target"));
  });

  document.addEventListener("click", function (e) {
    const btn = e.target.closest("button[data-case-bulk]");
    if (!btn) return;
    const target = btn.getAttribute("data-target");
    const checked = btn.getAttribute("data-case-bulk") === "all";
    document.querySelectorAll('input[data-case-filter][data-target="' + target + '"]')
      .forEach(function (box) { box.checked = checked; });
    applyCaseFilter(target);
  });

  document.addEventListener("submit", async function (e) {
    const form = e.target.closest("form[data-api]");
    if (!form) return;
    e.preventDefault();
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) return;
    const body = {};
    for (const el of form.elements) {
      if (!el.name || el.tagName === "BUTTON") continue;
      let v = el.value;
      if (el.type === "checkbox") {
        if (!Array.isArray(body[el.name])) body[el.name] = [];
        if (el.checked) body[el.name].push(Number(el.value));
        continue;
      }
      if (el.dataset.json !== undefined) {
        try { v = JSON.parse(v); } catch (err) {
          window.alert("字段 " + el.name + " 不是合法 JSON: " + err.message);
          return;
        }
      } else if (el.dataset.list !== undefined) {
        v = v.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
      } else if (el.multiple) {
        v = Array.from(el.selectedOptions).map(function (o) { return Number(o.value); });
      } else if (el.type === "number") {
        v = Number(v);
      } else if (v === "true") { v = true; }
      else if (v === "false") { v = false; }
      body[el.name] = v;
    }
    if (form.dataset.deployForm !== undefined) {
      const env = {};
      String(body.env_text || "").split(/\r?\n/).forEach(function (line) {
        line = line.trim();
        if (!line || line.charAt(0) === "#") return;
        line.split(/\s+(?=[A-Za-z_][A-Za-z0-9_]*=)/).forEach(function (pair) {
          const i = pair.indexOf("=");
          if (i > 0) env[pair.slice(0, i).trim()] = pair.slice(i + 1).trim();
        });
      });
      body.deploy = {
        docker_cmd: body.docker_cmd || "",
        serve_cmd: body.serve_cmd || "",
        env: env,
      };
      if (body.source) body.deploy.source = body.source;
      delete body.docker_cmd;
      delete body.serve_cmd;
      delete body.env_text;
      delete body.source;
    }
    const method = (form.dataset.method || "POST").toUpperCase();
    try {
      const resp = await fetch(form.dataset.api, {
        method: method,
        headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
        body: method === "GET" || method === "DELETE" ? undefined : JSON.stringify(body),
      });
      const text = await resp.text();
      let data;
      try { data = JSON.parse(text); } catch (_) { data = text; }
      if (!resp.ok) {
        const detail = data && data.detail ? data.detail : text;
        window.alert("操作失败 HTTP " + resp.status + ": " + detail);
        return;
      }
      window.location.reload();
    } catch (err) {
      window.alert("请求失败: " + err.message);
    }
  });

  let pkChartInstance = null;

  function pvInitPkChart() {
    const dataEl = document.getElementById("pk-data");
    const container = document.getElementById("pk-chart");
    if (!dataEl || !container) return;
    if (typeof echarts === "undefined") {
      container.innerHTML = '<p class="empty">ECharts 未加载，无法绘制 PK 图。</p>';
      return;
    }
    let rows;
    try { rows = JSON.parse(dataEl.textContent); } catch (_) { return; }
    if (!rows.length) {
      container.innerHTML = '<p class="empty">无共同测试项，未进行 PK。</p>';
      return;
    }
    const colors = { win: "#2e9d5b", loss: "#c0392b", tie: "#8a8a8a" };
    if (pkChartInstance) { pkChartInstance.dispose(); }
    pkChartInstance = echarts.init(container);
    pkChartInstance.setOption({
      title: {
        text: "配置 A 逐测试项 PK 得分（>0.5 胜，<0.5 负）",
        left: "center", textStyle: { fontSize: 13 },
      },
      tooltip: { trigger: "axis" },
      grid: { left: 60, right: 30, bottom: 120, top: 50 },
      xAxis: {
        type: "category", data: rows.map(function (r) { return r.label; }),
        axisLabel: { rotate: 30, fontSize: 10, width: 200, overflow: "truncate" },
      },
      yAxis: { type: "value", min: 0, max: 1 },
      series: [{
        type: "bar",
        data: rows.map(function (r) {
          return { value: r.score, itemStyle: { color: colors[r.status] || "#1a6fb0" } };
        }),
        barMaxWidth: 50,
        label: { show: true, position: "top", fontSize: 10 },
        markLine: {
          silent: true, symbol: "none",
          data: [{ yAxis: 0.5, lineStyle: { type: "dashed", color: "#999" } }],
        },
      }],
    }, true);
  }

  let chartInstance = null;

  function pvInitChart() {
    const dataEl = document.getElementById("chart-data");
    const modeEl = document.getElementById("chart-mode");
    const container = document.getElementById("chart");
    const caseSel = document.getElementById("chart-case");
    const metricSel = document.getElementById("chart-metric");
    if (!dataEl || !container || !caseSel || !metricSel) return;
    if (typeof echarts === "undefined") {
      container.innerHTML = '<p class="empty">ECharts 未加载（vendor 缺失），请使用表格查看。</p>';
      return;
    }
    let data, mode;
    try {
      data = JSON.parse(dataEl.textContent);
      mode = JSON.parse(modeEl.textContent);
    } catch (_) { return; }
    const caseKeys = Object.keys(data);
    if (!caseKeys.length) {
      container.innerHTML = '<p class="empty">暂无图表数据。</p>';
      return;
    }
    const prevCase = caseSel.value;
    caseSel.innerHTML = "";
    caseKeys.forEach(function (k) {
      const opt = document.createElement("option");
      opt.value = k; opt.textContent = k;
      caseSel.appendChild(opt);
    });
    if (caseKeys.indexOf(prevCase) >= 0) caseSel.value = prevCase;

    function fillMetrics() {
      const metrics = Object.keys(data[caseSel.value].metrics);
      const prev = metricSel.value;
      metricSel.innerHTML = "";
      metrics.forEach(function (m) {
        const opt = document.createElement("option");
        opt.value = m; opt.textContent = m;
        metricSel.appendChild(opt);
      });
      if (metrics.indexOf(prev) >= 0) metricSel.value = prev;
    }

    function render() {
      const entry = data[caseSel.value];
      if (!entry) return;
      const series = entry.metrics[metricSel.value] || [];
      const names = series.map(function (s) { return s.name; });
      const values = series.map(function (s) {
        if (mode === "best") return s.best;
        if (mode === "adjusted") return s.adjusted_mean;
        return s.node_mean;
      });
      const modeLabel = mode === "best" ? "（历史最好值）"
        : (mode === "adjusted" ? "（节点校正后）" : "（节点代表值均值）");
      if (!chartInstance) chartInstance = echarts.init(container);
      chartInstance.setOption({
        title: {
          text: caseSel.value + " · " + metricSel.value + modeLabel,
          left: "center", textStyle: { fontSize: 13 },
        },
        tooltip: { trigger: "axis" },
        grid: { left: 60, right: 30, bottom: 80, top: 50 },
        xAxis: {
          type: "category", data: names,
          axisLabel: { rotate: 20, fontSize: 11, width: 160, overflow: "truncate" },
        },
        yAxis: { type: "value", scale: true },
        series: [{
          type: "bar", data: values, barMaxWidth: 60,
          itemStyle: { color: "#1a6fb0" },
          label: { show: true, position: "top", fontSize: 11 },
        }],
      }, true);
    }

    caseSel.onchange = function () { fillMetrics(); render(); };
    metricSel.onchange = render;
    fillMetrics();
    render();
    window.addEventListener("resize", function () {
      if (chartInstance) chartInstance.resize();
    });
  }
})();