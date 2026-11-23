#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股票助手：监测亚洲主要股市（A股/港股）与美股中表现优异的股票，聚焦科技股

实时数据来自 stock_data.py（新浪行情/榜单 + 雅虎美股榜，本网络实测可用），
大模型（qwen-plus）负责汇总解读；未配置 key 时自动退回本地格式化输出。

用法：
    .venv/bin/python stock_agent.py "今天A股科技股表现怎么样？"
    .venv/bin/python stock_agent.py --chat
    .venv/bin/python stock_agent.py --serve --port 8902
    .venv/bin/python stock_agent.py --no-llm --overview     # 只看本地格式化行情

HTTP：
    curl -X POST http://127.0.0.1:8902/ask -H "Content-Type: application/json" \
         -d '{"question":"港股今天涨得最多的科技股？"}'
    curl http://127.0.0.1:8902/overview                     # 原始行情 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import stock_data

HERE = Path(__file__).resolve().parent
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

AGENT_SYSTEM = """你是「股票助手」，监测中国A股、中国香港股市与美股中表现优异的股票，重点关注科技股。
你可以调用工具获取实时行情。回答必须遵守：
1. 中文回答，先给结论再列数据，条理清晰，每只股票一行；
2. 说明数据时间（各市场收盘/交易时段不同，美股为最近收盘价）；
3. 价格单位：A股人民币元、港股港元、美股美元；涨跌幅为百分比；
4. 涨幅榜里可能含小市值股或非科技股，回答时如涉及科技股请说明；
5. 仅提供行情事实与数据解读，不做买卖建议；遇到“该不该买”类问题，提示数据仅供参考、投资需谨慎；
6. 不要编造工具未返回的数据；工具失败时如实说明。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": "全市场概览：主要指数（上证/深证/创业板/恒生/道指/纳指/标普）"
                           "与各市场当日涨幅榜前5、科技股观察清单行情。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_gainers",
            "description": "指定市场当日涨幅榜。market 取值：a（A股）/ hk（港股）/ us（美股）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {"type": "string", "enum": ["a", "hk", "us"]},
                    "n": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tech_watchlist",
            "description": "科技股观察清单行情（A股15只/港股12只/美股15只主流科技龙头），"
                           "按当日涨跌幅降序。market 取值：a / hk / us。",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {"type": "string", "enum": ["a", "hk", "us"]},
                },
                "required": ["market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_status",
            "description": "查看数据源连接状态与最近更新时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

def _load_key() -> str:
    key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text("utf-8").splitlines():
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def llm_available() -> bool:
    return bool(_load_key())


def _clean(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("md\n").strip()
    return text


def _chat(messages: list, tools: bool = True):
    from openai import OpenAI
    client = OpenAI(api_key=_load_key(), base_url=LLM_BASE, timeout=90.0)
    kwargs = dict(model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=1000)
    if tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs).choices[0].message


def _assistant_msg(msg) -> dict:
    data = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        data["tool_calls"] = [{
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in msg.tool_calls]
    return data


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def _json(data, limit: int = 8000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


def tool_market_overview() -> str:
    return _json(stock_data.market_overview())


def tool_top_gainers(market: str, n: int = 5) -> str:
    return _json({"market": market, "rows": stock_data.top_gainers(market, n)})


def tool_tech_watchlist(market: str) -> str:
    return _json({"market": market, "rows": stock_data.tech_watchlist(market)})


def tool_data_status() -> str:
    return _json({"fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "sources": ["新浪行情 hq.sinajs.cn", "新浪榜单 getHQNodeData/getHKStockData",
                              "雅虎 day_gainers"]})


TOOL_IMPL = {
    "get_market_overview": tool_market_overview,
    "get_top_gainers": tool_top_gainers,
    "get_tech_watchlist": tool_tech_watchlist,
    "data_status": tool_data_status,
}


# --------------------------------------------------------------------------- #
# 本地格式化（无 LLM 兜底）
# --------------------------------------------------------------------------- #

def _fmt_pct(v) -> str:
    if v is None:
        return "--"
    return f"{v:+.2f}%"


def _fmt_price(v, digits: int = 3) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def format_overview(ov: dict) -> str:
    """把 market_overview() 的字典渲染成可读文本，供无 LLM 时直接输出。"""
    lines = [f"【行情快照】{ov.get('fetched_at', '')}"]
    lines.append("")
    lines.append("一、主要指数")
    for i in ov.get("indices", []):
        lines.append(f"  {i['name']}: {i['price']}  {_fmt_pct(i['change_pct'])}")
    names = {"a": "A股", "hk": "港股", "us": "美股"}
    for m in ("a", "hk", "us"):
        lines.append("")
        lines.append(f"二、{names[m]}涨幅榜前{len(ov['top'][m])}")
        for s in ov["top"][m]:
            lines.append(f"  {s['name']} ({s['code']}): {_fmt_price(s['price'])}  "
                         f"{_fmt_pct(s['change_pct'])}")
    lines.append("")
    lines.append("三、科技股观察清单")
    for m in ("a", "hk", "us"):
        lines.append(f"  【{names[m]}】")
        for s in ov["watchlist"][m]:
            lines.append(f"    {s['name']} ({s['code']}): {_fmt_price(s['price'])}  "
                         f"{_fmt_pct(s['change_pct'])}")
    lines.append("")
    lines.append("（数据来自新浪财经/雅虎财经，仅供参考，不构成投资建议）")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent 循环
# --------------------------------------------------------------------------- #

def run_agent(question: str, history: list | None = None, use_llm: bool = True,
              max_steps: int = 4):
    """执行一次问答，返回 (answer, tools_used, error)。"""
    question = (question or "").strip()
    if not question:
        return "请输入问题。", [], ""
    if not use_llm or not llm_available():
        # 无 key 或显式关闭：本地渲染全市场概览兜底
        try:
            return format_overview(stock_data.market_overview()), [], "no-llm-fallback"
        except Exception as exc:
            return f"获取行情失败：{exc}", [], f"{type(exc).__name__}: {exc}"

    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    if history:
        messages.append({"role": "system",
                         "content": "以下是最近的对话，供理解指代：\n" + "\n".join(history[-6:])})
    messages.append({"role": "user", "content": question})

    used = []
    try:
        for _ in range(max_steps):
            msg = _chat(messages)
            calls = getattr(msg, "tool_calls", None) or []
            messages.append(_assistant_msg(msg))
            if not calls:
                text = (msg.content or "").strip()
                if text:
                    return _clean(text), used, ""
                break
            for tc in calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = TOOL_IMPL.get(name)
                if not fn:
                    result = _json({"error": f"未知工具 {name}"})
                else:
                    try:
                        result = fn(**args)
                        used.append({"tool": name, "args": args, "ok": True})
                    except Exception as exc:
                        result = _json({"error": f"{type(exc).__name__}: {exc}"})
                        used.append({"tool": name, "args": args, "ok": False})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        messages.append({"role": "user",
                         "content": "请基于已获取的行情数据直接给出最终回答，不要再调用工具。"})
        msg = _chat(messages, tools=False)
        text = (msg.content or "").strip()
        if text:
            return _clean(text), used, ""
        return "助手没有生成有效回答，请换个问法。", used, "空响应"
    except Exception as exc:
        return f"调用大模型失败：{exc}", used, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 实时监测页（ECharts）
# --------------------------------------------------------------------------- #

STOCK_ASSETS = HERE / "stock_assets"

STOCK_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>股票实时监测 · A股/港股/美股科技股</title>
<script src="/assets/echarts.min.js"></script>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8;
          --up:#e0403f; --down:#22a06b; --accent:#3b82f6; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:20px 16px 40px; }
  header { display:flex; align-items:center; gap: 14px; flex-wrap:wrap; margin-bottom:18px; }
  header h1 { font-size:20px; font-weight:700; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
  .dot.on { background:#22c55e; } .dot.off { background:#ef4444; }
  .meta { color:var(--muted); font-size:12px; }
  .btn { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:6px;
         padding:4px 12px; font-size:12px; cursor:pointer; }
  .btn:hover { border-color:var(--accent); }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:10px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
  .card .nm { font-size:12px; color:var(--muted); }
  .card .px { font-size:19px; font-weight:700; margin:2px 0; }
  .card .pc { font-size:13px; font-weight:600; }
  .up { color:var(--up); } .down { color:var(--down); }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .panel h2 { font-size:15px; font-weight:600; margin-bottom:10px; }
  .chart { width:100%; height:320px; }
  .tabs { display:flex; gap:8px; margin-bottom:10px; }
  .tab { background:transparent; color:var(--muted); border:1px solid var(--line); border-radius:6px;
         padding:3px 14px; font-size:12px; cursor:pointer; }
  .tab.active { color:#fff; background:var(--accent); border-color:var(--accent); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { padding:6px 10px; text-align:right; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; font-size:12px; }
  th:first-child, td:first-child { text-align:left; }
  td.nm { text-align:left; } td.rank { color:var(--muted); width:40px; }
  .foot { color:var(--muted); font-size:11px; text-align:center; margin-top:6px; }
  .loading { color:var(--muted); padding:40px; text-align:center; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>股票实时监测</h1>
    <span class="meta"><span class="dot" id="dot"></span><span id="status">连接中…</span></span>
    <span class="meta">最近更新：<span id="updated">--</span></span>
    <span class="meta">自动刷新：15 秒</span>
    <button class="btn" onclick="load(true)">立即刷新</button>
  </header>

  <div class="cards" id="idxCards"><div class="loading">加载中…</div></div>

  <div class="panel">
    <h2>主要指数走势（本次打开页面后的采样）</h2>
    <div class="chart" id="trend"></div>
  </div>

  <div class="panel">
    <h2>科技股观察清单 · 涨跌幅</h2>
    <div class="tabs" id="watchTabs">
      <button class="tab active" data-m="a">A股</button>
      <button class="tab" data-m="hk">港股</button>
      <button class="tab" data-m="us">美股</button>
    </div>
    <div class="chart" id="watch"></div>
  </div>

  <div class="panel">
    <h2>各市场当日涨幅榜</h2>
    <div class="tabs" id="topTabs">
      <button class="tab active" data-m="a">A股</button>
      <button class="tab" data-m="hk">港股</button>
      <button class="tab" data-m="us">美股</button>
    </div>
    <table>
      <thead><tr><th>#</th><th>名称</th><th>代码</th><th>现价</th><th>涨跌幅</th></tr></thead>
      <tbody id="topBody"></tbody>
    </table>
  </div>

  <div class="foot">数据来源：新浪财经 / 雅虎财经 · 仅供学习参考，不构成投资建议</div>
</div>

<script>
const REFRESH_MS = 15000, TREND_MAX = 120;
const MARKET = { a: "A股", hk: "港股", us: "美股" };
let overview = null, watchM = "a", topM = "a";
const trend = { labels: [], series: {} };
const trendChart = echarts.init(document.getElementById("trend"));
const watchChart = echarts.init(document.getElementById("watch"));

const fmt = v => (v == null ? "--" : Number(v).toFixed(2));
const pct = v => (v == null ? "--" : (v > 0 ? "+" : "") + Number(v).toFixed(2) + "%");
const cls = v => (v == null ? "" : v >= 0 ? "up" : "down");

async function load(manual) {
  try {
    const resp = await fetch("/overview", { cache: "no-store" });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    overview = (await resp.json()).data;
    setStatus(true);
    document.getElementById("updated").textContent = overview.fetched_at;
    renderCards(); renderTrend(); renderWatch(); renderTop();
  } catch (e) {
    setStatus(false);
    if (manual) alert("刷新失败：" + e.message);
  }
}

function setStatus(ok) {
  document.getElementById("dot").className = "dot " + (ok ? "on" : "off");
  document.getElementById("status").textContent = ok ? "在线" : "离线（等待重连…）";
}

function renderCards() {
  document.getElementById("idxCards").innerHTML = overview.indices.map(i =>
    '<div class="card"><div class="nm">' + i.name + '</div>' +
    '<div class="px ' + cls(i.change_pct) + '">' + fmt(i.price) + '</div>' +
    '<div class="pc ' + cls(i.change_pct) + '">' + pct(i.change_pct) + '</div></div>'
  ).join("");
}

function renderTrend() {
  const ts = (overview.fetched_at || "").slice(11, 19) || new Date().toLocaleTimeString();
  trend.labels.push(ts);
  for (const i of overview.indices) {
    (trend.series[i.name] = trend.series[i.name] || []).push(i.price);
    if (trend.series[i.name].length > TREND_MAX) trend.series[i.name].shift();
  }
  if (trend.labels.length > TREND_MAX) trend.labels.shift();
  trendChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: Object.keys(trend.series), textStyle: { color: "#94a3b8" } },
    grid: { left: 60, right: 16, top: 34, bottom: 26 },
    xAxis: { type: "category", data: trend.labels, boundaryGap: false,
             axisLine: { lineStyle: { color: "#334155" } }, axisLabel: { color: "#94a3b8" } },
    yAxis: { type: "value", scale: true, splitLine: { lineStyle: { color: "#26324a" } },
             axisLabel: { color: "#94a3b8" } },
    series: Object.entries(trend.series).map(([name, data]) => ({
      name, type: "line", data, showSymbol: false, smooth: true, lineStyle: { width: 2 },
    })),
  });
}

function renderWatch() {
  const rows = (overview.watchlist[watchM] || []).slice().reverse();
  watchChart.setOption({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
               formatter: p => p[0].name + "：" + (p[0].value == null ? "--" : p[0].value + "%") },
    grid: { left: 96, right: 44, top: 8, bottom: 26 },
    xAxis: { type: "value", name: "涨跌幅 %", nameTextStyle: { color: "#94a3b8" },
             axisLabel: { color: "#94a3b8" }, splitLine: { lineStyle: { color: "#26324a" } } },
    yAxis: { type: "category", data: rows.map(r => r.name),
             axisLabel: { color: "#e2e8f0" }, axisLine: { lineStyle: { color: "#334155" } } },
    series: [{ type: "bar", barMaxWidth: 14,
               data: rows.map(r => ({ value: r.change_pct,
                 itemStyle: { color: (r.change_pct ?? 0) >= 0 ? "#e0403f" : "#22a06b",
                              borderRadius: r.change_pct >= 0 ? [0,3,3,0] : [3,0,0,3] } })) }],
  }, true);
}

function renderTop() {
  const rows = (overview.top[topM] || []);
  document.getElementById("topBody").innerHTML = rows.map((s, i) =>
    '<tr><td class="rank">' + (i + 1) + '</td>' +
    '<td class="nm">' + (s.name || "") + '</td>' +
    '<td>' + (s.code || "") + '</td>' +
    '<td>' + fmt(s.price) + '</td>' +
    '<td class="' + cls(s.change_pct) + '">' + pct(s.change_pct) + '</td></tr>'
  ).join("");
}

for (const [tid, fn] of [["watchTabs", m => { watchM = m; renderWatch(); }],
                          ["topTabs", m => { topM = m; renderTop(); }]]) {
  document.getElementById(tid).addEventListener("click", ev => {
    const btn = ev.target.closest(".tab"); if (!btn) return;
    document.querySelectorAll("#" + tid + " .tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active"); fn(btn.dataset.m);
  });
}
window.addEventListener("resize", () => { trendChart.resize(); watchChart.resize(); });
setInterval(load, REFRESH_MS);
load(true);
</script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask("stock_assistant")

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.get("/")
    def index():
        """ECharts 实时监测页（每 15 秒轮询 /overview 渲染）。"""
        return Response(STOCK_PAGE, mimetype="text/html")

    @app.get("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(STOCK_ASSETS, filename)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "stock-assistant",
                        "llm": "ready" if llm_available() else "missing-key",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.get("/overview")
    def overview():
        """原始行情 JSON，供页面/脚本直接消费。"""
        try:
            return jsonify({"ok": True, "data": stock_data.market_overview()})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502

    @app.route("/ask", methods=["GET", "POST", "OPTIONS"])
    def ask():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            question = body.get("question") or body.get("q") or ""
        else:
            question = request.args.get("q") or request.args.get("question") or ""
        question = str(question).strip()
        if not question:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        started = time.time()
        answer, used, err = run_agent(question)
        return jsonify({
            "ok": bool(answer),
            "question": question,
            "answer": answer,
            "tools": used,
            "error": err,
            "elapsed_seconds": round(time.time() - started, 2),
        })

    return app


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def _print_result(answer: str, used: list, err: str, trace: bool = False):
    if trace and used:
        print("[工具调用]")
        for item in used:
            flag = "ok" if item["ok"] else "fail"
            print(f"  - {item['tool']}({json.dumps(item['args'], ensure_ascii=False)}) {flag}")
        print()
    print(answer)
    if err and not answer:
        print(f"\n（助手不可用：{err}）", file=sys.stderr)


def _chat_loop(trace: bool, use_llm: bool = True):
    print("股票助手已就绪（A股/港股/美股科技股监测），输入问题开始对话；输入 exit / quit 退出。\n")
    history = []
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        started = time.time()
        answer, used, err = run_agent(q, history=history, use_llm=use_llm)
        _print_result(answer, used, err, trace)
        print(f"\n({time.time() - started:.1f}s)\n")
        history.append(f"用户：{q}")
        history.append(f"助手：{answer}")


def main():
    ap = argparse.ArgumentParser(description="股票助手（亚/美市场科技股监测）")
    ap.add_argument("question", nargs="?", default="", help="要问的问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--trace", action="store_true", help="显示工具调用过程")
    ap.add_argument("--no-llm", action="store_true",
                    help="不调用大模型，直接输出本地格式化行情")
    ap.add_argument("--overview", action="store_true",
                    help="只输出全市场概览（等价于 --no-llm 且忽略问题）")
    args = ap.parse_args()

    if args.overview:
        try:
            print(format_overview(stock_data.market_overview()))
        except Exception as exc:
            print(f"获取行情失败：{exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.serve:
        app = create_app()
        print(f"股票助手 HTTP 服务：http://{args.host}:{args.port}")
        print(f"  监测页: http://{args.host}:{args.port}/")
        print(f"  POST /ask  {{\"question\": \"...\"}}   GET /ask?q=...   GET /overview   GET /health")
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if args.chat or not args.question:
        _chat_loop(args.trace, use_llm=not args.no_llm)
        return

    answer, used, err = run_agent(args.question, use_llm=not args.no_llm)
    _print_result(answer, used, err, args.trace)


if __name__ == "__main__":
    main()