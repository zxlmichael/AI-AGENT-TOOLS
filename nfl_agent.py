#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFL 赛事助手：在线发布每月新闻 + 实时跟进赛事周边新闻

- 数据层 nfl_data.py：ESPN 赛程比分 / 新闻流 / 月度归档（JSONL）
- 助手层：qwen-plus 汇总解读，无 key 时本地格式化兜底
- HTTP :8905：/（新闻+赛程+月报页面）、/api/news、/api/scores、
  /api/monthly、/api/report、/ask、/health

用法：
    .venv/bin/python nfl_agent.py "今天NFL有什么大新闻？"
    .venv/bin/python nfl_agent.py --chat
    .venv/bin/python nfl_agent.py --serve --port 8905
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import nfl_data as nd

HERE = Path(__file__).resolve().parent
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

AGENT_SYSTEM = """你是「NFL 赛事助手」，跟进北美职业橄榄球大联盟（NFL）的赛事与周边新闻。
你可以调用工具获取实时赛果、新闻流与月度归档。回答必须遵守：
1. 中文回答，先给结论；新闻按时间顺序列要点（标题+时间+一句话摘要）；
2. 时间统一为北京时间；数据以工具返回为准，不要编造；
3. 提及球队时可用英文队名（如 Kansas City Chiefs）或其中文常用译名；
4. 涉及"月度新闻"时，说明月份与条数，并给出代表性新闻；
5. 新闻来源为 ESPN，仅供参考。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "实时拉取 ESPN NFL 最新新闻流（标题/摘要/时间/链接）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回条数，默认 10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scores",
            "description": "最新 NFL 赛程与比分（含进行中/已结束状态）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回场次，默认 10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_news",
            "description": "读取某月归档新闻（ym=YYYYMM，空则最近月份），用于生成在线月报。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ym": {"type": "string", "description": "月份，如 202611；空取最近"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_rank",
            "description": "简易球队排名（按近 14 场比分统计胜率，非官方排名）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_status",
            "description": "查看数据源状态与归档情况。",
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


def _json(data, limit: int = 12000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def tool_get_news(limit: int = 10) -> str:
    items = nd.news(limit)
    return _json({"count": len(items), "items": items})


def tool_get_scores(limit: int = 10) -> str:
    return _json(nd.scores(limit))


def tool_get_monthly_news(ym: str = "") -> str:
    return _json(nd.monthly_news(ym), limit=9000)


def tool_get_team_rank() -> str:
    return _json({"rows": nd.leaders()[:16]})


def tool_data_status() -> str:
    m = nd.monthly_news("")
    return _json({"source": "ESPN NFL",
                  "available_months": m["available"],
                  "archived_news": len(m["items"]),
                  "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


TOOL_IMPL = {
    "get_news": tool_get_news,
    "get_scores": tool_get_scores,
    "get_monthly_news": tool_get_monthly_news,
    "get_team_rank": tool_get_team_rank,
    "data_status": tool_data_status,
}


# --------------------------------------------------------------------------- #
# Agent 循环
# --------------------------------------------------------------------------- #

def run_agent(question: str, history: list | None = None, use_llm: bool = True,
              max_steps: int = 4):
    """执行一次问答，返回 (answer, tools_used, error)。"""
    question = (question or "").strip()
    if not question:
        return "请输入要了解的问题。", [], ""
    if not use_llm or not llm_available():
        return format_news(question), [], "no-llm-fallback"

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
                         "content": "请基于已获取的数据直接给出最终回答，不要再调用工具。"})
        msg = _chat(messages, tools=False)
        text = (msg.content or "").strip()
        if text:
            return _clean(text), used, ""
        return "助手没有生成有效回答，请换个问法。", used, "空响应"
    except Exception as exc:
        return f"调用大模型失败：{exc}", used, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 本地格式化（无 LLM 兜底 / CLI 输出）
# --------------------------------------------------------------------------- #

def format_news(question: str) -> str:
    """本地兜底：直接输出最新新闻 + 赛果。"""
    lines = [f"【NFL 最新动态】{datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    try:
        s = nd.scores(5)
        lines.append("")
        lines.append("一、最新赛果/赛程")
        for g in s["games"]:
            lines.append(f"  {g['away']['name']} {g['away']['score']} @ "
                         f"{g['home']['name']} {g['home']['score']} [{g['detail']}]")
    except Exception:
        pass
    try:
        n = nd.news(8)
        lines.append("")
        lines.append("二、最新新闻")
        for x in n:
            lines.append(f"  [{x['published'][5:]}] {x['headline']}")
    except Exception:
        pass
    lines.append("")
    lines.append("（来源：ESPN，仅供参考）")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask("nfl_agent")
    STOCK_ASSETS = HERE / "stock_assets"

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(STOCK_ASSETS, filename)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "nfl-assistant",
                        "llm": "ready" if llm_available() else "missing-key",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.get("/api/news")
    def api_news():
        limit = int(request.args.get("limit", 10))
        try:
            return jsonify({"ok": True, "items": nd.news(limit)})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502

    @app.get("/api/scores")
    def api_scores():
        try:
            return jsonify({"ok": True, "data": nd.scores(14)})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502

    @app.get("/api/rank")
    def api_rank():
        try:
            return jsonify({"ok": True, "rows": nd.leaders()})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502

    @app.get("/api/monthly")
    def api_monthly():
        ym = str(request.args.get("ym", "") or "")
        try:
            return jsonify({"ok": True, "data": nd.monthly_news(ym)})
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502

    @app.get("/api/report")
    def api_report():
        """月度新闻 Markdown 报告（在线发布内容源）。"""
        ym = str(request.args.get("ym", "") or "")
        try:
            return Response(nd.build_monthly_report(ym), mimetype="text/markdown; charset=utf-8")
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
        return jsonify({"ok": bool(answer), "question": question, "answer": answer,
                        "tools": used, "error": err,
                        "elapsed_seconds": round(time.time() - started, 2)})

    return app


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL 赛事助手 · 月度新闻与实时跟进</title>
<style>
  :root { --bg:#0b1220; --panel:#131c31; --line:#26324a; --text:#e2e8f0; --muted:#94a3b8;
          --accent:#38bdf8; --warm:#f59e0b; --green:#4ade80; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:20px 16px 40px; }
  header { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  header h1 { font-size:20px; font-weight:700; }
  .badge { background:var(--accent); color:#06283d; font-size:11px; font-weight:700;
           border-radius:10px; padding:2px 10px; }
  .meta { color:var(--muted); font-size:12px; }
  .tabs { display:flex; gap:8px; margin-bottom:14px; }
  .tab { background:var(--panel); color:var(--muted); border:1px solid var(--line); border-radius:8px;
         padding:6px 18px; font-size:13px; cursor:pointer; }
  .tab.active { color:#fff; background:var(--accent); border-color:var(--accent); }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .panel h2 { font-size:15px; font-weight:600; margin-bottom:12px; }
  .news-item { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--line); }
  .news-item:last-child { border-bottom:none; }
  .news-item img { width:120px; height:76px; object-fit:cover; border-radius:8px; background:#0b1220; flex-shrink:0; }
  .news-body { flex:1; min-width:0; }
  .news-title { font-size:14px; font-weight:600; }
  .news-title a { color:var(--text); text-decoration:none; }
  .news-title a:hover { color:var(--accent); }
  .news-sum { color:var(--muted); font-size:12.5px; margin:3px 0; display:-webkit-box;
              -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .news-meta { color:var(--warm); font-size:11.5px; }
  .game { display:flex; justify-content:space-between; align-items:center; padding:8px 4px;
          border-bottom:1px solid var(--line); font-size:13.5px; }
  .game:last-child { border-bottom:none; }
  .game .teams { flex:1; }
  .game .score { font-weight:700; font-size:15px; }
  .game .st { color:var(--muted); font-size:11.5px; width:110px; text-align:right; }
  .st.final { color:var(--green); }
  .qa { display:flex; gap:10px; }
  .qa input { flex:1; background:#0b1222; border:1px solid var(--line); color:var(--text);
        border-radius:8px; padding:10px 14px; font-size:14px; }
  .qa button { background:var(--accent); color:#06283d; border:none; border-radius:8px;
        padding:0 22px; font-size:14px; font-weight:600; cursor:pointer; }
  .qaresult { background:#0b1222; border:1px solid var(--line); border-radius:8px;
        padding:12px 16px; font-size:13.5px; margin-top:10px; white-space:pre-wrap; display:none;
        max-height:340px; overflow:auto; }
  .btn { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:6px;
         padding:4px 12px; font-size:12px; cursor:pointer; }
  .btn:hover { border-color:var(--accent); }
  .foot { color:var(--muted); font-size:11px; text-align:center; margin-top:6px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { padding:5px 10px; text-align:left; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>NFL 赛事助手</h1>
    <span class="badge">NFL</span>
    <span class="meta">北美职业橄榄球大联盟 · 月度新闻在线发布 + 实时跟进</span>
    <span class="meta">更新：<span id="updated">--</span></span>
  </header>

  <div class="tabs">
    <button class="tab active" data-v="news">实时新闻</button>
    <button class="tab" data-v="scores">赛程比分</button>
    <button class="tab" data-v="monthly">月度新闻</button>
  </div>

  <div class="panel" id="panel-news">
    <h2>最新新闻 <button class="btn" onclick="loadNews()">刷新</button></h2>
    <div id="newsList"><div class="meta">加载中…</div></div>
  </div>

  <div class="panel" id="panel-scores" style="display:none">
    <h2>赛程与比分 <button class="btn" onclick="loadScores()">刷新</button></h2>
    <div id="scoreList"></div>
    <h2 style="margin-top:16px">球队排名（近14场胜率，非官方）</h2>
    <table><thead><tr><th>#</th><th>球队</th><th>胜-负</th><th>胜率</th></tr></thead>
      <tbody id="rankBody"></tbody></table>
  </div>

  <div class="panel" id="panel-monthly" style="display:none">
    <h2>月度新闻在线发布
      <select id="monthSel" style="background:#0b1222;color:var(--text);border:1px solid var(--line);
              border-radius:6px;padding:3px 8px;margin:0 8px;font-size:12px"></select>
      <a class="btn" id="reportLink" target="_blank" style="text-decoration:none">查看 Markdown 月报</a>
    </h2>
    <div id="monthlyList"></div>
  </div>

  <div class="panel">
    <h2>赛事问答（大模型）</h2>
    <div class="qa">
      <input id="qaInput" placeholder="如：今天NFL有什么大新闻？酋长队最近战绩如何？本月有什么重要赛事？" onkeydown="if(event.key==='Enter')ask()">
      <button onclick="ask()">提问</button>
    </div>
    <div class="qaresult" id="qaResult"></div>
  </div>

  <div class="foot">数据来源：ESPN · 仅供学习参考</div>
</div>

<script>
let curTab = "news";

function show(tab) {
  curTab = tab;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.v === tab));
  for (const v of ["news", "scores", "monthly"])
    document.getElementById("panel-" + v).style.display = v === tab ? "" : "none";
}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
  show(t.dataset.v);
  if (t.dataset.v === "scores") loadScores();
  if (t.dataset.v === "monthly") loadMonthly();
}));

async function loadNews() {
  try {
    const r = await (await fetch("/api/news?limit=12", { cache: "no-store" })).json();
    if (!r.ok) return;
    document.getElementById("updated").textContent =
      r.items.length ? r.items[0].published : "--";
    document.getElementById("newsList").innerHTML = r.items.map(x =>
      '<div class="news-item">' +
      (x.image ? '<img src="' + x.image + '" loading="lazy">' : "") +
      '<div class="news-body">' +
      '<div class="news-title"><a href="' + x.link + '" target="_blank">' + x.headline + '</a></div>' +
      '<div class="news-sum">' + (x.summary || "") + '</div>' +
      '<div class="news-meta">' + x.published + " · " + (x.category || "NFL") + '</div>' +
      "</div></div>").join("");
  } catch (e) { /* 静默 */ }
}

async function loadScores() {
  try {
    const r = await (await fetch("/api/scores", { cache: "no-store" })).json();
    if (!r.ok) return;
    document.getElementById("scoreList").innerHTML = r.data.games.map(g =>
      '<div class="game"><div class="teams">' + g.away.name + " <b>" + g.away.score + "</b> @ " +
      g.home.name + " <b>" + g.home.score + "</b></div>" +
      '<div class="st' + (g.state === "post" ? " final" : "") + '">' + g.detail + "</div></div>").join("");
    const rr = await (await fetch("/api/rank", { cache: "no-store" })).json();
    if (rr.ok) {
      document.getElementById("rankBody").innerHTML = rr.rows.slice(0, 10).map((x, i) =>
        "<tr><td>" + (i + 1) + "</td><td>" + x.name + "</td><td>" + x.wins + "-" + x.losses +
        "</td><td>" + x.pct.toFixed(3) + "</td></tr>").join("");
    }
  } catch (e) { /* 静默 */ }
}

async function loadMonthly() {
  try {
    const r = await (await fetch("/api/monthly", { cache: "no-store" })).json();
    if (!r.ok) return;
    const d = r.data;
    const sel = document.getElementById("monthSel");
    sel.innerHTML = (d.available || []).slice().reverse().map(m =>
      '<option value="' + m + '"' + (m === d.month ? " selected" : "") + ">" +
      m.slice(0, 4) + "-" + m.slice(4) + "</option>").join("");
    sel.onchange = () => { document.getElementById("reportLink").href = "/api/report?ym=" + sel.value; loadMonthly(); };
    document.getElementById("reportLink").href = "/api/report?ym=" + d.month;
    document.getElementById("monthlyList").innerHTML = d.items.length ?
      d.items.map(x => '<div class="news-item"><div class="news-body">' +
        '<div class="news-title"><a href="' + x.link + '" target="_blank">' + x.headline + '</a></div>' +
        '<div class="news-sum">' + (x.summary || "") + '</div>' +
        '<div class="news-meta">' + x.published + " · " + (x.category || "NFL") + "</div>" +
        "</div></div>").join("")
      : '<div class="meta">该月暂无归档，可先到「实时新闻」刷新归档。</div>';
  } catch (e) { /* 静默 */ }
}

async function ask() {
  const q = document.getElementById("qaInput").value.trim();
  const box = document.getElementById("qaResult");
  if (!q) return;
  box.style.display = "block"; box.textContent = "思考中…";
  try {
    const r = await (await fetch("/ask", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }) })).json();
    box.textContent = r.answer || ("回答失败：" + (r.error || "未知错误"));
  } catch (e) { box.textContent = "请求失败：" + e.message; }
}

loadNews();
</script>
</body>
</html>"""


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
    print("NFL 赛事助手已就绪，输入问题开始；输入 exit / quit 退出。\n")
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
    ap = argparse.ArgumentParser(description="NFL 赛事助手（月报+实时新闻）")
    ap.add_argument("question", nargs="?", default="", help="问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8905)
    ap.add_argument("--trace", action="store_true", help="显示工具调用过程")
    ap.add_argument("--no-llm", action="store_true", help="不调用大模型，本地格式化输出")
    ap.add_argument("--update-archive", action="store_true",
                    help="拉取最新新闻并归档后退出（供定时任务用）")
    args = ap.parse_args()

    if args.update_archive:
        items = nd.news(20)
        print("归档结果:", nd.archive_news(items))
        return

    if args.serve:
        # 启动时归档一次当前新闻（供月度页使用）
        try:
            res = nd.archive_news(nd.news(10))
            print(f"新闻已同步归档：+{res['added']}（共 {res['total']} 条）")
        except Exception as exc:
            print(f"新闻同步失败：{exc}")
        app = create_app()
        print(f"NFL 赛事助手 HTTP 服务：http://{args.host}:{args.port}")
        print("  页面: /   新闻: /api/news   赛程: /api/scores")
        print("  月报: /api/report   问答: POST /ask   健康: /health")
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if args.chat or not args.question:
        _chat_loop(args.trace, use_llm=not args.no_llm)
        return

    if args.no_llm:
        print(format_news(args.question))
        return
    answer, used, err = run_agent(args.question, use_llm=True)
    _print_result(answer, used, err, args.trace)


if __name__ == "__main__":
    main()