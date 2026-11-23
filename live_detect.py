#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""衣物直播商品识别助手：识别直播/电商商品是否为「衣物类且定价 ≥ 1000 元」

- 数据层 live_detect_data.py：纯规则判定（类目词表 + 价格提取），记录入 JSONL
- 助手层：qwen-plus 辅助复核（规则不明确时），本地格式化兜底（--no-llm）
- HTTP :8903：/（看板）、/detect、/records、/stats、/ask、/health

用法：
    .venv/bin/python live_detect.py "羊绒大衣 券后2680元"
    .venv/bin/python live_detect.py --chat
    .venv/bin/python live_detect.py --serve --port 8903

设计文档：.comate/specs/live-detect/doc.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import live_detect_data as ldd

HERE = Path(__file__).resolve().parent
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

AGENT_SYSTEM = """你是「衣物直播商品识别助手」，识别直播/电商商品是否为衣物类（服装/鞋靴/箱包）且定价 ≥ 1000 元。
你可以调用工具识别商品并查询识别记录。回答必须遵守：
1. 中文回答，先给结论（是否衣物、是否 ≥1000 元），再给判定依据；
2. 价格以工具返回为准，注明价格来源（如"券后价/到手价"）；
3. 只识别衣物类商品（服装、鞋靴、箱包）；其他品类明确说"非衣物类"；
4. 判定依据写清命中的类目词（如"命中「羊绒大衣」→ 服装"）；
5. 识别结果为规则 + 模型复核，价格仍以商品页实际为准，不构成购买建议；
6. 不要编造工具未返回的信息；工具失败时如实说明。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_product",
            "description": "识别商品文本（名称/价格/JSON），判断是否衣物类且 ≥1000 元，结果入库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "商品文本，如：羊绒大衣 券后2680元"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_records",
            "description": "查询识别记录。kind 取值：all（全部）/ cloth（仅衣物）/ premium（仅达标≥1000元）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["all", "cloth", "premium"]},
                    "n": {"type": "integer", "description": "返回条数，默认 30"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stats",
            "description": "识别统计：累计数/衣物数/达标数/类目分布/价格区间分布。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_status",
            "description": "查看数据源与存储状态（记录数、最近写入时间）。",
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
    kwargs = dict(model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=900)
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


def _json(data, limit: int = 8000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def _verify_text(text: str) -> str:
    text = (text or "").strip()
    return text[:600] if len(text) > 600 else text


def tool_detect_product(text: str) -> str:
    return _json(ldd.detect(_verify_text(text)))


def tool_list_records(kind: str = "all", n: int = 30) -> str:
    return _json({"kind": kind, "rows": ldd.list_records(kind, n)})


def tool_stats() -> str:
    s = ldd.stats()
    s["updated"] = ldd.updated_at()
    return _json(s)


def tool_data_status() -> str:
    return _json({"records_total": ldd.stats()["total"],
                  "updated": ldd.updated_at(),
                  "max_records": ldd.MAX_RECORDS,
                  "threshold": ldd.PRICE_THRESHOLD})


TOOL_IMPL = {
    "detect_product": tool_detect_product,
    "list_records": tool_list_records,
    "stats": tool_stats,
    "data_status": tool_data_status,
}


# --------------------------------------------------------------------------- #
# LLM 复核（仅识别不清时增强回复；入库记录保持 rule 结果）
# --------------------------------------------------------------------------- #

def llm_recheck(name: str, desc: str, price) -> str:
    """对规则未命中的商品做一次 LLM 补充判断，返回一个附加说明串；失败返回空串。"""
    if not llm_available():
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_load_key(), base_url=LLM_BASE, timeout=60.0)
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0.1, max_tokens=200,
            messages=[{"role": "system",
                       "content": "你是商品分类助手。判断该商品是否属于衣物类"
                                  "（服装/鞋靴/箱包），只回『衣物类』或『非衣物类』"
                                  "，再补一句不超过30字依据。"},
                      {"role": "user", "content": f"商品：{name}\n描述：{desc[:200]}"}])
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Agent 循环
# --------------------------------------------------------------------------- #

def run_agent(question: str, history: list | None = None, use_llm: bool = True,
              max_steps: int = 4):
    """执行一次问答，返回 (answer, tools_used, error)。"""
    question = (question or "").strip()
    if not question:
        return "请输入要识别的商品信息。", [], ""
    if not use_llm or not llm_available():
        try:
            return format_detect(question), [], "no-llm-fallback"
        except Exception as exc:
            return f"识别失败：{exc}", [], f"{type(exc).__name__}: {exc}"

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
                         "content": "请基于已获取的识别结果直接给出最终回答，不要再调用工具。"})
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

def format_detect(text: str) -> str:
    r = ldd.detect(text)
    res = r["results"][0]
    kind = "衣物类" if res["is_cloth"] else "非衣物类"
    verdict = ("✅ 达标（衣物类且 ≥1000 元）" if res["premium"]
               else "⚠️ 未达标" if res["is_cloth"] else "❌ 非衣物类")
    price = f"{res['price']:g} 元" if res["price"] is not None else "未提取到价格"
    extra = ""
    if not res["is_cloth"] and res["price"] is not None:
        extra = llm_recheck(res["name"], text, res["price"])
        if extra:
            extra = f"\n[模型复核] {extra}"
    lines = [f"识别结果：{verdict}",
             f"  商品：{res['name']}",
             f"  类目：{res['category'] or '未知'}",
             f"  价格：{price}",
             f"  依据：{res['reason']}",
             f"  方式：{'规则' if res['method'] == 'rule' else '模型'}"]
    if res["premium"]:
        lines.append("  （价格以商品页实际为准，仅供参考）")
    return "\n".join(lines) + extra


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask("live_detect_assistant")
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
        s = ldd.stats()
        return jsonify({"ok": True, "service": "live-detect-assistant",
                        "llm": "ready" if llm_available() else "missing-key",
                        "records": s["total"],
                        "updated": ldd.updated_at(),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.route("/detect", methods=["POST", "OPTIONS"])
    def detect():
        if request.method == "OPTIONS":
            return ("", 204)
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or body.get("q") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "缺少 text 参数"}), 400
        try:
            return jsonify(ldd.detect(text))
        except Exception as exc:
            return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502

    @app.get("/records")
    def records():
        kind = request.args.get("kind", "all")
        n = int(request.args.get("n", 30))
        return jsonify({"ok": True, "kind": kind, "rows": ldd.list_records(kind, n)})

    @app.get("/stats")
    def stats():
        s = ldd.stats()
        s["updated"] = ldd.updated_at()
        return jsonify({"ok": True, "data": s})

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
<title>衣物直播商品识别监测</title>
<script src="/assets/echarts.min.js"></script>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8;
          --up:#e0403f; --down:#22a06b; --accent:#3b82f6; --gold:#f59e0b; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:20px 16px 40px; }
  header { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  header h1 { font-size:20px; font-weight:700; }
  .meta { color:var(--muted); font-size:12px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .card .nm { font-size:12px; color:var(--muted); }
  .card .num { font-size:24px; font-weight:700; margin-top:2px; }
  .card .num.gold { color:var(--gold); } .card .num.blue { color:var(--accent); }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .panel h2 { font-size:15px; font-weight:600; margin-bottom:10px; }
  .chart { width:100%; height:300px; }
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:900px){ .charts { grid-template-columns:1fr; } }
  .tabs { display:flex; gap:8px; margin-bottom:10px; }
  .tab { background:transparent; color:var(--muted); border:1px solid var(--line); border-radius:6px;
         padding:3px 14px; font-size:12px; cursor:pointer; }
  .tab.active { color:#fff; background:var(--accent); border-color:var(--accent); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { padding:6px 10px; text-align:right; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; font-size:12px; }
  th:first-child, td:first-child { text-align:left; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; }
  .b-ok { background:rgba(240,89,63,.15); color:#f0593f; }
  .b-hit { background:rgba(59,130,246,.15); color:#60a5fa; }
  .b-off { background:rgba(148,163,184,.15); color:#94a3b8; }
  .foot { color:var(--muted); font-size:11px; text-align:center; margin-top:6px; }
  .detect-box { display:flex; gap:10px; margin-bottom:8px; }
  .detect-box input { flex:1; background:#0b1222; border:1px solid var(--line); color:var(--text);
         border-radius:8px; padding:9px 14px; font-size:14px; }
  .detect-box button { background:var(--accent); color:#fff; border:none; border-radius:8px;
         padding:0 22px; font-size:14px; cursor:pointer; }
  .detect-result { background:#0b1222; border:1px solid var(--line); border-radius:8px;
         padding:10px 14px; font-size:13px; white-space:pre-wrap; margin-top:8px; display:none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>衣物直播商品识别监测</h1>
    <span class="meta">识别标准：衣物类（服装/鞋靴/箱包）且定价 ≥ 1000 元</span>
    <span class="meta">最近更新：<span id="updated">--</span></span>
  </header>

  <div class="cards" id="statCards"></div>

  <div class="panel">
    <h2>快速识别</h2>
    <div class="detect-box">
      <input id="detectInput" placeholder="粘贴商品名称/链接/JSON，如：羊绒大衣 券后2680元" onkeydown="if(event.key==='Enter')doDetect()">
      <button onclick="doDetect()">识别</button>
    </div>
    <div class="detect-result" id="detectResult"></div>
  </div>

  <div class="charts">
    <div class="panel"><h2>达标商品价格分布（元）</h2><div class="chart" id="priceChart"></div></div>
    <div class="panel"><h2>识别类目占比</h2><div class="chart" id="catChart"></div></div>
  </div>

  <div class="panel">
    <h2>识别记录</h2>
    <div class="tabs" id="recTabs">
      <button class="tab active" data-k="all">全部</button>
      <button class="tab" data-k="cloth">仅衣物</button>
      <button class="tab" data-k="premium">仅达标(≥1000元)</button>
    </div>
    <table>
      <thead><tr><th>时间</th><th>商品名称</th><th>类目</th><th>价格</th><th>是否衣物</th><th>是否达标</th><th>方式</th></tr></thead>
      <tbody id="recBody"></tbody>
    </table>
  </div>

  <div class="foot">识别结果由规则 + 模型复核生成，价格以商品页实际为准 · 仅供学习参考</div>
</div>

<script>
const REFRESH_MS = 15000;
const priceChart = echarts.init(document.getElementById("priceChart"));
const catChart = echarts.init(document.getElementById("catChart"));
let recKind = "all";
const fmt = v => (v == null ? "--" : Number(v).toFixed(2));

async function loadStats() {
  try {
    const r = await (await fetch("/stats", { cache: "no-store" })).json();
    if (!r.ok) return;
    const d = r.data;
    document.getElementById("updated").textContent = d.updated || "--";
    document.getElementById("statCards").innerHTML =
      '<div class="card"><div class="nm">累计识别</div><div class="num blue">' + d.total + '</div></div>' +
      '<div class="card"><div class="nm">衣物类</div><div class="num blue">' + d.cloth + '</div></div>' +
      '<div class="card"><div class="nm">达标(≥1000元)</div><div class="num gold">' + d.premium + '</div></div>' +
      '<div class="card"><div class="nm">类目数</div><div class="num">' + Object.keys(d.by_category).length + '</div></div>';
    const bins = d.price_bins || {};
    priceChart.setOption({
      tooltip: { trigger: "axis" },
      grid: { left: 50, right: 16, top: 20, bottom: 26 },
      xAxis: { type: "category", data: Object.keys(bins), axisLabel: { color: "#94a3b8", rotate: 20 } },
      yAxis: { type: "value", axisLabel: { color: "#94a3b8" }, splitLine: { lineStyle: { color: "#26324a" } } },
      series: [{ type: "bar", barMaxWidth: 40, color: "#f59e0b",
                 data: Object.values(bins) }],
    }, true);
    const cats = d.by_category || {};
    catChart.setOption({
      tooltip: { trigger: "item" },
      legend: { bottom: 0, textStyle: { color: "#94a3b8" } },
      series: [{ type: "pie", radius: ["38%", "66%"], center: ["50%", "44%"],
                 data: Object.entries(cats).map(([name, v]) => ({ name, value: v })),
                 label: { color: "#e2e8f0" },
                 itemStyle: { borderColor: "#1e293b", borderWidth: 2 } }],
    }, true);
  } catch (e) { /* 静默重试 */ }
}

async function loadRecords() {
  try {
    const r = await (await fetch("/records?kind=" + recKind + "&n=30", { cache: "no-store" })).json();
    if (!r.ok) return;
    document.getElementById("recBody").innerHTML = r.rows.map(x =>
      "<tr><td>" + (x.ts || "").slice(11, 19) + "</td>" +
      "<td>" + (x.name || "--") + "</td>" +
      "<td>" + (x.category || "--") + "</td>" +
      "<td>" + fmt(x.price) + "</td>" +
      '<td>' + (x.is_cloth ? '<span class="badge b-hit">衣物</span>' : '<span class="badge b-off">否</span>') + "</td>" +
      '<td>' + (x.premium ? '<span class="badge b-ok">达标</span>' : '<span class="badge b-off">--</span>') + "</td>" +
      "<td>" + (x.method === "rule" ? "规则" : "模型") + "</td></tr>").join("");
  } catch (e) { /* 静默重试 */ }
}

async function doDetect() {
  const text = document.getElementById("detectInput").value.trim();
  const box = document.getElementById("detectResult");
  if (!text) { box.style.display = "block"; box.textContent = "请先输入商品信息"; return; }
  box.style.display = "block"; box.textContent = "识别中…";
  try {
    const r = await (await fetch("/detect", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }) })).json();
    if (!r.ok) { box.textContent = "识别失败：" + (r.error || "未知错误"); return; }
    const res = r.results[0];
    box.textContent =
      (res.premium ? "✅ 达标（衣物类且 ≥1000 元）" : res.is_cloth ? "⚠️ 衣物类但未达标（<1000 元）" : "❌ 非衣物类") +
      "\\n商品：" + (res.name || "--") + "\\n类目：" + (res.category || "未知") +
      "\\n价格：" + (res.price == null ? "未提取到" : res.price + " 元") +
      "\\n依据：" + res.reason + (r.duplicate ? "\\n（该记录已存在，未重复入库）" : "");
    loadStats(); loadRecords();
  } catch (e) {
    box.textContent = "识别失败：" + e.message;
  }
}

document.getElementById("recTabs").addEventListener("click", ev => {
  const btn = ev.target.closest(".tab"); if (!btn) return;
  document.querySelectorAll("#recTabs .tab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active"); recKind = btn.dataset.k; loadRecords();
});
window.addEventListener("resize", () => { priceChart.resize(); catChart.resize(); });
setInterval(() => { loadStats(); loadRecords(); }, REFRESH_MS);
loadStats(); loadRecords();
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
    print("衣物直播商品识别助手已就绪，输入商品信息开始识别；输入 exit / quit 退出。\n")
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
    ap = argparse.ArgumentParser(description="衣物直播商品识别助手（≥1000元自动识别）")
    ap.add_argument("question", nargs="?", default="", help="商品文本；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8903)
    ap.add_argument("--trace", action="store_true", help="显示工具调用过程")
    ap.add_argument("--no-llm", action="store_true", help="不调用大模型，本地规则输出")
    args = ap.parse_args()

    if args.serve:
        app = create_app()
        print(f"衣物识别助手 HTTP 服务：http://{args.host}:{args.port}")
        print("  看板: /   识别: POST /detect   记录: /records   统计: /stats")
        print('  问答: POST /ask {"question":"..."}   /health')
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if args.chat or not args.question:
        _chat_loop(args.trace, use_llm=not args.no_llm)
        return

    if args.no_llm:
        print(format_detect(args.question))
        return
    answer, used, err = run_agent(args.question, use_llm=True)
    _print_result(answer, used, err, args.trace)


if __name__ == "__main__":
    main()