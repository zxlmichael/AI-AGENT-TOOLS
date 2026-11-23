#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""天气预报助手：全国主要城市未来 10 天天气预报 + 问答页面

- 数据层 weather_data.py：Open-Meteo（免费无 key，10 天）
- 助手层：qwen-plus 汇总解读，无 key 时本地格式化兜底
- HTTP :8904：/（看板+问答页）、/api/forecast、/api/cities、/ask、/health

用法：
    .venv/bin/python weather_agent.py "北京未来三天天气怎么样？"
    .venv/bin/python weather_agent.py --chat
    .venv/bin/python weather_agent.py --serve --port 8904
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import weather_data as wd

HERE = Path(__file__).resolve().parent
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

AGENT_SYSTEM = """你是「天气预报助手」，提供全国主要城市未来 10 天天气预报与出行建议。
你可以调用工具获取实时预报。回答必须遵守：
1. 中文回答，先给结论（某城市未来 N 天天气趋势），再列每日要点；
2. 天气数据以工具返回为准，说明数据时间；不要编造工具未返回的城市或数据；
3. 涉及"带伞/穿衣/出行"等建议时，结合降水概率与温度给出实用提示；
4. 城市不在支持列表时，告知支持的城市范围（直辖市+省会+计划单列市等）；
5. 预报仅供参考，天气变化以官方发布为准。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "查询某城市未来 10 天天气预报（天气/最高最低温/降水概率）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如：北京/上海/深圳"},
                    "days": {"type": "integer", "description": "预报天数 1-10，默认 10"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_cities",
            "description": "按名称/省份模糊搜索支持的城市。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "城市名或省份关键词"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cities",
            "description": "对比多个城市同一时段的天气（如：北京、上海、广州）。cities 为城市名数组，最多 5 个。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cities": {"type": "array", "items": {"type": "string"},
                               "description": "城市名列表，1-5 个"},
                },
                "required": ["cities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_status",
            "description": "查看数据源状态与支持城市数量。",
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


def _json(data, limit: int = 10000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def tool_get_forecast(city: str, days: int = 10) -> str:
    return _json(wd.forecast(city, days))


def tool_find_cities(query: str = "") -> str:
    return _json({"query": query, "hits": wd.find_cities(query)[:10]})


def tool_compare_cities(cities: list) -> str:
    cities = [c for c in (cities or []) if isinstance(c, str)][:5]
    return _json({"cities": wd.multi_forecast(cities)})


def tool_data_status() -> str:
    return _json({"source": "Open-Meteo (api.open-meteo.com)",
                  "cities": len(wd.CITIES), "max_days": 10,
                  "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


TOOL_IMPL = {
    "get_forecast": tool_get_forecast,
    "find_cities": tool_find_cities,
    "compare_cities": tool_compare_cities,
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
        return "请输入要查询的城市与问题。", [], ""
    if not use_llm or not llm_available():
        return format_forecast(question), [], "no-llm-fallback"

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
                         "content": "请基于已获取的预报数据直接给出最终回答，不要再调用工具。"})
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

def format_forecast(text: str) -> str:
    """从问题里提取城市名（支持"北京天气""上海未来三天"等），输出 10 天预报文本。"""
    city = None
    for name in wd.CITIES:
        if name in text:
            city = name
            break
    if not city:
        return (f"未能从问题中识别城市。支持的城市（{len(wd.CITIES)} 个）：\n"
                + "、".join(list(wd.CITIES)[:20]) + " …")
    f = wd.forecast(city)
    lines = [f"【{city}（{f['province']}）未来 {len(f['days'])} 天预报】"
             f"（更新于 {f['updated']}）"]
    for d in f["days"]:
        prob = f" 降水{d['precip_prob']}%" if d["precip_prob"] is not None else ""
        lines.append(f"  {d['date']} {d['weekday']} {d['desc']} "
                     f"{d['tmin']:.0f}~{d['tmax']:.0f}℃{prob}")
    lines.append("（数据来源：Open-Meteo，仅供参考）")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask("weather_agent")
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
        return jsonify({"ok": True, "service": "weather-assistant",
                        "llm": "ready" if llm_available() else "missing-key",
                        "cities": len(wd.CITIES),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.get("/api/cities")
    def cities():
        return jsonify(wd.cities_info())

    @app.get("/api/forecast")
    def forecast_api():
        city = str(request.args.get("city", "")).strip()
        days = int(request.args.get("days", 10))
        if not city:
            return jsonify({"ok": False, "error": "缺少 city 参数"}), 400
        try:
            return jsonify({"ok": True, "data": wd.forecast(city, days)})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
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
<title>天气预报助手 · 全国主要城市未来10天</title>
<script src="/assets/echarts.min.js"></script>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8;
          --accent:#3b82f6; --warm:#f59e0b; --cool:#38bdf8; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
  .wrap { max-width:1280px; margin:0 auto; padding:20px 16px 40px; }
  header { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  header h1 { font-size:20px; font-weight:700; }
  .meta { color:var(--muted); font-size:12px; }
  .toolbar { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
  .toolbar input { flex:1; min-width:220px; background:#0b1222; border:1px solid var(--line);
        color:var(--text); border-radius:8px; padding:9px 14px; font-size:14px; }
  .toolbar select { background:#0b1222; border:1px solid var(--line); color:var(--text);
        border-radius:8px; padding:9px 12px; font-size:14px; }
  .toolbar button { background:var(--accent); color:#fff; border:none; border-radius:8px;
        padding:0 22px; font-size:14px; cursor:pointer; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .panel h2 { font-size:15px; font-weight:600; margin-bottom:10px; }
  .days { display:grid; grid-template-columns:repeat(auto-fill,minmax(112px,1fr)); gap:10px; }
  .day { background:#0b1222; border:1px solid var(--line); border-radius:10px; padding:10px 8px; text-align:center; }
  .day.today { border-color:var(--accent); }
  .day .dt { font-size:11px; color:var(--muted); }
  .day .dw { font-size:12px; font-weight:600; margin:2px 0; }
  .day .ic { font-size:24px; line-height:1.3; }
  .day .ds { font-size:12px; color:var(--cool); }
  .day .tp { font-size:15px; font-weight:700; }
  .day .pr { font-size:11px; color:var(--warm); }
  .chart { width:100%; height:300px; }
  .qa { display:flex; gap:10px; }
  .qa input { flex:1; background:#0b1222; border:1px solid var(--line); color:var(--text);
        border-radius:8px; padding:10px 14px; font-size:14px; }
  .qa button { background:var(--accent); color:#fff; border:none; border-radius:8px;
        padding:0 22px; font-size:14px; cursor:pointer; }
  .qaresult { background:#0b1222; border:1px solid var(--line); border-radius:8px;
        padding:12px 16px; font-size:13.5px; margin-top:10px; white-space:pre-wrap; display:none;
        max-height:320px; overflow:auto; }
  .foot { color:var(--muted); font-size:11px; text-align:center; margin-top:6px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>天气预报助手</h1>
    <span class="meta">全国主要城市 · 未来 10 天 · Open-Meteo 数据</span>
    <span class="meta">更新：<span id="updated">--</span></span>
  </header>

  <div class="toolbar">
    <input id="cityInput" list="cityList" placeholder="输入城市名，如：北京 / 上海 / 深圳" onkeydown="if(event.key==='Enter')loadForecast()">
    <datalist id="cityList"></datalist>
    <button onclick="loadForecast()">查询</button>
  </div>

  <div class="panel">
    <h2 id="cityTitle">10 天预报</h2>
    <div class="days" id="dayCards"></div>
    <div class="chart" id="tempChart"></div>
  </div>

  <div class="panel">
    <h2>天气问答（大模型）</h2>
    <div class="qa">
      <input id="qaInput" placeholder="如：北京未来一周有雨吗？会不会降温？上海和广州哪个更暖和？" onkeydown="if(event.key==='Enter')ask()">
      <button onclick="ask()">提问</button>
    </div>
    <div class="qaresult" id="qaResult"></div>
  </div>

  <div class="foot">数据来源：Open-Meteo 免费天气 API · 预报仅供参考，出行以官方发布为准</div>
</div>

<script>
const WMO_ICON = {0:"☀️",1:"🌤️",2:"⛅",3:"☁️",45:"🌫️",48:"🌫️",51:"🌦️",53:"🌦️",55:"🌧️",
  56:"🌧️",57:"🌧️",61:"🌧️",63:"🌧️",65:"🌧️",66:"🌧️",67:"🌧️",71:"🌨️",73:"🌨️",75:"❄️",
  77:"❄️",80:"🌦️",81:"🌧️",82:"⛈️",85:"🌨️",86:"❄️",95:"⛈️",96:"⛈️",99:"⛈️"};
const tempChart = echarts.init(document.getElementById("tempChart"));
let currentCity = "";

async function initCities() {
  try {
    const r = await (await fetch("/api/cities")).json();
    document.getElementById("cityList").innerHTML =
      (r.cities || []).map(c => '<option value="' + c.name + '">').join("");
  } catch (e) { /* 忽略 */ }
}

async function loadForecast() {
  const city = document.getElementById("cityInput").value.trim();
  if (!city) return;
  try {
    const r = await (await fetch("/api/forecast?city=" + encodeURIComponent(city) + "&days=10", { cache: "no-store" })).json();
    if (!r.ok) { alert("查询失败：" + (r.error || "未知错误")); return; }
    currentCity = city;
    renderForecast(r.data);
  } catch (e) { alert("查询失败：" + e.message); }
}

function renderForecast(d) {
  document.getElementById("cityTitle").textContent = d.city + "（" + d.province + "）未来 " + d.days.length + " 天预报";
  document.getElementById("updated").textContent = d.updated;
  document.getElementById("dayCards").innerHTML = d.days.map(x =>
    '<div class="day' + (x.weekday === "今天" ? " today" : "") + '">' +
    '<div class="dt">' + x.date.slice(5) + '</div>' +
    '<div class="dw">' + x.weekday + '</div>' +
    '<div class="ic">' + (WMO_ICON[x.code] || "🌡️") + '</div>' +
    '<div class="ds">' + x.desc + '</div>' +
    '<div class="tp">' + x.tmin + "~" + x.tmax + "℃</div>" +
    (x.precip_prob != null ? '<div class="pr">降水 ' + x.precip_prob + "%</div>" : "") +
    "</div>").join("");
  tempChart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["最高温", "最低温"], textStyle: { color: "#94a3b8" } },
    grid: { left: 46, right: 16, top: 36, bottom: 26 },
    xAxis: { type: "category", data: d.days.map(x => x.date.slice(5) + " " + x.weekday),
             axisLabel: { color: "#94a3b8" }, axisLine: { lineStyle: { color: "#334155" } } },
    yAxis: { type: "value", name: "℃", nameTextStyle: { color: "#94a3b8" },
             axisLabel: { color: "#94a3b8" }, splitLine: { lineStyle: { color: "#26324a" } } },
    series: [
      { name: "最高温", type: "line", data: d.days.map(x => x.tmax), smooth: true,
        showSymbol: true, lineStyle: { width: 2, color: "#f59e0b" }, itemStyle: { color: "#f59e0b" } },
      { name: "最低温", type: "line", data: d.days.map(x => x.tmin), smooth: true,
        showSymbol: true, lineStyle: { width: 2, color: "#38bdf8" }, itemStyle: { color: "#38bdf8" } },
    ],
  }, true);
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

window.addEventListener("resize", () => tempChart.resize());
initCities();
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
    print("天气预报助手已就绪，输入问题开始查询；输入 exit / quit 退出。\n")
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
    ap = argparse.ArgumentParser(description="天气预报助手（全国主要城市未来10天）")
    ap.add_argument("question", nargs="?", default="", help="问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8904)
    ap.add_argument("--trace", action="store_true", help="显示工具调用过程")
    ap.add_argument("--no-llm", action="store_true", help="不调用大模型，本地格式化输出")
    args = ap.parse_args()

    if args.serve:
        app = create_app()
        print(f"天气预报助手 HTTP 服务：http://{args.host}:{args.port}")
        print("  页面: /   预报: /api/forecast?city=北京   城市: /api/cities")
        print('  问答: POST /ask {"question":"..."}   /health')
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if args.chat or not args.question:
        _chat_loop(args.trace, use_llm=not args.no_llm)
        return

    if args.no_llm:
        print(format_forecast(args.question))
        return
    answer, used, err = run_agent(args.question, use_llm=True)
    _print_result(answer, used, err, args.trace)


if __name__ == "__main__":
    main()