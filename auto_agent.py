# -*- coding: utf-8 -*-
"""
auto_agent.py — 全球汽车出口(外贸)动态监测助手 (HTTP :8906)

三层架构:
  data  : auto_data.py  (UN Comtrade 年度出口 + 月度维护库 + 技术报告)
  agent : qwen-plus TOOLS 模式问答 (无 key 时降级为本地格式化回复)
  web   : Flask + ECharts 动态监测页 (全球出口/月度动态/技术报告 三页签)

启动: python3 auto_agent.py               # 前台
      python3 auto_agent.py --port 8906   # 指定端口
"""
import json
import os
import re
import sys
from datetime import datetime

import auto_data

# ---------------------------------------------------------------- 配置
PORT = 8906
HOST = "0.0.0.0"
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen-plus")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECHARTS_JS = os.path.join(BASE_DIR, "stock_assets", "echarts.min.js")

# ---------------------------------------------------------------- LLM
def _load_key():
    """读取 key: 环境变量 LLM_API_KEY -> DASHSCOPE_API_KEY -> .env 文件"""
    env = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY_ALT")
    if env:
        return env.strip()
    for name in (".env", os.path.join(BASE_DIR, ".env")):
        if os.path.exists(name):
            with open(name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DASHSCOPE_API_KEY") and "=" in line:
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _chat(messages, tools=None):
    """调用 qwen-plus (compatible-mode), 返回 dict 或 None"""
    import json as _j
    import ssl as _ssl
    import urllib.request as _u
    key = _load_key()
    if not key:
        return None
    try:  # venv 缺 CA 链, 用 certifi 上下文
        import certifi
        _ctx = _ssl.create_default_context(cafile=certifi.where())
    except Exception:
        _ctx = _ssl.create_default_context()
    body = {"model": MODEL, "messages": messages, "temperature": 0.3}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = _u.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=_j.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "User-Agent": "auto-agent/1.0"})
    try:
        with _u.urlopen(req, timeout=60, context=_ctx) as r:
            return _j.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"[llm] chat error: {e}", file=sys.stderr)
        return None


def _tool_schema(name, desc, props, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}


TOOLS = [
    _tool_schema(
        "get_export_overview",
        "获取全球主要国家/地区汽车(HS 8703)年度出口概览: 出口额/出口量排名、同比变化、年度趋势。参数 year 可省略(用最新年)。",
        {"year": {"type": "integer", "description": "年份, 如 2024; 缺省为最新年"}}),
    _tool_schema(
        "get_monthly_trend",
        "获取中国汽车月度出口量序列(万辆)与最近月份环比, 用于分析月度动态与季节性。",
        {"months": {"type": "integer", "description": "返回最近 N 个月, 缺省 12"}}),
    _tool_schema(
        "get_manufacturer_rank",
        "获取中国主要出口车企(奇瑞/比亚迪/上汽/吉利/长安/长城/特斯拉中国等)月度出口结构排名。",
        {"ym": {"type": "string", "description": "月份 YYYYMM, 缺省为最新月"}}),
    _tool_schema(
        "get_report",
        "生成并获取完整技术分析报告(Markdown), 含全球格局/中国月度/车企结构/结论与数据来源说明。",
        {}),
    _tool_schema(
        "data_status",
        "查看数据源状态: Comtrade 最近更新时间、错误、月度维护库记录数、演示数据标记。",
        {}),
]


def _tool_call(name, args):
    args = args or {}
    if name == "get_export_overview":
        year = args.get("year")
        tep = auto_data.top_exporters(year=year, limit=10)
        if not tep:
            return {"ok": False, "info": "暂无数据, 请稍后重试"}
        g = auto_data.global_trend()
        lines = [f"最新年: {tep[0]['year']} | 全球TOP10出口国(按出口额):"]
        for i, r in enumerate(tep, 1):
            q = f"{r['qty'] / 1e4:,.1f}万辆" if r["qty"] else "-"
            lines.append(f"{i}. {r['name_cn']}: 出口额 {r['value_usd'] / 1e8:,.0f} 亿美元, 出口量 {q}")
        return {"ok": True, "text": "\n".join(lines),
                "top": tep, "trend": {"years": g["years"], "series": g["series"]}}
    if name == "get_monthly_trend":
        n = int(args.get("months") or 12)
        ms = auto_data.monthly_series()[-n:]
        if not ms:
            return {"ok": False, "info": "暂无月度数据"}
        lines = [f"最近 {len(ms)} 个月中国汽车出口(万辆):"]
        for i, r in enumerate(ms):
            mom = ""
            if i > 0 and ms[i - 1].get("total"):
                d = (r["total"] / ms[i - 1]["total"] - 1) * 100
                mom = f" ({d:+.1f}%)"
            lines.append(f"- {r['ym'][:4]}-{r['ym'][4:]}: {r['total']} 万辆{mom}{'[演示]' if r.get('source') == 'sample' else ''}")
        return {"ok": True, "text": "\n".join(lines), "series": ms}
    if name == "get_manufacturer_rank":
        ms = auto_data.monthly_series()
        if not ms:
            return {"ok": False, "info": "暂无月度数据"}
        ym = args.get("ym") or ms[-1]["ym"]
        rec = next((r for r in ms if r["ym"] == ym), ms[-1])
        mfr = rec.get("mfr") or {}
        lines = [f"{rec['ym'][:4]}-{rec['ym'][4:]} 中国车企出口结构(万辆):"]
        for k, v in sorted(mfr.items(), key=lambda x: -x[1]):
            lines.append(f"- {k}: {v} 万辆")
        return {"ok": True, "text": "\n".join(lines), "ym": rec["ym"], "mfr": mfr}
    if name == "get_report":
        return {"ok": True, "text": auto_data.build_report()}
    if name == "data_status":
        w = auto_data.world_exports()
        ms = auto_data.monthly_series()
        real = [r for r in ms if r.get("source") != "sample"]
        return {"ok": True, "text": (
            f"Comtrade: rows={len(w.get('rows', []))} updated={w.get('updated', '-')} err={w.get('error') or '无'}\n"
            f"月度维护库: 共 {len(ms)} 个月, 真实数据 {len(real)} 条, 其余为演示标记")}
    return {"ok": False, "info": f"未知工具 {name}"}


SYSTEM_PROMPT = """你是"全球汽车出口监测助手", 负责分析全球及中国汽车出口(外贸)动态。
可以调用工具获取: 全球年度出口格局(Comtrade 真实数据)、中国月度出口序列、中国车企出口结构、技术报告。
回答要求:
1. 优先用中文并以数据说话, 引用具体数字时给出单位和时间;
2. 若数据为演示数据(source=sample), 必须说明"该月度数据为演示占位, 真实月度请录入";
3. 不编造不存在的出口数据; 出口量与出口额概念区分清楚(量=辆, 额=美元);
4. 结尾可给出一句简短的分析结论或风险提示。"""


def agent_answer(question, history=None):
    """问答入口: 返回 (answer, tools_used, error)"""
    if not _load_key():
        return ("当前未配置 DASHSCOPE_API_KEY, 已切换到本地模式。\n\n" + _local_answer(question), [], "no-key")
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}] +
                (history or [])[-6:] +
                [{"role": "user", "content": question}])
    used = []
    for _ in range(4):
        resp = _chat(messages, tools=TOOLS)
        if not resp:
            return "模型调用失败, 请稍后重试。", used, "llm-error"
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content") or "(空回复)", used, None
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            res = _tool_call(name, args)
            used.append(name)
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "name": name,
                             "content": json.dumps(res, ensure_ascii=False)})
    return "步骤过多, 已停止。", used, "too-many-rounds"


def _local_answer(question):
    """无 LLM key 时的本地格式化回复"""
    if re.search(r"全球|国家|排名|top", question, re.I):
        tep = auto_data.top_exporters(limit=8)
        if not tep:
            return "全球出口数据暂不可用, 请检查网络后重试。"
        y = tep[0]["year"]
        lines = [f"{y}年全球汽车出口额 TOP8(UN Comtrade, HS 8703):"]
        for i, r in enumerate(tep, 1):
            q = f"{r['qty'] / 1e4:,.1f}万辆" if r["qty"] else "数量未报告"
            lines.append(f"{i}. {r['name_cn']}: {r['value_usd'] / 1e8:,.0f} 亿美元, {q}")
        return "\n".join(lines)
    if re.search(r"月度|每月|趋势|环比", question):
        ms = auto_data.monthly_series()[-6:]
        if not ms:
            return "暂无月度数据。"
        lines = ["最近 6 个月中国汽车出口(万辆):"]
        for r in ms:
            tag = "(演示)" if r.get("source") == "sample" else ""
            lines.append(f"- {r['ym'][:4]}-{r['ym'][4:]}: {r['total']} 万辆 {tag}")
        return "\n".join(lines)
    if re.search(r"车企|厂家|奇瑞|比亚迪|厂家排名", question):
        ms = auto_data.monthly_series()
        if not ms:
            return "暂无月度数据。"
        rec = ms[-1]
        mfr = rec.get("mfr") or {}
        lines = [f"{rec['ym'][:4]}-{rec['ym'][4:]} 中国车企出口结构(万辆):"]
        for k, v in sorted(mfr.items(), key=lambda x: -x[1]):
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
    if re.search(r"报告|分析", question):
        return auto_data.build_report()[:800]
    return ("可问我: 全球汽车出口排名 / 中国月度出口趋势 / 车企出口结构 / 生成技术报告。\n"
            "也可以查看网页上的动态监测图。")


# ---------------------------------------------------------------- HTTP
try:
    from flask import Flask, request, jsonify, Response
    app = Flask(__name__)
except ImportError:
    print("需要 Flask: pip install flask", file=sys.stderr)
    sys.exit(1)


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/")
def page():
    return Response(PAGE, content_type="text/html; charset=utf-8")


@app.route("/echarts.min.js")
def echarts():
    return Response(open(ECHARTS_JS, "rb").read(), content_type="application/javascript")


@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "auto-export-monitor",
                    "llm": "ready" if _load_key() else "no-key", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


@app.route("/api/overview")
def api_overview():
    """全球出口监测: 逐年 TOP 榜单 + 多国趋势"""
    tep = auto_data.world_exports().get("rows", [])
    g = auto_data.global_trend()
    latest = sorted(set(r["year"] for r in tep))[-1] if tep else None
    top = [r for r in tep if r["year"] == latest] if latest else []
    return jsonify({"ok": True, "latest_year": latest, "top": top,
                    "trend": g, "updated": auto_data.world_exports().get("updated", "")})


@app.route("/api/monthly")
def api_monthly():
    ms = auto_data.monthly_series()
    return jsonify({"ok": True, "series": ms})


@app.route("/api/monthly", methods=["POST"])
def api_monthly_add():
    """半自动录入/更新一条月度记录: {ym, total, note?, mfr?}"""
    try:
        rec = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "JSON 请求体无效"}), 400
    r = auto_data.add_monthly(rec)
    if not r["ok"]:
        return jsonify(r), 400
    return jsonify(r)


@app.route("/api/news-parse", methods=["POST"])
def api_news_parse():
    """best-effort: 从新闻抓取最新中国月度出口快讯并自动录入"""
    res = auto_data.update_from_news()
    added = []
    for p in res.get("parsed", []):
        r = auto_data.add_monthly({**p, "source": "news"})
        if r.get("ok"):
            added.append(p)
    return jsonify({"ok": True, "hits": res.get("hits", []), "parsed": res.get("parsed", []), "added": added})


@app.route("/api/report")
def api_report():
    md = auto_data.build_report()
    fmt = request.args.get("format", "html")
    if fmt == "md":
        return Response(md, content_type="text/markdown; charset=utf-8")
    return jsonify({"ok": True, "md": md, "html": _md_to_html(md)})


@app.route("/api/qa", methods=["POST"])
def api_qa():
    data = request.get_json(force=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "问题为空"}), 400
    ans, used, err = agent_answer(q, data.get("history"))
    return jsonify({"ok": True, "answer": ans, "tools_used": used, "error": err})


def _md_to_html(md):
    """极简 Markdown -> HTML (标题/表格/列表/引用/粗体/分隔线)"""
    lines = md.split("\n")
    out, in_table, in_list = [], False, False

    def close():
        nonlocal in_table, in_list
        if in_table:
            out.append("</table>")
            in_table = False
        if in_list:
            out.append("</ul>")
            in_list = False

    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            if not in_table:
                close()
                out.append('<table style="border-collapse:collapse;width:100%;margin:8px 0">')
                in_table = True
            tag = "th" if (in_table and not re.search(r"<t[hd]", "".join(out[-2:])) and len([x for x in out if "<tr>" in x]) == 0) else "td"
            # 简单启发: 表头后紧跟分隔行, 第一行数据为表头
            if in_table and not any("<th" in x for x in out):
                tag = "th"
            row = "".join(f'<{tag} style="border:1px solid #ddd;padding:6px 10px;text-align:left">{_esc(c)}</{tag}>' for c in cells)
            out.append(f"<tr>{row}</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if s.startswith("#"):
            close()
            lvl = min(len(s) - len(s.lstrip("#")), 4)
            out.append(f"<h{lvl} style='margin:14px 0 8px'>{_esc(s.lstrip('#').strip())}</h{lvl}>")
        elif s.startswith("- ") or s.startswith("* "):
            if not in_list:
                out.append("<ul style='margin:6px 0;padding-left:22px'>")
                in_list = True
            out.append(f"<li>{_esc(s[2:])}</li>")
        elif s.startswith("> "):
            close()
            out.append(f"<blockquote style='border-left:3px solid #ccc;padding:4px 12px;color:#666'>{_esc(s[2:])}</blockquote>")
        elif re.fullmatch(r"-{3,}", s):
            close()
            out.append("<hr style='border:none;border-top:1px solid #eee;margin:12px 0'>")
        elif not s:
            close()
        else:
            close()
            out.append(f"<p style='margin:6px 0'>{_esc(s)}</p>")
    close()
    return "\n".join(out)


def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("**", "").replace("`", ""))


# ---------------------------------------------------------------- 页面
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>全球汽车出口动态监测</title>
<script src="/echarts.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f7fa;color:#222}
header{background:linear-gradient(135deg,#1e3c72,#2a5298);color:#fff;padding:16px 24px}
header h1{font-size:20px}header p{font-size:12px;opacity:.85;margin-top:4px}
.wrap{max-width:1200px;margin:0 auto;padding:16px}
.tabs{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
.tab{padding:9px 20px;border-radius:20px;background:#e8edf5;cursor:pointer;font-size:14px;user-select:none}
.tab.active{background:#2a5298;color:#fff}
.tab:hover:not(.active){background:#d7e0ee}
.panel{display:none}.panel.show{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:12px 0}
.card{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.card .t{font-size:12px;color:#888}.card .v{font-size:22px;font-weight:700;color:#1e3c72;margin-top:4px}
.card .s{font-size:11px;color:#999;margin-top:2px}
.chart{background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin:12px 0}
.chart h3{font-size:14px;color:#444;margin:2px 4px 8px}
#c1,#c2,#c3,#c5{height:340px}
#c4{height:300px}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;background:#fff;border-radius:8px;padding:12px;line-height:1.7}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;background:#fef3c7;color:#92400e;margin-left:6px}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left}
th{background:#f0f4fb;font-weight:600;color:#444}
tr:hover td{background:#f8fafd}
.form{display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin:12px 0}
.form input,.form select{padding:8px 10px;border:1px solid #ccc;border-radius:6px;font-size:13px}
.btn{background:#2a5298;color:#fff;border:none;padding:9px 18px;border-radius:6px;cursor:pointer;font-size:13px}
.btn:hover{background:#1e3c72}
.btn.ghost{background:#fff;color:#2a5298;border:1px solid #2a5298}
.qa{position:fixed;right:18px;bottom:18px;width:380px;max-height:560px;background:#fff;border-radius:14px;box-shadow:0 6px 30px rgba(0,0,0,.18);display:flex;flex-direction:column;overflow:hidden;z-index:99}
.qa .head{background:#2a5298;color:#fff;padding:10px 14px;font-size:14px;display:flex;justify-content:space-between;align-items:center}
.qa .body{flex:1;overflow-y:auto;padding:12px;font-size:13px;min-height:200px;max-height:340px}
.qa .row{margin-bottom:10px;line-height:1.6}
.qa .q{background:#eef3fc;border-radius:8px;padding:6px 10px;margin-left:30px;color:#1e3c72}
.qa .a{background:#f5f5f5;border-radius:8px;padding:6px 10px;margin-right:30px;color:#333;white-space:pre-wrap}
.qa .in{display:flex;border-top:1px solid #eee}
.qa .in input{flex:1;border:none;padding:10px 12px;font-size:13px;outline:none}
.qa .in button{border:none;background:#2a5298;color:#fff;padding:0 16px;cursor:pointer}
.footer{text-align:center;color:#999;font-size:11px;padding:20px 0 60px}
@media (max-width:900px){.qa{position:static;width:100%;margin:12px 0;max-height:400px}}
</style>
</head>
<body>
<header>
  <h1>全球汽车出口(外贸)动态监测 <span id="srcBadge" class="badge">加载中…</span></h1>
  <p>数据源: UN Comtrade (HS 8703 年度出口) · 月度维护库 · 自动刷新 | 端口 :8906</p>
</header>
<div class="wrap">
  <div class="tabs">
    <div class="tab active" data-p="p1">全球出口监测</div>
    <div class="tab" data-p="p2">月度动态</div>
    <div class="tab" data-p="p3">技术报告</div>
  </div>

  <div class="panel show" id="p1">
    <div class="cards" id="cards1"></div>
    <div class="chart"><h3>全球汽车出口额 TOP15 (最新年, 亿美元)</h3><div id="c1"></div></div>
    <div class="chart"><h3>主要国家/地区出口量趋势 (万辆/年)</h3><div id="c2"></div></div>
  </div>

  <div class="panel" id="p2">
    <div class="chart"><h3>中国汽车月度出口量 (万辆)<span id="monthSrc" class="badge" style="display:none">演示数据</span></h3><div id="c3"></div></div>
    <div class="chart"><h3>中国主要车企月度出口结构 (万辆, 堆叠)</h3><div id="c4"></div></div>
    <div class="form">
      <b style="font-size:13px">录入真实月度数据(半自动维护):</b>
      <input id="fym" placeholder="如 202610" style="width:110px" maxlength="6">
      <input id="ftotal" placeholder="出口量(万辆)" style="width:120px" type="number" step="0.1">
      <input id="fnote" placeholder="备注/来源, 如 中汽协快讯" style="flex:1;min-width:160px">
      <button class="btn" onclick="addMonthly()">录入</button>
      <button class="btn ghost" onclick="parseNews()">抓取最新快讯</button>
    </div>
  </div>

  <div class="panel" id="p3">
    <div class="cards" id="cards3" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"></div>
    <div style="display:flex;gap:10px;margin:10px 0">
      <a class="btn" href="/api/report?format=md" target="_blank" style="text-decoration:none">下载 Markdown 报告</a>
      <button class="btn ghost" onclick="loadReport()">刷新报告</button>
    </div>
    <div id="reportBox" class="mono" style="max-height:600px;overflow:auto">加载中…</div>
  </div>
</div>

<div class="qa">
  <div class="head"><span>智能问答</span><span style="font-size:11px;opacity:.8">qwen-plus</span></div>
  <div class="body" id="qaBody"><div class="row"><div class="a">你好! 可问: 全球汽车出口排名 / 中国最近月度出口趋势 / 车企出口结构 / 生成分析报告</div></div></div>
  <div class="in"><input id="qai" placeholder="输入问题, 回车发送" onkeydown="if(event.key==='Enter')ask()"><button onclick="ask()">发送</button></div>
</div>

<div class="footer">Global Auto Export Monitor · 数据源不可用时自动降级 · 月度数据以页面标注为准</div>

<script>
const fmt = n => (n == null ? '-' : Number(n).toLocaleString('zh-CN'));
let OV = null, MS = [], REPORT_LOADED = false;
async function j(u, o) { const r = await fetch(u, o); return r.json(); }
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}

/* ---------- Tab 切换 ---------- */
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('show'));
  t.classList.add('active');
  document.getElementById(t.dataset.p).classList.add('show');
  if(t.dataset.p==='p3' && !REPORT_LOADED) loadReport();
});

/* ---------- Tab1 全球出口 ---------- */
async function loadOverview() {
  const d = await j('/api/overview');
  OV = d;
  const top = d.top || [];
  document.getElementById('srcBadge').textContent = '已更新 ' + (d.updated || '').slice(5, 16);
  const y = d.latest_year || '-';
  const totalV = top.length ? top.reduce((s, r) => s + (r.value_usd || 0), 0) / 1e8 : 0;
  const totalQ = top.length ? top.reduce((s, r) => s + (r.qty || 0), 0) / 1e4 : 0;
  const cn = top.find(r => r.code === 156);
  document.getElementById('cards1').innerHTML = [
    ['监测年份', y, '全球主要出口国'],
    ['TOP15 出口额合计', fmt(Math.round(totalV)) + ' 亿美元', 'HS 8703 汽车'],
    ['TOP15 出口量合计', fmt(Math.round(totalQ)) + ' 万辆', '含估算值'],
    ['中国出口额', cn ? fmt(Math.round(cn.value_usd / 1e8)) + ' 亿美元' : '-', cn ? ('出口量 ' + fmt(Math.round(cn.qty / 1e4)) + ' 万辆') : ''],
  ].map(x => `<div class="card"><div class="t">${x[0]}</div><div class="v">${x[1]}</div><div class="s">${x[2]}</div></div>`).join('');

  const top10 = top.slice(0, 15);
  chart('c1', {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 8, right: 60, top: 20, bottom: 10, containLabel: true },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: top10.map(r => r.name_cn).reverse() },
    series: [{
      type: 'bar', data: top10.map(r => Math.round(r.value_usd / 1e8)).reverse(),
      itemStyle: { borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', formatter: p => fmt(p.value) },
      color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: '#2a5298' }, { offset: 1, color: '#5b9bd5' }])
    }]
  });

  // 多国趋势: 取出口额前6
  const g = d.trend || { years: [], series: [] };
  const top6 = g.series.slice(0, 6);
  chart('c2', {
    tooltip: { trigger: 'axis' },
    legend: { data: top6.map(s => s.name_cn), top: 0 },
    grid: { left: 8, right: 20, top: 30, bottom: 10, containLabel: true },
    xAxis: { type: 'category', data: g.years.map(String) },
    yAxis: { type: 'value', name: '万辆' },
    series: top6.map((s, i) => ({
      name: s.name_cn, type: 'line', smooth: true, data: s.qty.map(v => Math.round(v / 1e4)),
      symbolSize: 6, lineStyle: { width: 2.5 },
      itemStyle: { color: ['#2a5298', '#e67e22', '#27ae60', '#c0392b', '#8e44ad', '#16a085'][i % 6] }
    }))
  });
}

/* ---------- Tab2 月度 ---------- */
async function loadMonthly() {
  const d = await j('/api/monthly');
  MS = d.series || [];
  const hasReal = MS.some(r => r.source !== 'sample');
  document.getElementById('monthSrc').style.display = hasReal ? 'none' : '';
  const xs = MS.map(r => r.ym.slice(4) + '月');
  chart('c3', {
    tooltip: { trigger: 'axis' },
    grid: { left: 8, right: 20, top: 30, bottom: 10, containLabel: true },
    xAxis: { type: 'category', data: xs },
    yAxis: { type: 'value', name: '万辆' },
    series: [{
      name: '出口量', type: 'line', smooth: true, data: MS.map(r => r.total),
      symbolSize: 5, areaStyle: { opacity: .15 }, lineStyle: { width: 2.5, color: '#2a5298' },
      itemStyle: { color: '#2a5298' },
      markLine: { data: [{ type: 'average', name: '均值' }], lineStyle: { type: 'dashed', color: '#999' } }
    }]
  });

  const mfrs = ['奇瑞', '比亚迪', '上汽', '吉利', '长安', '长城', '特斯拉中国', '其他'];
  const colors = ['#2a5298', '#e67e22', '#27ae60', '#c0392b', '#8e44ad', '#16a085', '#d35400', '#95a5a6'];
  chart('c4', {
    tooltip: { trigger: 'axis' },
    legend: { data: mfrs, top: 0, type: 'scroll' },
    grid: { left: 8, right: 20, top: 34, bottom: 40, containLabel: true },
    xAxis: { type: 'category', data: xs, axisLabel: { rotate: 45 } },
    yAxis: { type: 'value', name: '万辆' },
    series: mfrs.map((m, i) => ({
      name: m, type: 'bar', stack: 't', barMaxWidth: 34,
      data: MS.map(r => (r.mfr && r.mfr[m] != null) ? r.mfr[m] : 0),
      itemStyle: { color: colors[i] }
    }))
  });
}

async function addMonthly() {
  const ym = document.getElementById('fym').value.trim();
  const total = parseFloat(document.getElementById('ftotal').value);
  const note = document.getElementById('fnote').value.trim();
  if (!/^\d{6}$/.test(ym) || isNaN(total)) { alert('请填写 YYYYMM 和出口量'); return; }
  const r = await j('/api/monthly', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ym, total, note, source: 'manual' }) });
  if (r.ok) { alert('已录入 ' + ym + ': ' + total + ' 万辆'); loadMonthly(); }
  else alert('录入失败: ' + (r.error || ''));
}

async function parseNews() {
  const r = await j('/api/news-parse', { method: 'POST' });
  const added = r.added || [];
  if (added.length) { alert('已从新闻解析并录入 ' + added.length + ' 条'); loadMonthly(); }
  else alert('未解析到新的出口快讯 (hits=' + (r.hits || []).length + '), 请手动录入');
}

/* ---------- Tab3 报告 ---------- */
async function loadReport() {
  REPORT_LOADED = true;
  document.getElementById('reportBox').innerHTML = '加载中…';
  const d = await j('/api/report');
  document.getElementById('reportBox').innerHTML = d.html || ('<pre>' + esc(d.md) + '</pre>');
  document.getElementById('cards3').innerHTML = [
    ['全球年度出口', d.md.includes('中国') ? '可用' : '不可用', 'UN Comtrade'],
    ['中国月度序列', MS.some ? (MS.some(r => r.source !== 'sample') ? '含真实数据' : '演示数据') : '-', '月度维护库'],
    ['报告更新', new Date().toLocaleTimeString('zh-CN'), '自动生成'],
  ].map(x => `<div class="card"><div class="t">${x[0]}</div><div class="v" style="font-size:16px">${x[1]}</div><div class="s">${x[2]}</div></div>`).join('');
}

/* ---------- 问答 ---------- */
async function ask() {
  const inp = document.getElementById('qai');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  const body = document.getElementById('qaBody');
  body.insertAdjacentHTML('beforeend', `<div class="row"><div class="q">${esc(q)}</div><div class="a">思考中…</div></div>`);
  body.scrollTop = body.scrollHeight;
  try {
    const r = await j('/api/qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) });
    const a = body.lastElementChild.querySelector('.a');
    a.textContent = r.answer || '(无回复)';
    if (r.error === 'no-key') a.innerHTML = a.textContent.replace(/\n/g, '<br>');
  } catch (e) {
    body.lastElementChild.querySelector('.a').textContent = '请求失败: ' + e;
  }
  body.scrollTop = body.scrollHeight;
}

function chart(id, opt) {
  const el = document.getElementById(id);
  if (!el) return;
  const c = echarts.getInstanceByDom(el) || echarts.init(el);
  c.setOption(opt, true);
}

loadOverview(); loadMonthly();
setInterval(() => { loadOverview(); loadMonthly(); }, 5 * 60 * 1000);
window.addEventListener('resize', () => ['c1','c2','c3','c4'].forEach(i => { const el = document.getElementById(i); if (el) echarts.getInstanceByDom(el)?.resize(); }));
</script>
</body>
</html>"""


# ---------------------------------------------------------------- main
def main():
    global PORT
    if "--port" in sys.argv:
        PORT = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"[auto-export] 启动 http://127.0.0.1:{PORT}  (llm={'ready' if _load_key() else 'NO-KEY'})")
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()