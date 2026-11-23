# -*- coding: utf-8 -*-
"""
doc_kb_agent.py — 本地文档统计 + 本地知识库助手 (HTTP :8907)

  data : doc_kb.py (扫描统计 + RAG 向量检索)
  agent: qwen-plus TOOLS 模式问答, 检索增强 (RAG)
  web  : Flask + ECharts 页 (文档统计 / 知识库问答 两页签)

启动: .venv/bin/python doc_kb_agent.py
"""
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

import doc_kb

PORT = 8907
HOST = "0.0.0.0"
# LLM 供应商可配置(均 OpenAI 兼容 /v1/chat/completions):
#   LLM_BASE_URL + LLM_API_KEY + LLM_MODEL 环境变量 >
#   DASHSCOPE_API_KEY (默认百炼 qwen-plus)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")
MODEL = LLM_MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECHARTS_JS = os.path.join(BASE_DIR, "stock_assets", "echarts.min.js")

_build_thread = None


# ---------------------------------------------------------------- LLM
def _load_key():
    env = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
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
    import json as _j
    import ssl as _ssl
    import urllib.request as _u
    key = _load_key()
    if not key:
        return None
    try:
        import certifi
        _ctx = _ssl.create_default_context(cafile=certifi.where())
    except Exception:
        _ctx = _ssl.create_default_context()
    body = {"model": LLM_MODEL, "messages": messages, "temperature": 0.3}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = _u.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=_j.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "User-Agent": "doc-kb-agent/1.0"})
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
    _tool_schema("doc_stats", "获取本地文档扫描统计: 文档数量/总字符/中文字符/行数/扩展名与目录分布。", {}),
    _tool_schema("kb_search", "在本地知识库中按语义检索相关文档片段, 返回相似度/标题/来源文件。",
                 {"query": {"type": "string", "description": "检索问题/关键词"},
                  "k": {"type": "integer", "description": "返回条数, 缺省 4"}}),
    _tool_schema("kb_status", "查看知识库索引构建状态与统计数据。", {}),
]


def _tool_call(name, args):
    args = args or {}
    if name == "doc_stats":
        s = doc_kb.scan_stats()
        return {"ok": True, "text": (
            f"本地共 {s['n_files']:,} 个文档, {s['n_bytes']/1e6:.1f}MB; "
            f"字符 {s['n_chars']:,} (中文 {s['n_cn_chars']:,}), 行数 {s['n_lines']:,}"),
            "stats": {k: s[k] for k in ("n_files", "n_chars", "n_cn_chars", "n_lines", "n_bytes")}}
    if name == "kb_search":
        q = args.get("query", "")
        k = int(args.get("k") or 4)
        hits = doc_kb.retrieve(q, k=k)
        if not hits:
            return {"ok": False, "info": "无检索结果(索引未构建或未配置 API key)"}
        method = "向量" if (hits and hits[0].get("method") == "vector") else "关键词降级"
        return {"ok": True, "text": "\n".join(
            f"[{h['score']:.3f}] 《{h['title']}》 {h['src']} - {h['text'][:80]}..." for h in hits),
            "hits": hits, "method": method}
    if name == "kb_status":
        st = doc_kb.stats()
        bs = doc_kb.build_status()
        return {"ok": True, "text": f"索引: {st} | 构建: {bs}"}
    return {"ok": False, "info": f"未知工具 {name}"}


def _rag_answer(question):
    """RAG 问答: 检索 top-k -> 拼上下文 -> qwen-plus 生成; 无 key 时返回检索原文。"""
    hits = doc_kb.retrieve(question, k=4)
    method_note = ""
    if hits and hits[0].get("method") == "kw":
        method_note = "（向量服务不可用, 已降级为关键词检索）"
    if not _load_key():
        if not hits:
            return "知识库暂无匹配内容(索引未构建或未配置 API key)。"
        lines = ["(本地模式, 未配置 LLM key, 展示检索结果)" + method_note]
        for h in hits:
            lines.append(f"\n【{h['title']} · 相似度 {h['score']}】\n{h['text'][:300]}")
        return "\n".join(lines)
    ctx = "\n\n".join(f"【{h['title']} | {h['src']}】\n{h['text']}" for h in hits) or "(未检索到相关内容)"
    sys_p = ("你是本地文档知识库助手。请严格依据提供的文档片段回答问题; "
             "若片段不足, 明确说'根据现有文档无法确定', 不要编造。回答用中文, 简洁分点。")
    resp = _chat([{"role": "system", "content": sys_p},
                  {"role": "user", "content": f"文档片段:\n{ctx}\n\n问题: {question}"}])
    if not resp:
        return "模型调用失败, 请稍后重试。" + method_note
    ans = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or "(空回复)"
    if hits:
        srcs = sorted({h["src"] for h in hits})[:3]
        ans += f"\n\n—— 参考: " + " | ".join(srcs) + method_note
    return ans


SYSTEM_PROMPT = """你是"本地文档知识库助手", 面向个人本地文档(课程笔记/手册/代码)。
可以调用工具: doc_stats(文档统计), kb_search(语义检索文档片段), kb_status(索引状态)。
回答要求: 中文, 引用检索结果时标明来源文件; 不要编造文档内容; 统计数字给出单位。"""


def agent_answer(question, history=None):
    if not _load_key():
        return _rag_answer(question), [], "no-key"
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
    return jsonify({"ok": True, "service": "doc-kb", "llm": "ready" if _load_key() else "no-key",
                    "llm_model": LLM_MODEL, "llm_base": LLM_BASE_URL,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


@app.route("/api/docstats")
def api_docstats():
    return jsonify({"ok": True, **doc_kb.scan_stats()})


@app.route("/api/kb/status")
def api_kb_status():
    st = doc_kb.stats()
    bs = doc_kb.build_status()
    return jsonify({"ok": True, "index": st, "build": bs})


@app.route("/api/kb/search")
def api_kb_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "参数 q 为空"}), 400
    k = int(request.args.get("k") or 4)
    return jsonify({"ok": True, "hits": doc_kb.retrieve(q, k=k)})


@app.route("/api/kb/build", methods=["POST"])
def api_kb_build():
    global _build_thread
    if _build_thread and _build_thread.is_alive():
        return jsonify({"ok": False, "error": "构建已在进行中"}), 409
    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    _build_thread = threading.Thread(target=doc_kb.update, kwargs={"force": force}, daemon=True)
    _build_thread.start()
    return jsonify({"ok": True, "started": True, "force": force})


def _auto_update_loop(interval_minutes: int = 60):
    """常驻线程: 定期检查文件变化并增量更新索引(无变化则数秒结束)。"""
    global _build_thread
    while True:
        time.sleep(interval_minutes * 60)
        if _build_thread and _build_thread.is_alive():
            continue
        if not doc_kb.stats().get("ok"):
            continue  # 无索引时由 _start_build_if_needed 处理
        print(f"[kb] {datetime.now().strftime('%H:%M:%S')} 自动增量检查...", file=sys.stderr)
        try:
            res = doc_kb.update(force=False)
            if res.get("added"):
                print(f"[kb] 自动更新: 新增 {res['added']} 个切片, 跳过 {res.get('failed', 0)} 个",
                      file=sys.stderr)
        except Exception as e:
            print(f"[kb] 自动更新失败: {e}", file=sys.stderr)


@app.route("/api/qa", methods=["POST"])
def api_qa():
    data = request.get_json(force=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "问题为空"}), 400
    if "rag" in (data.get("mode") or ""):
        ans = _rag_answer(q)
        return jsonify({"ok": True, "answer": ans, "mode": "rag"})
    ans, used, err = agent_answer(q, data.get("history"))
    return jsonify({"ok": True, "answer": ans, "tools_used": used, "error": err})


# ---------------------------------------------------------------- 页面
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地文档统计 · 本地知识库</title>
<script src="/echarts.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f7fa;color:#222}
header{background:linear-gradient(135deg,#0f5c3e,#2a9d6e);color:#fff;padding:16px 24px}
header h1{font-size:20px}header p{font-size:12px;opacity:.85;margin-top:4px}
.wrap{max-width:1200px;margin:0 auto;padding:16px}
.tabs{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
.tab{padding:9px 20px;border-radius:20px;background:#e8f2ec;cursor:pointer;font-size:14px;user-select:none}
.tab.active{background:#0f5c3e;color:#fff}
.panel{display:none}.panel.show{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:12px 0}
.card{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.card .t{font-size:12px;color:#888}.card .v{font-size:21px;font-weight:700;color:#0f5c3e;margin-top:4px}
.card .s{font-size:11px;color:#999;margin-top:2px}
.chart{background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin:12px 0}
.chart h3{font-size:14px;color:#444;margin:2px 4px 8px}
#c1,#c2{height:320px}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left}
th{background:#f0f7f2;font-weight:600;color:#444}
tr:hover td{background:#f8fdf9}
.path{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#555;word-break:break-all}
.search{display:flex;gap:10px;margin:14px 0}
.search input{flex:1;padding:11px 14px;border:1px solid #ccc;border-radius:8px;font-size:14px;outline:none}
.search input:focus{border-color:#2a9d6e}
.btn{background:#0f5c3e;color:#fff;border:none;padding:0 22px;border-radius:8px;cursor:pointer;font-size:14px}
.btn:hover{opacity:.9}
.btn.ghost{background:#fff;color:#0f5c3e;border:1px solid #0f5c3e}
.hit{background:#fff;border-radius:10px;padding:12px 14px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.hit .h{font-size:14px;font-weight:600;color:#0f5c3e}
.hit .m{font-size:11px;color:#999;margin:3px 0}
.hit .b{font-size:13px;line-height:1.7;color:#333}
.hit .b mark{background:#fde68a;padding:0 2px;border-radius:2px}
.prog{background:#e8f2ec;border-radius:8px;height:10px;overflow:hidden;margin:10px 0}
.prog>div{background:linear-gradient(90deg,#2a9d6e,#0f5c3e);height:100%;width:0;transition:width .5s}
.qa{position:fixed;right:18px;bottom:18px;width:380px;max-height:560px;background:#fff;border-radius:14px;box-shadow:0 6px 30px rgba(0,0,0,.18);display:flex;flex-direction:column;overflow:hidden;z-index:99}
.qa .head{background:#0f5c3e;color:#fff;padding:10px 14px;font-size:14px;display:flex;justify-content:space-between}
.qa .body{flex:1;overflow-y:auto;padding:12px;font-size:13px;min-height:200px;max-height:340px}
.qa .row{margin-bottom:10px;line-height:1.6}
.qa .q{background:#e8f6ee;border-radius:8px;padding:6px 10px;margin-left:30px;color:#0f5c3e}
.qa .a{background:#f5f5f5;border-radius:8px;padding:6px 10px;margin-right:30px;color:#333;white-space:pre-wrap}
.qa .in{display:flex;border-top:1px solid #eee}
.qa .in input{flex:1;border:none;padding:10px 12px;font-size:13px;outline:none}
.qa .in button{border:none;background:#0f5c3e;color:#fff;padding:0 16px;cursor:pointer}
.footer{text-align:center;color:#999;font-size:11px;padding:20px 0 60px}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;background:#d1fae5;color:#065f46;margin-left:8px}
@media (max-width:900px){.qa{position:static;width:100%;margin:12px 0;max-height:400px}}
</style>
</head>
<body>
<header>
  <h1>本地文档统计 · 本地知识库 <span class="badge" id="srcBadge">…</span></h1>
  <p>扫描根: ~/Desktop/AI大模型 · ~/Downloads · ~/Documents · 自动增量更新(每小时) · 端口 :8907</p>
</header>
<div class="wrap">
  <div class="tabs">
    <div class="tab active" data-p="p1">文档统计</div>
    <div class="tab" data-p="p2">知识库问答</div>
  </div>

  <div class="panel show" id="p1">
    <div class="cards" id="cards1"></div>
    <div class="chart"><h3>文档类型分布 (按扩展名, 数量)</h3><div id="c1"></div></div>
    <div class="chart"><h3>目录分布 TOP10 (按文件数)</h3><div id="c2"></div></div>
    <div style="margin-top:14px;overflow-x:auto"><h3 style="font-size:14px;color:#444;margin:2px 4px 8px">最大的 15 个文档</h3>
      <table id="tbl"><thead><tr><th style="width:60%">文件</th><th>扩展名</th><th>大小</th><th>字符数</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <div class="panel" id="p2">
    <div class="cards" id="cards2"></div>
    <div class="prog" id="progWrap" style="display:none"><div id="prog"></div></div>
    <div class="chart" style="padding:8px" id="buildMsg" ></div>
    <div class="search">
      <input id="sq" placeholder="输入问题/关键词, 语义检索本地文档…" onkeydown="if(event.key==='Enter')doSearch()">
      <button class="btn" onclick="doSearch()">语义检索</button>
      <button class="btn ghost" onclick="updateInc()">增量更新</button>
      <button class="btn ghost" onclick="rebuild()">重建索引</button>
    </div>
    <div id="hits"></div>
  </div>
</div>

<div class="qa">
  <div class="head"><span>智能问答 (RAG)</span><span style="font-size:11px;opacity:.8">qwen-plus</span></div>
  <div class="body" id="qaBody"><div class="row"><div class="a">你好! 可问: 本地有多少文档 / RAG 是什么 / 迪士尼知识库讲了什么 / 怎么优化向量检索</div></div></div>
  <div class="in"><input id="qai" placeholder="输入问题, 回车发送" onkeydown="if(event.key==='Enter')ask()"><button onclick="ask()">发送</button></div>
</div>

<div class="footer">Local Docs Statistics &amp; Knowledge Base · 向量模型 text-embedding-v3 · 构建进度实时可见</div>

<script>
const fmt = n => (n == null ? '-' : Number(n).toLocaleString('zh-CN'));
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function j(u, o) { const r = await fetch(u, o); return r.json(); }
let KB = null, POLL = null;

/* ---- Tab ---- */
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('show'));
  t.classList.add('active');
  document.getElementById(t.dataset.p).classList.add('show');
  if (t.dataset.p === 'p2') refreshKB();
});

/* ---- Tab1 统计 ---- */
async function loadStats() {
  const d = await j('/api/docstats');
  document.getElementById('srcBadge').textContent = '扫描 ' + fmt(d.n_files) + ' 个文档';
  document.getElementById('cards1').innerHTML = [
    ['文档数量', fmt(d.n_files), '共 ' + (d.n_bytes / 1e6).toFixed(1) + ' MB'],
    ['总字符', fmt(d.n_chars), '含空格与符号'],
    ['中文字符', fmt(d.n_cn_chars), '占比 ' + (d.n_chars ? (d.n_cn_chars / d.n_chars * 100).toFixed(1) : 0) + '%'],
    ['总行数', fmt(d.n_lines), '词数 ' + fmt(d.n_words)],
  ].map(x => `<div class="card"><div class="t">${x[0]}</div><div class="v">${x[1]}</div><div class="s">${x[2]}</div></div>`).join('');

  const ext = (d.by_ext || []).slice(0, 10);
  chart('c1', {
    tooltip: { trigger: 'item', formatter: p => `${p.name}: ${fmt(p.value)} 个 (${p.percent}%)` },
    legend: { orient: 'vertical', right: 0, top: 'middle', textStyle: { fontSize: 11 } },
    series: [{
      type: 'pie', radius: ['38%', '68%'], center: ['38%', '50%'],
      data: ext.map(([n, v]) => ({ name: n, value: v })),
      label: { formatter: '{b} {c}', fontSize: 11 }
    }]
  });

  const dirs = (d.by_dir || []).slice(0, 10);
  chart('c2', {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 8, right: 30, top: 10, bottom: 10, containLabel: true },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: dirs.map(x => x[0] === '.' ? '(根目录)' : x[0]).reverse() },
    series: [{ type: 'bar', data: dirs.map(x => x[1]).reverse(), barMaxWidth: 22,
      itemStyle: { color: '#2a9d6e', borderRadius: [0, 4, 4, 0] },
      label: { show: true, position: 'right', fontSize: 11 } }]
  });

  document.querySelector('#tbl tbody').innerHTML = (d.top_files || []).map(f =>
    `<tr><td class="path">${esc(f.path)}</td><td>${esc(f.ext)}</td><td>${(f.size / 1024).toFixed(0)} KB</td><td>${fmt(f.chars)}</td></tr>`).join('');
}

/* ---- Tab2 知识库 ---- */
async function refreshKB() {
  if (POLL) { clearInterval(POLL); POLL = null; }
  const d = await j('/api/kb/status');
  KB = d;
  const st = d.index || {}, b = d.build || {};
  document.getElementById('cards2').innerHTML = [
    ['索引切片数', fmt(st.n_chunks || 0), st.n_files ? `覆盖 ${st.n_files} 个文件` : '尚未构建'],
    ['向量模型', st.model || '-', st.updated || ''],
    ['构建状态', b.state == 'done' ? '已完成' : (b.state == 'building' ? '构建中 ' + b.done + '/' + b.total : (b.state == 'scanning' ? '扫描中' : '待命')), b.msg || ''],
  ].map(x => `<div class="card"><div class="t">${x[0]}</div><div class="v" style="font-size:17px">${x[1]}</div><div class="s">${esc(x[2])}</div></div>`).join('');

  const pw = document.getElementById('progWrap');
  if (b.state === 'building' || b.state === 'scanning') {
    pw.style.display = '';
    const pct = b.total ? Math.round(b.done / b.total * 100) : 0;
    document.getElementById('prog').style.width = pct + '%';
    document.getElementById('buildMsg').innerHTML = `<span style="font-size:13px;color:#0f5c3e">⏳ 正在构建索引: ${b.done}/${b.total} 文件 (${pct}%) · ${esc(b.msg || '')}</span>`;
    POLL = setInterval(() => { fetch('/api/kb/status').then(r => r.json()).then(x => {
      if (!(x.build && (x.build.state === 'building' || x.build.state === 'scanning'))) { refreshKB(); }
      else if (x.build.done !== b.done) refreshKB();
    }); }, 3000);
  } else {
    pw.style.display = 'none';
    const skipN = (b.msg.match(/\d+/) || [null])[0];
    document.getElementById('buildMsg').innerHTML = st.n_chunks ?
      `<span style="font-size:13px;color:#666">✓ 索引就绪: ${st.n_chunks} 个切片 / ${st.n_files} 个文件, 更新于 ${st.updated}</span>` +
      (/跳过\s*\d+/.test(b.msg || '') && skipN && +skipN > 5 ?
        `<div style="margin-top:6px;font-size:13px;color:#b45309">⚠ ${b.msg} — 多为 DashScope 账户欠费(Arrearage), 充值后点「重建索引」可补齐</div>` : '') :
      `<span style="font-size:13px;color:#b45309">尚未构建索引 — 点击右上「重建索引」开始(首次约需数十分钟, 之后增量更新)</span>`;
  }
}

async function doSearch() {
  const q = document.getElementById('sq').value.trim();
  if (!q) return;
  document.getElementById('hits').innerHTML = '<div style="color:#888;padding:8px">检索中…</div>';
  const d = await j('/api/kb/search?q=' + encodeURIComponent(q) + '&k=5');
  const hits = d.hits || [];
  if (!hits.length) { document.getElementById('hits').innerHTML = '<div style="color:#b45309;padding:8px">无匹配结果(索引未构建或未配置 API key)</div>'; return; }
  const methodBadge = hits[0].method === 'vector'
    ? '<span style="font-size:11px;color:#0f5c3e;background:#d1fae5;padding:2px 8px;border-radius:10px;margin-left:8px">向量检索</span>'
    : '<span style="font-size:11px;color:#b45309;background:#fef3c7;padding:2px 8px;border-radius:10px;margin-left:8px">关键词降级(向量不可用)</span>';
  const qn = q.toLowerCase();
  document.getElementById('hits').innerHTML = methodBadge + hits.map(h => {
    let body = esc(h.text);
    const i = body.toLowerCase().indexOf(qn);
    if (i >= 0) body = body.slice(0, i) + '<mark>' + body.slice(i, i + qn.length) + '</mark>' + body.slice(i + qn.length);
    return `<div class="hit"><div class="h">《${esc(h.title)}》<span style="font-weight:400;font-size:11px;color:#999;margin-left:8px">相似度 ${h.score}</span></div>
      <div class="m path">${esc(h.src)}</div><div class="b">${body.slice(0, 400)}…</div></div>`;
  }).join('');
}

async function rebuild() {
  const r = await j('/api/kb/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"force":true}' });
  if (!r.ok) { alert(r.error || '构建失败'); return; }
  refreshKB();
}

async function updateInc() {
  const r = await j('/api/kb/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"force":false}' });
  if (!r.ok) { alert(r.error || '更新失败'); return; }
  refreshKB();
}

/* ---- 问答 ---- */
async function ask() {
  const inp = document.getElementById('qai');
  const q = inp.value.trim();
  if (!q) return;
  inp.value = '';
  const body = document.getElementById('qaBody');
  body.insertAdjacentHTML('beforeend', `<div class="row"><div class="q">${esc(q)}</div><div class="a">思考中…</div></div>`);
  body.scrollTop = body.scrollHeight;
  try {
    const r = await j('/api/qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q, mode: 'rag' }) });
    body.lastElementChild.querySelector('.a').textContent = r.answer || '(无回复)';
  } catch (e) { body.lastElementChild.querySelector('.a').textContent = '请求失败: ' + e; }
  body.scrollTop = body.scrollHeight;
}

function chart(id, opt) {
  const el = document.getElementById(id);
  if (!el) return;
  const c = echarts.getInstanceByDom(el) || echarts.init(el);
  c.setOption(opt, true);
}

loadStats(); refreshKB();
setInterval(loadStats, 5 * 60 * 1000);
window.addEventListener('resize', () => ['c1', 'c2'].forEach(i => { const el = document.getElementById(i); if (el) echarts.getInstanceByDom(el)?.resize(); }));
</script>
</body>
</html>"""


# ---------------------------------------------------------------- main
def _start_build_if_needed():
    """启动时若索引缺失或已过期, 后台构建。"""
    st = doc_kb.stats()
    if not st.get("ok"):
        t = threading.Thread(target=doc_kb.update, kwargs={"force": False}, daemon=True)
        t.start()
        print("[kb] 索引缺失, 已在后台构建...")


def main():
    if "--port" in sys.argv:
        global PORT
        PORT = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"[doc-kb] 启动 http://127.0.0.1:{PORT}  (llm={'ready' if _load_key() else 'NO-KEY'})")
    _start_build_if_needed()
    threading.Thread(target=_auto_update_loop, kwargs={"interval_minutes": 60}, daemon=True).start()
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()