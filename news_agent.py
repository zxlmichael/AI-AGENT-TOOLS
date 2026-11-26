#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时事新闻助手：国内外最新时事新闻抓取 + 总结概述 + ECharts 可视化看板

- 数据层 news_data.py：多源 RSS（BBC中文/BBC World/纽约时报中文/Google News 中文）
- 助手层：规则问答（秒回）+ 可选 LLM 概述（LLM_BASE_URL 指向本地 Ollama 或任意 OpenAI 兼容端点）
- HTTP :8915：/（看板+问答页）、/api/news、/api/qa、/health

用法：
    .venv/bin/python news_agent.py "总结一下最近的国内外大事"
    .venv/bin/python news_agent.py --chat
    .venv/bin/python news_agent.py --serve --port 8915
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime

from flask import Flask, Response, jsonify, request, send_from_directory

import news_data as nd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "news_assets")
ECHARTS_URL = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"

# LLM 概述：默认本地 Ollama（无需任何云密钥）；可用环境变量切到任意 OpenAI 兼容端点
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_KEY = os.environ.get("LLM_API_KEY", "ollama")
LLM_TOPIC = os.environ.get("NEWS_LLM_MODEL", LLM_MODEL)


def _ensure_echarts() -> str:
    """确保 echarts.min.js 本地可用，返回页面用 src（本地路径或 CDN）。"""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    local = os.path.join(ASSETS_DIR, "echarts.min.js")
    if os.path.exists(local):
        return "/assets/echarts.min.js"
    try:
        ctx = ssl.create_default_context(cafile=__import__("certifi").where())
        req = urllib.request.Request(ECHARTS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            data = r.read()
        with open(local, "wb") as f:
            f.write(data)
        return "/assets/echarts.min.js"
    except Exception:
        return ECHARTS_URL


def _llm_ok() -> bool:
    try:
        req = urllib.request.Request(f"{LLM_BASE}/models",
                                     headers={"Authorization": f"Bearer {LLM_KEY}"})
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def llm_overview(stats: dict, items: list[dict]) -> str | None:
    """让 LLM 基于统计数据+头条生成一段新闻概述；失败返回 None（调用方兜底）。

    默认本地 Ollama：原生 /api/chat 可关思考（qwen3 系在 OpenAI 兼容接口上
    content 会全进 reasoning 字段导致正文为空），失败回退 OpenAI 兼容端点。
    """
    headlines = "\n".join(
        f"- [{it['region']}/{it['topic']}] {it['title']}" for it in items[:25])
    system = ("你是时事新闻编辑。用简体中文写一段 150-250 字的本期新闻概述，"
              "结构：①总体态势一句 ②国内要点 2-3 句 ③国际要点 2-3 句 "
              "④值得关注的趋势。数据必须来自给定材料，不得编造事件。")
    user = (f"本期共 {stats['total']} 条新闻（12h 内 {stats['fresh']} 条），"
            f"地区分布：{[(x['name'], x['count']) for x in stats['by_region']]}；"
            f"主题分布：{[(x['name'], x['count']) for x in stats['by_topic'] if x['count']]}；"
            f"热搜词：{[(x['name'], x['count']) for x in stats['top_keywords'][:8]]}。\n"
            f"头条如下：\n{headlines}")
    # 路径①：ollama 原生 API（think:false 关闭思考，正文必出）
    if LLM_BASE.rstrip("/").endswith(":11434/v1") or "localhost:11434" in LLM_BASE:
        try:
            req = urllib.request.Request(
                LLM_BASE.rstrip("/").replace("/v1", "/api/chat"),
                data=json.dumps({"model": LLM_TOPIC, "stream": False, "think": False,
                                 "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": user}],
                                 "options": {"temperature": 0.4, "num_predict": 500}}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read().decode())
            txt = ((out.get("message") or {}).get("content") or "").strip()
            if txt:
                return txt
        except Exception:
            pass
    # 路径②：OpenAI 兼容端点（qwen 系思考模型 content 可能为空，用 reasoning 兜底）
    try:
        req = urllib.request.Request(f"{LLM_BASE}/chat/completions",
                                     data=json.dumps({
                                         "model": LLM_TOPIC,
                                         "messages": [{"role": "system", "content": system},
                                                      {"role": "user", "content": user}],
                                         "temperature": 0.4, "max_tokens": 500}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {LLM_KEY}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode())
        msg = out["choices"][0]["message"]
        txt = (msg.get("content") or msg.get("reasoning") or "").strip()
        return txt or None
    except Exception:
        return None


def local_overview(stats: dict, items: list[dict]) -> str:
    """无 LLM 时的本地格式化概述（数字与要点永远可用）。"""
    reg = "、".join(f"{x['name']} {x['count']} 条" for x in stats["by_region"])
    top = "、".join(x["name"] for x in stats["by_topic"] if x["count"] >= 3)
    kw = "、".join(f"{x['name']}({x['count']})" for x in stats["top_keywords"][:6])
    heads = "\n".join(f"  {i+1}. {it['title']}" for i, it in enumerate(items[:8]))
    return (f"本期共收录 {stats['total']} 条时事新闻（近 12 小时 {stats['fresh']} 条），"
            f"地区分布：{reg}。\n热度主题：{top or '综合'}。\n热搜词：{kw}。\n\n头条速览：\n{heads}")


def answer(q: str) -> str:
    """规则 + LLM 问答：总结/概述 → LLM 概述（失败本地兜底）；特指领域 → 规则直答。"""
    force = "刷新" in q or "更新" in q or "重新" in q
    items, errors = nd.cached_items(force=force)
    if not items:
        return f"新闻源暂不可用：{errors or '无数据'}。稍后重试。"
    stats = nd.stats(items)

    if any(k in q for k in ("总结", "概述", "概览", "综述", "大事", "盘点", "汇总", "报告")):
        # 优先 LLM 概述（带摘要时把错误信息附后）
        txt = llm_overview(stats, items) if _llm_ok() else None
        body = txt or local_overview(stats, items)
        if errors:
            body += f"\n\n（{len(errors)} 个源抓取失败：{'；'.join(errors)}）"
        return body

    if "国内" in q:
        sub = [it for it in items if it["region"] == "国内"][:12]
        return f"国内要闻（{len(sub)} 条）：\n" + "\n".join(
            f"- [{it['topic']}] {it['title']}" for it in sub)
    if "国际" in q:
        sub = [it for it in items if it["region"] == "国际"][:12]
        return f"国际要闻（{len(sub)} 条）：\n" + "\n".join(
            f"- [{it['topic']}] {it['title']}" for it in sub)
    if "科技" in q or "AI" in q:
        sub = [it for it in items if it["topic"] == "科技"][:12]
        return f"科技要闻（{len(sub)} 条）：\n" + "\n".join(
            f"- [{it['source']}] {it['title']}" for it in sub)
    if "经济" in q:
        sub = [it for it in items if it["topic"] == "经济"][:12]
        return f"财经要闻（{len(sub)} 条）：\n" + "\n".join(
            f"- [{it['source']}] {it['title']}" for it in sub)
    if "头条" in q or "最新" in q or "要闻" in q or "热点" in q or "新闻" in q or \
            not any(k in q for k in ("怎么", "为什么")):
        return f"最新时事新闻（{len(items)} 条收录，最新 {items[0]['time'][:16]} UTC）：\n" + \
               "\n".join(f"- [{it['region']}/{it['topic']}] {it['title']}" for it in items[:15])
    return local_overview(stats, items)


# ---------------- HTTP 服务 ----------------
def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "news-assistant",
                        "llm": "ready" if _llm_ok() else "missing",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.route("/api/news", methods=["GET"])
    def api_news():
        force = request.args.get("refresh") == "1"
        limit = int(request.args.get("limit", 30))
        items, errors = nd.cached_items(force=force)
        return jsonify({"ok": True, "stats": nd.stats(items),
                        "items": nd.recent(items, limit), "errors": errors,
                        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")})

    @app.route("/api/qa", methods=["GET", "POST"])
    def api_qa():
        if request.method == "POST":
            q = (request.get_json(silent=True) or {}).get("question", "")
        else:
            q = request.args.get("question", "")
        q = str(q).strip()
        if not q:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        started = time.time()
        try:
            ans = answer(q)
        except Exception as exc:
            # 兜底：LLM 或本地格式化任何异常都不让 /api/qa 500
            try:
                items, errors = nd.cached_items()
                ans = local_overview(nd.stats(items), items) + \
                      f"\n\n（生成异常已降级：{type(exc).__name__}）"
            except Exception:
                return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 502
        return jsonify({"ok": True, "question": q, "answer": ans,
                        "elapsed_seconds": round(time.time() - started, 2)})

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(ASSETS_DIR, name)

    @app.get("/")
    def index():
        src = _ensure_echarts()
        return Response(PAGE.replace("__ECHARTS__", src), mimetype="text/html")

    return app


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>时事新闻助手 · 国内外要闻总结与可视化</title>
<script src="__ECHARTS__"></script>
<style>
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f6f7f9;color:#222}
.wrap{max-width:1060px;margin:0 auto;padding:22px 16px 40px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#888;font-size:13px;margin-bottom:16px}
.card{background:#fff;border-radius:12px;padding:18px 20px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card h2{margin:0 0 10px;font-size:15px}
input,button{font-size:15px;padding:10px 14px;border-radius:8px;border:1px solid #d5d9e0}
input{flex:1;margin-right:10px;outline:none}
button{background:#2563eb;color:#fff;border:none;cursor:pointer}
#out{white-space:pre-wrap;background:#f8fafc;border:1px solid #eef1f5;border-radius:8px;padding:12px;font-size:14px;margin-top:10px;display:none}
.row{display:flex}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
.chart{height:320px}
.news{font-size:14px;line-height:1.8}
.news a{color:#2563eb;text-decoration:none}
.news .tag{background:#eef4ff;color:#2563eb;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:6px}
</style></head>
<body><div class="wrap">
<h1>时事新闻助手 · 国内外要闻</h1>
<div class="sub">多源公开 RSS（BBC中文 / BBC World / 纽约时报中文 / Google News中文）· 本地统计 · 摘要由本地 LLM 生成</div>
<div class="card"><h2>新闻总结问答</h2>
<div class="row"><input id="q" placeholder="如：总结一下最近的国内外大事；国际要闻；科技热点">
<button onclick="ask()">问</button></div>
<div id="out"></div></div>
<div class="grid">
<div class="card"><h2>地区分布</h2><div id="regionChart" class="chart"></div></div>
<div class="card"><h2>主题分布</h2><div id="topicChart" class="chart"></div></div>
</div>
<div class="grid">
<div class="card"><h2>热搜词 TOP</h2><div id="kwChart" class="chart"></div></div>
<div class="card"><h2>24 小时发文热度</h2><div id="hourChart" class="chart"></div></div>
</div>
<div class="card"><h2>最新头条</h2><div class="news" id="news"></div></div>
</div>
<script>
let DATA_STATS=null;
function ask(){const q=document.getElementById('q').value.trim();if(!q)return;
 const o=document.getElementById('out');o.style.display='block';o.textContent='生成中…';
 fetch('/api/qa',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})
 .then(r=>r.json()).then(d=>{o.textContent=d.ok?d.answer:('错误：'+(d.error||''))})
 .catch(e=>{o.textContent='失败：'+e});}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}
function load(){
 fetch('/api/news?limit=25').then(r=>r.json()).then(d=>{
  if(!d.ok)return; const s=d.stats; DATA_STATS=s;
  document.querySelector('.sub').textContent=
   '共收录 '+s.total+' 条（近12h '+s.fresh+' 条）· BBC中文/BBC World/纽时中文/Google News中文 · 本地统计';
  echarts.init(document.getElementById('regionChart')).setOption({
   tooltip:{trigger:'item'},legend:{bottom:0},
   series:[{type:'pie',radius:['38%','62%'],data:s.by_region,
    itemStyle:{borderColor:'#fff',borderWidth:2}}]});
  echarts.init(document.getElementById('topicChart')).setOption({
   tooltip:{trigger:'axis'},grid:{left:40,right:16,bottom:30,top:20},
   xAxis:{type:'category',data:s.by_topic.map(x=>x.name)},
   yAxis:{type:'value'},series:[{type:'bar',data:s.by_topic.map(x=>x.count),
    itemStyle:{color:'#2563eb'},label:{show:true,position:'top',fontSize:10}}]});
  echarts.init(document.getElementById('kwChart')).setOption({
   tooltip:{trigger:'axis'},grid:{left:90,right:30,bottom:20,top:20},
   xAxis:{type:'value'},yAxis:{type:'category',data:s.top_keywords.map(x=>x.name).reverse()},
   series:[{type:'bar',data:s.top_keywords.map(x=>x.count).reverse(),
    itemStyle:{color:'#0ea5e9'},barWidth:14}]});
  echarts.init(document.getElementById('hourChart')).setOption({
   tooltip:{trigger:'axis'},grid:{left:40,right:16,bottom:30,top:20},
   xAxis:{type:'category',data:s.per_hour.map(x=>x.name)},
   yAxis:{type:'value'},series:[{type:'line',data:s.per_hour.map(x=>x.count),
    smooth:true,areaStyle:{opacity:.15},itemStyle:{color:'#16a34a'}}]});
  document.getElementById('news').innerHTML=(d.items||[]).map(it=>
   '<div><span class="tag">'+esc(it.region)+'/'+esc(it.topic)+'</span>'+
   '<a href="'+esc(it.link)+'" target="_blank">'+esc(it.title)+'</a>'+
   '<span style="color:#999;font-size:12px"> '+esc(it.source)+'</span></div>').join('');
 }).catch(()=>{document.querySelector('.sub').textContent='数据加载失败';});
}
load();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="时事新闻助手（国内外要闻总结+可视化）")
    ap.add_argument("question", nargs="?", default="", help="问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8915)
    args = ap.parse_args()

    if args.serve:
        app = create_app()
        print(f"时事新闻助手 HTTP 服务：http://{args.host}:{args.port}")
        print("  页面: /    数据: /api/news    问答: POST /api/qa   /health")
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if args.chat or not args.question:
        print("输入问题（quit 退出）：")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print(answer(q), "\n")
        return

    print(answer(args.question))


if __name__ == "__main__":
    main()