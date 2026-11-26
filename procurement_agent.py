#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""政府招采地图需求监测助手（:8916）

- 数据层 procurement_data.py：多源抓取 → SQLite 增量入库（30 分钟一轮后台线程）
- 一问一答：规则秒回；含"深挖/分析/趋势"等问题自动升级 OpenManus（一次性子进程）
- 导出：项目清单 CSV（utf-8-sig）+ 问答/统计报告 Markdown
- 看板：ECharts 4+1 图，60s 轮询实时刷新

用法:
  python procurement_agent.py --serve --port 8916   # HTTP 服务 + 看板
  python procurement_agent.py "最近有哪些地图项目"    # CLI 一问一答
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

import procurement_data as pd

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "procurement_assets"
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"
PORT = 8916
SERVICE = "procurement-monitor"
_REFETCH_TTL = 1800          # 抓取间隔（与数据层缓存一致）

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ---------- 后台抓取线程 ----------
_snap = {"lock": threading.Lock(), "t": 0.0}


def _refetch_loop() -> None:
    """常驻线程：每 30 分钟触发一轮抓取（数据层内部还有 TTL 缓存兜底）。"""
    while True:
        try:
            meta = pd.fetch()
            app.logger.info("refetch: added=%s total=%s errors=%s",
                            meta["added"], meta["total"], len(meta["errors"]))
        except Exception as exc:
            app.logger.error("refetch error: %s", exc)
        time.sleep(_REFETCH_TTL)


def _ensure_snapshot() -> dict:
    """服务启动后尽快触发首轮抓取（不阻塞请求；届时返回上次状态）。"""
    with _snap["lock"]:
        if time.time() - _snap["t"] > _REFETCH_TTL:
            _snap["t"] = time.time()
            t = threading.Thread(target=_refetch_loop, daemon=True)
            t.start()
    return dict(_snap)


# ---------- echarts 本地化 ----------

def _ensure_echarts() -> str:
    ASSETS_DIR.mkdir(exist_ok=True)
    local = ASSETS_DIR / "echarts.min.js"
    if local.exists() and local.stat().st_size > 100000:
        return "/assets/echarts.min.js"
    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(ECHARTS_CDN, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60, context=ctx).read()
        local.write_bytes(data)
        return "/assets/echarts.min.js" if local.stat().st_size > 100000 else ECHARTS_CDN
    except Exception:
        return ECHARTS_CDN


# ---------- OpenManus 深挖 ----------

_OPENMANUS_ROOT = Path.home() / "OpenManus"
_OPENMANUS_PY = _OPENMANUS_ROOT / ".venv" / "bin" / "python"
_RUNNER = _OPENMANUS_ROOT / "oneshot_runner.py"
_DEEP = {"lock": threading.Lock(), "t": 0.0, "q": "", "answer": "", "engine": "local"}


def openmanus_available() -> bool:
    if os.environ.get("OPENMANUS_DISABLE") == "1":
        return False
    return _RUNNER.exists() and _OPENMANUS_PY.exists()


def _deep_context() -> str:
    return pd.deepdive_input(30)


def _openmanus_deepdive(question: str) -> dict:
    """子进程调用 oneshot_runner.py；stdout 末行 JSON。"""
    prompt = (
        "你是政府招采地图需求分析专家。以下是近 30 天收集的招标/采购意向项目简述（可能不完整）：\n"
        f"{_deep_context()}\n\n"
        f"用户问题：{question}\n"
        "请结合项目数据，按【总体态势/重点地区/项目类型与预算/值得关注的线索】结构化回答；"
        "数据不足时明确说明，不要编造具体项目。中文，正文 300-800 字。"
    )
    try:
        cp = subprocess.run(
            [_OPENMANUS_PY, str(_RUNNER),
             "--prompt", prompt, "--timeout", "240", "--max-steps", "8"],
            cwd=str(_OPENMANUS_ROOT), text=True,
            capture_output=True, timeout=270, stdin=subprocess.DEVNULL,
            env={**os.environ, "OPENMANUS_DISABLE_BROWSER_USE": "1"},
        )
        if cp.returncode != 0:
            return {"engine": "local", "answer": pd.answer_local(question)}
        out = cp.stdout.strip().splitlines()
        if not out:
            return {"engine": "local", "answer": pd.answer_local(question)}
        result = json.loads(out[-1])
        if result.get("status") != "ok" or not result.get("answer"):
            return {"engine": "local", "answer": pd.answer_local(question)}
        return {"engine": "openmanus", "answer": result["answer"]}
    except Exception:
        return {"engine": "local", "answer": pd.answer_local(question)}


def deepdive(question: str) -> dict:
    """深挖（30 分钟内同问题命中缓存）。"""
    with _DEEP["lock"]:
        if question == _DEEP["q"] and time.time() - _DEEP["t"] < 1800 and _DEEP["answer"]:
            return {"engine": _DEEP["engine"], "answer": _DEEP["answer"], "cached": True}
    if openmanus_available():
        got = _openmanus_deepdive(question)
    else:
        got = {"engine": "local", "answer": pd.answer_local(question)}
    with _DEEP["lock"]:
        _DEEP.update({"t": time.time(), "q": question, "answer": got["answer"],
                      "engine": got["engine"]})
    return got


# ---------- 规则问答 ----------

_DEEP_HINTS = ("深挖", "分析", "解读", "趋势", "洞察", "策略", "价值", "机会")


def answer(q: str) -> dict:
    stat = pd.stats()
    conn = pd.connected()
    try:
        # 深挖类问题 → OpenManus
        if any(h in q for h in _DEEP_HINTS):
            got = deepdive(q)
            return {"question": q, "engine": got["engine"], "answer": got["answer"]}
        # 统计类
        if any(k in q for k in ("多少", "几个", "统计", "总数", "总共")):
            txt = (f"当前库内共 **{stat['total']}** 条近 30 天地图需求项目。\n\n"
                   f"- 地区分布：" + "，".join(f"{r['name']} {r['count']} 条"
                                            for r in stat["by_region"][:6]) + "\n"
                   + "".join(f"- {t['name']}: {t['count']} 条\n" for t in stat["by_type"]))
            return {"question": q, "engine": "rule", "answer": txt}
        # 地区项目
        region = next((r for r in pd._REGIONS if r in q), None)
        if region:
            rows = pd.projects(region=region, limit=10)
            txt = f"### {region} 相关地图需求项目（近 30 天）\n" if rows else f"{region} 暂无条目\n"
            for r in rows:
                b = f"{r['budget']:.1f}万" if r["budget"] else "预算未知"
                txt += f"- {r['pub_date'] or ''} {r['title']}（{b}）\n"
            return {"question": q, "engine": "rule", "answer": txt}
        # 预算/大单
        if any(k in q for k in ("预算", "金额", "大单", "多少钱")):
            rows = sorted(pd.projects(limit=500), key=lambda x: x["budget"] or 0, reverse=True)[:10]
            txt = "### 预算 Top10 地图需求项目\n"
            for r in rows:
                if not r["budget"]:
                    continue
                txt += (f"- {r['pub_date'] or ''} {r['title']} "
                        f"（{r['region'] or '?'}，**{r['budget']:.1f} 万元**）\n")
            return {"question": q, "engine": "rule", "answer": txt}
        # 关键词过滤（无命中则回落到默认概览，避免误导性"共 0 条"）
        kw = next((k for k in pd.KEYWORDS if k.lower() in q.lower()), None)
        if kw:
            rows = [r for r in pd.projects(limit=500)
                    if kw in r["title"] or kw in ",".join(r["keywords"])][:10]
            if rows:
                txt = f"### 含「{kw}」的项目（近 30 天，共 {len(rows) if len(rows)<10 else '10+'} 条）\n"
                for r in rows[:10]:
                    b = f"{r['budget']:.1f}万" if r["budget"] else "预算未知"
                    txt += f"- {r['pub_date'] or ''} {r['title']}（{r['region'] or '?'}，{b}）\n"
                return {"question": q, "engine": "rule", "answer": txt}
        # 默认：概览 + 最新（answer_local 已含总量/地区/类型/关键词/最新5条）
        return {"question": q, "engine": "rule", "answer": pd.answer_local(q)}
    finally:
        conn.close()


# ---------- 导出 ----------

def _build_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["标题", "来源", "地区", "公告类型", "发布日期", "预算(万元)",
                "采购人", "命中关键词", "链接"])
    for r in pd.projects(limit=5000):
        w.writerow([r["title"], r["source"], r["region"], r["pub_type"],
                    r["pub_date"] or "", r["budget"] or "",
                    r["buyer"] or "", "、".join(r["keywords"]), r["link"]])
    return "\ufeff" + buf.getvalue()          # BOM：Excel 打开中文不乱码


def _build_report(question: str) -> str:
    if question:
        got = answer(question)
        body = got["answer"]
    else:
        body = pd.answer_local("概览")
    stat = pd.stats()
    lines = [
        "# 招采地图需求监测 · 问答报告",
        f"\n- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 数据抓取：{stat['fetched_at'] or '—'}",
        f"- 库内项目：{stat['total']} 条（近 30 天窗口）",
        f"- 数据源：中国政府采购网 / 北京公共资源交易平台"
        + ("（上海/广东适配中）" if any("适配中" in s["note"] for s in stat["source_status"]) else ""),
        "",
        f"## 问题：{question or '（概览）'}",
        "",
        body,
        "",
        "---",
        "*数据来源为公开招采信息，仅供学习参考，请以官方公告为准。*",
    ]
    return "\n".join(lines)


# ---------- 路由 ----------

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": SERVICE,
                    "total": pd.stats()["total"],
                    "time": datetime.now().isoformat(timespec="seconds")})


@app.get("/api/stats")
def api_stats():
    return jsonify({"ok": True, **pd.stats()})


@app.get("/api/projects")
def api_projects():
    q = request.args.get("q", "")
    region = request.args.get("region", "")
    limit = min(int(request.args.get("limit", 200)), 1000)
    try:
        items = pd.projects(q=q or None, region=region or None, limit=limit)
    except Exception:
        items = []
    return jsonify({"ok": True, "items": items})


@app.route("/api/qa", methods=["GET", "POST"])
def api_qa():
    q = (request.args.get("question") if request.method == "GET"
         else (request.json or {}).get("question", ""))
    if not q:
        return jsonify({"ok": False, "error": "缺少 question"}), 400
    t0 = time.time()
    try:
        got = answer(q)
    except Exception as exc:
        got = {"question": q, "engine": "fallback", "answer": pd.answer_local(q)}
    got["elapsed_seconds"] = round(time.time() - t0, 1)
    return jsonify({"ok": True, **got})


@app.post("/api/deepdive")
def api_deepdive():
    q = (request.json or {}).get("question", "")
    if not q:
        return jsonify({"ok": False, "error": "缺少 question"}), 400
    t0 = time.time()
    got = deepdive(q)
    got["elapsed_seconds"] = round(time.time() - t0, 1)
    return jsonify({"ok": True, **got})


@app.get("/api/export/projects.csv")
def export_csv():
    data = _build_csv()
    name = f"procurement_projects_{datetime.now().strftime('%Y%m%d')}.csv"
    return Response(data, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@app.route("/api/export/report.md", methods=["GET", "POST"])
def export_report():
    q = (request.args.get("question") if request.method == "GET"
         else (request.json or {}).get("question", ""))
    data = _build_report(q)
    name = f"procurement_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    return Response(data, mimetype="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@app.get("/assets/<path:name>")
def assets(name):
    return send_from_directory(ASSETS_DIR, name)


@app.get("/")
def index():
    html = PAGE.replace("__ECHARTS__", _ensure_echarts()).replace("__OPTS__", _ECHARTS_OPTS)
    return Response(html, mimetype="text/html")


# ---------- 看板页面 ----------

_ECHARTS_OPTS = """
function load(){
  fetch('/api/stats').then(r=>r.json()).then(s=>{
    document.getElementById('stats').textContent =
      '项目 '+s.total+' 条 · 抓取 '+ (s.fetched_at||'—').slice(0,16);
    const es=(s.errors||[]).length;
    document.getElementById('srcs').textContent =
      (s.source_status||[]).map(x=>x.name+(x.note==='适配中'?'(适配中)':'')).join(' · ')
      + (es? ' · 源错误 '+es : '');
    reg.setOption(VM(s.by_region.slice(0,8).map(x=>({name:x.name,value:x.count}))));
    typ.setOption(VM(s.by_type.map(x=>({name:x.name,value:x.count}))));
    kw.setOption({yAxis:{data:s.top_keywords.slice(0,12).map(x=>x.name).reverse()},
                  series:[{data:s.top_keywords.slice(0,12).map(x=>x.count).reverse()}]});
    day.setOption({xAxis:{data:s.per_day.map(x=>x.name.slice(5))},
                   series:[{data:s.per_day.map(x=>x.count)}]});
    if(s.budget_range){
      buf.setOption(VM(s.budget_range.map(x=>({name:x.name,value:x.count}))));
      document.getElementById('budget_card').style.display='';
    }else{document.getElementById('budget_card').style.display='none';}
  }).catch(()=>{});
  fetch('/api/projects?limit=25').then(r=>r.json()).then(d=>{
    document.getElementById('list').innerHTML = (d.items||[]).map(p=>
      '<li><a href="'+esc(p.link)+'" target="_blank">'+esc(p.title)+'</a>'
      +'<span>'+esc(p.region||'')+' · '+esc(p.pub_date||'')+' · '+(p.budget?esc(p.budget+'万'):'')+'</span></li>'
    ).join('')||'<li>暂无数据（首轮抓取中…）</li>';
  }).catch(()=>{});
}
function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const out=document.getElementById('out');
  out.innerHTML='<div class="loading">分析中（深挖问题约 1-4 分钟）…</div>';
  fetch('/api/qa',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})}).then(r=>r.json()).then(d=>{
    out.innerHTML='<div class="a">'+esc(d.answer).replace(/\\n/g,'<br>')+'</div>'
      +'<a class="dl" href="/api/export/report.md?question='+encodeURIComponent(q)+'">导出本问答 (MD)</a>';
  }).catch(e=>{out.innerHTML='<div class="a">出错：'+esc(e)+'</div>'});
}
function dlCsv(){location.href='/api/export/projects.csv'}
function dlMd(){location.href='/api/export/report.md'}
setInterval(load, 60000);
"""

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>政府招采 · 地图需求监测</title>
<script src="__ECHARTS__"></script>
<style>
body{font-family:-apple-system,'PingFang SC',sans-serif;margin:0;background:#f5f7fa;color:#222}
header{background:#1f4e79;color:#fff;padding:14px 22px;display:flex;justify-content:space-between;align-items:center}
header b{font-size:19px}header .meta{font-size:12px;opacity:.85;text-align:right}
.wrap{max-width:1180px;margin:18px auto;padding:0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.card{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.card h3{margin:0 0 10px;font-size:14px;color:#1f4e79}
.chart{height:260px}
#budget_card{display:none}
.qa{background:#fff;border-radius:10px;padding:14px;margin-top:14px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.qa input{width:70%;padding:9px 12px;border:1px solid #cbd5e1;border-radius:6px;font-size:14px}
.qa button{background:#1f4e79;color:#fff;border:0;border-radius:6px;padding:9px 18px;margin-left:8px;cursor:pointer}
.qa .dl{display:inline-block;margin-top:10px;font-size:13px;color:#1f4e79}
#out{margin-top:10px;white-space:pre-wrap;background:#f8fafc;border-radius:8px;padding:12px;font-size:14px;line-height:1.7}
.loading{color:#64748b}
.btn{background:#fff;border:1px solid #1f4e79;color:#1f4e79;border-radius:6px;padding:7px 14px;margin-left:8px;cursor:pointer;font-size:13px}
#list{list-style:none;padding:0;margin:0;max-height:420px;overflow:auto;font-size:13px}
#list li{padding:7px 4px;border-bottom:1px dashed #e2e8f0;display:flex;justify-content:space-between;gap:10px}
#list li a{color:#1f4e79;text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%}
#list li span{color:#64748b;white-space:nowrap}
</style></head><body>
<header><b>政府招采 · 地图需求监测（近 30 天）</b>
<div class="meta"><div id="stats"></div><div id="srcs" style="opacity:.8"></div>
<button class="btn" onclick="dlCsv()">导出项目清单(CSV)</button>
<button class="btn" onclick="dlMd()">导出概览报告(MD)</button></div></header>
<div class="wrap">
<div class="cards">
<div class="card"><h3>地区分布</h3><div class="chart" id="reg"></div></div>
<div class="card"><h3>项目类型</h3><div class="chart" id="typ"></div></div>
<div class="card" id="budget_card"><h3>预算区间（万元）</h3><div class="chart" id="buf"></div></div>
<div class="card"><h3>热搜关键词</h3><div class="chart" id="kw"></div></div>
<div class="card" style="grid-column:span 1"><h3>近 30 天趋势</h3><div class="chart" id="day"></div></div>
<div class="card"><h3>最新项目</h3><ul id="list"></ul></div>
</div>
<div class="qa"><b>问一问：</b>
<input id="q" placeholder="例如：最近有哪些地图相关的招标项目？/ 分析一下市场趋势" onkeydown="if(event.key==='Enter')ask()">
<button onclick="ask()">提问</button><div id="out"></div>
</div></div>
<script>
__OPTS__
const esc=s=>{const d=document.createElement('div');d.textContent=s||'';return d.innerHTML};
const VM=v=>({tooltip:{},legend:{},series:[{type:'pie',radius:['35%','65%'],data:v,label:{fontSize:11}}]});
let reg=echarts.init(document.getElementById('reg')),
    typ=echarts.init(document.getElementById('typ')),
    buf=echarts.init(document.getElementById('buf'));
let kw=echarts.init(document.getElementById('kw')),
    day=echarts.init(document.getElementById('day'));
kw.setOption({grid:{left:120,right:30},xAxis:{type:'value'},yAxis:{type:'category'},
  series:[{type:'bar',itemStyle:{color:'#2f7ec2'}}]});
day.setOption({tooltip:{trigger:'axis'},xAxis:{type:'category'},yAxis:{type:'value'},
  series:[{type:'line',smooth:true,areaStyle:{opacity:.2},itemStyle:{color:'#2f7ec2'}}]});
window.addEventListener('resize',()=>[reg,typ,buf,kw,day].forEach(c=>c.resize()));
load();
</script></body></html>"""


# ---------- CLI ----------

def _chat() -> None:
    print("招采地图需求监测问答（输入 exit 退出）")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        got = answer(q)
        print(f"\n[{got['engine']}]\n{got['answer']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="政府招采地图需求监测助手")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    if args.serve:
        _ensure_snapshot()
        _ensure_echarts()
        print(f"招采监测助手 :{args.port}")
        print(f"  看板  http://{args.host}:{args.port}/")
        print(f"  问答  POST /api/qa  (深挖含'分析/趋势'自动升级 OpenManus)")
        print(f"  导出  /api/export/projects.csv · /api/export/report.md")
        app.run(host=args.host, port=args.port, threaded=True)
    elif args.question:
        print(answer(args.question)["answer"])
    elif args.chat:
        _chat()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()