#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""星链卫星助手：SpaceX Starlink 实时位置统计 + 问答 + 在线浏览图

- 数据层 starlink_live.py：CelesTrak 公开 TLE → sgp4 传播到当前时刻 →
  WGS84 星下点 + 轨道统计（卫星数/高度/倾角/周期）
- 助手层：规则问答（秒回、无 LLM 依赖），TLE 本地缓存 6 小时
- HTTP :8914：/（看板+问答页）、/map（在线地图）、/api/qa、/api/stats、/health

用法：
    .venv/bin/python starlink_agent.py "现在有多少颗星链卫星？"
    .venv/bin/python starlink_agent.py --chat
    .venv/bin/python starlink_agent.py --serve --port 8914
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request, send_from_directory

import starlink_live as sl

HERE = os.path.dirname(os.path.abspath(__file__))
TLE_TTL = 6 * 3600          # TLE 复用窗口（星链根数约每 3-6 小时更新）

# ---------------- 数据快照（带缓存 + 锁，避免并发重复传播） ----------------
_SNAP = {"lock": threading.Lock(), "t": 0.0, "pts": [], "stats": {}, "fetched_at": "",
         "note": "", "period_mean": 0.0}


def build_snapshot(force: bool = False) -> dict:
    """构建当前位置快照；缓存未过期且未强制则直接复用。"""
    with _SNAP["lock"]:
        age = time.time() - _SNAP["t"]
        if _SNAP["pts"] and age < TLE_TTL and not force:
            return dict(_SNAP, lock=None)
        try:
            latest = os.path.join(sl.OUT_DIR, os.path.basename(sl.LATEST_TLE))
            if os.path.exists(latest) and time.time() - os.path.getmtime(latest) < TLE_TTL \
                    and not force:
                tles = sl.load_cached_tles()
                note = "本地缓存 TLE（最近 6 小时内抓取）"
            else:
                tles = sl.fetch_tles()
                note = "实时抓取 CelesTrak TLE"
        except SystemExit as exc:
            # 联网失败且无本地缓存，兜底返回错误
            raise
        now = datetime.now(timezone.utc)
        pts, extra = sl.propagate(tles, now)
        stats = sl.summarize(pts)
        stats.update(extra)
        periods = [p["period"] for p in pts]
        _SNAP.update({
            "t": time.time(), "pts": pts, "stats": stats,
            "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"), "note": note,
            "period_mean": round(sum(periods) / len(periods), 2) if periods else 0.0,
        })
        print(f"快照已构建：{stats['count']:,} 颗（{note}）")
        return dict(_SNAP, lock=None)


def _hist_top(hist: list[dict], n: int = 4) -> str:
    top = sorted(hist, key=lambda h: -h["count"])[:n]
    return "、".join(f"{h['name']}km {h['count']}颗" for h in top)


def answer(q: str) -> str:
    """规则问答：按问题关键词组装统计回答（无 LLM 依赖，秒回）。"""
    force = any(k in q for k in ("刷新", "更新", "重新"))
    snap = build_snapshot(force=force)
    st = snap["stats"]
    head = (f"SpaceX 星链卫星当前概览（数据时刻 {snap['fetched_at']}，"
            f"{snap['note']}）：")

    secs: list[str] = []
    if any(k in q for k in ("多少", "数量", "总数", "几颗")):
        secs.append(f"当前共追踪 {st['count']:,} 颗星链卫星"
                    f"（含退役在轨，传播失败 {st.get('propagate_errors', 0)}）。")
    if "高度" in q:
        secs.append(f"平均轨道高度 {st['mean_alt']:.0f} km，"
                    f"范围 {st['alt_min']:.0f}~{st['alt_max']:.0f} km；"
                    f"主要区间：{_hist_top(st['alt_hist'])}。")
    if "倾角" in q:
        secs.append(f"平均轨道倾角 {st['mean_incl']:.1f}°，"
                    f"范围 {st['incl_min']:.1f}~{st['incl_max']:.1f}°；"
                    f"分布：{_hist_top(st['incl_hist'], 4)}。")
    if "周期" in q:
        secs.append(f"平均轨道周期 {snap['period_mean']:.0f} 分钟。")
    if "时间" in q or "最新" in q or "数据" in q:
        secs.append(f"本次数据基于 {snap['fetched_at']} 抓取的 TLE（{snap['note']}）。")

    if not secs:   # 无特指主题 → 完整概览
        secs = [
            f"共追踪 {st['count']:,} 颗（含退役在轨，传播失败 {st.get('propagate_errors', 0)}）",
            f"· 平均轨道高度 {st['mean_alt']:.0f} km（范围 {st['alt_min']:.0f}~{st['alt_max']:.0f} km）；"
            f"高度分布：{_hist_top(st['alt_hist'])}",
            f"· 平均轨道倾角 {st['mean_incl']:.1f}°（范围 {st['incl_min']:.1f}~{st['incl_max']:.1f}°）；"
            f"倾角分布：{_hist_top(st['incl_hist'], 4)}",
            f"· 平均轨道周期 {snap['period_mean']:.0f} 分钟",
        ]
    secs.append("在线浏览图：http://127.0.0.1:8914/map")
    return "\n".join([head, *secs])


# ---------------- HTTP 服务 ----------------
def _asset_srcs() -> tuple[str, str]:
    """本地已缓存则走 /assets/，否则回退 CDN 在线引用。"""
    e = sl.ensure_echarts_local()
    w = sl.ensure_world_js()
    e = "/assets/echarts.min.js" if os.path.exists(e) else e
    w = "/assets/world.js" if os.path.exists(w) else w
    return e, w


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        try:
            st = _SNAP["stats"]
            count = st.get("count")
        except Exception:
            count = None
        return jsonify({"ok": True, "service": "starlink-assistant",
                        "count": count, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.get("/api/stats")
    def api_stats():
        snap = build_snapshot()
        st = snap["stats"]
        return jsonify({"ok": True, "fetched_at": snap["fetched_at"], "note": snap["note"],
                        "stats": st})

    @app.route("/api/qa", methods=["GET", "POST"])
    def api_qa():
        if request.method == "POST":
            q = (request.get_json(silent=True) or {}).get("question", "")
        else:
            q = request.args.get("question", "")
        q = str(q).strip()
        if not q:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        try:
            ans = answer(q)
        except SystemExit as exc:
            return jsonify({"ok": False, "error": str(exc)}), 502
        return jsonify({"ok": True, "question": q, "answer": ans})

    @app.get("/map")
    def map_page():
        snap = build_snapshot()
        ech, wrl = _asset_srcs()
        html = sl.render_html(snap["pts"], snap["stats"], ech,
                              snap["fetched_at"], world_src=wrl)
        return Response(html, mimetype="text/html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(sl.OUT_DIR, name)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    return app


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>星链卫星助手 · SpaceX Starlink 实时位置</title>
<style>
body{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f6f7f9;color:#222}
.wrap{max-width:860px;margin:0 auto;padding:22px 16px 40px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#888;font-size:13px;margin-bottom:16px}
.card{background:#fff;border-radius:12px;padding:18px 20px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card h2{margin:0 0 10px;font-size:15px}
input,button{font-size:15px;padding:10px 14px;border-radius:8px;border:1px solid #d5d9e0}
input{flex:1;margin-right:10px;outline:none}
button{background:#2563eb;color:#fff;border:none;cursor:pointer}
button:disabled{background:#93b4f5}
#out{white-space:pre-wrap;background:#f8fafc;border:1px solid #eef1f5;border-radius:8px;padding:12px;font-size:14px;margin-top:10px;display:none}
iframe{width:100%;height:620px;border:1px solid #e5e9f0;border-radius:10px;background:#0d1117}
.row{display:flex}
</style></head>
<body><div class="wrap">
<h1>SpaceX 星链（Starlink）卫星实时位置</h1>
<div class="sub">公开数据 CelesTrak TLE + sgp4 轨道传播 · 本地运算 · 无密钥</div>
<div class="card"><h2>数据问答</h2>
<div class="row"><input id="q" placeholder="如：现在有多少颗星链卫星？平均高度多少？">
<button id="go" onclick="ask()">问</button></div>
<div id="out"></div></div>
<div class="card"><h2>实时星下点分布（颜色 = 轨道高度）</h2>
<iframe src="/map"></iframe></div>
</div>
<script>
function ask(){const q=document.getElementById('q').value.trim();if(!q)return;
 const o=document.getElementById('out');o.style.display='block';o.textContent='查询中…';
 fetch('/api/qa',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})
 .then(r=>r.json()).then(d=>{o.textContent=d.ok?d.answer:('错误：'+(d.error||''))})
 .catch(e=>{o.textContent='失败：'+e});}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
</script>
</body></html>"""


# ---------------- 命令行 ----------------
def main():
    ap = argparse.ArgumentParser(description="星链卫星助手（实时位置/轨道统计）")
    ap.add_argument("question", nargs="?", default="", help="问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8914)
    args = ap.parse_args()

    if args.serve:
        app = create_app()
        print(f"星链卫星助手 HTTP 服务：http://{args.host}:{args.port}")
        print("  页面: /    在线地图: /map   统计: /api/stats")
        print('  问答: POST /api/qa {"question":"..."}   /health')
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
            try:
                print(answer(q), "\n")
            except SystemExit as exc:
                print(f"错误：{exc}")
        return

    print(answer(args.question))


if __name__ == "__main__":
    main()