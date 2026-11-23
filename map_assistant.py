#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地图助手：地址关键词 → 坐标 → 多平台地图链接 + 网页可视化

用高德地理编码 API 把地址转成经纬度（GCJ-02 火星坐标，与高德瓦片同系，标点对准），
生成高德 / 百度 / 谷歌三种地图链接；--serve 启动本地网页，内嵌 Leaflet + 高德瓦片直接查看。

用法：
    .venv/bin/python map_assistant.py "天安门"                # CLI 检索
    .venv/bin/python map_assistant.py "深圳 科技园" --json     # JSON 输出
    .venv/bin/python map_assistant.py --serve --port 8892      # 启动网页版
    open "http://127.0.0.1:8892/?q=天安门"                     # 网页带参直达

配置：环境变量 AMAP_MAPS_API_KEY（高德开放平台 Web 服务 Key），或写在脚本同目录 .env
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "map_assets"
GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
DEFAULT_PORT = 8892

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>地图助手 · 地址检索</title>
<link rel="stylesheet" href="/assets/leaflet.css">
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f8ff; color: #1f2937; }
  .top { background: #fff; border-bottom: 1px solid #dbe3ef; padding: 14px 22px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .top h1 { font-size: 19px; color: #1e3a8a; }
  .top .sub { font-size: 12px; color: #6b7280; }
  .search { display: flex; gap: 8px; flex: 1; min-width: 260px; }
  .search input { flex: 1; padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 14px; outline: none; }
  .search input:focus { border-color: #1d4ed8; }
  .search .city { flex: 0 0 150px; }
  .search button { padding: 8px 20px; background: #1e3a8a; color: #fff; border: 0; border-radius: 8px; font-size: 14px; cursor: pointer; }
  .search button:hover { background: #1d4ed8; }
  .main { display: flex; gap: 14px; padding: 14px 22px; height: calc(100vh - 66px); }
  .panel { width: 350px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 3px rgba(30,58,138,.06); }
  .card.active { border-color: #1d4ed8; box-shadow: 0 0 0 2px rgba(29,78,216,.15); }
  .card .name { font-size: 15px; font-weight: 600; color: #1e3a8a; }
  .card .meta { font-size: 12px; color: #6b7280; margin-top: 4px; line-height: 1.6; }
  .card .badge { display: inline-block; background: #dcfce7; color: #166534; border-radius: 4px; padding: 1px 7px; font-size: 11px; margin-right: 4px; }
  .card .row { display: flex; gap: 6px; margin-top: 9px; flex-wrap: wrap; }
  .card a, .card button { font-size: 12px; padding: 4px 10px; border-radius: 6px; border: 1px solid #c7d2fe; background: #eef2ff; color: #3730a3; text-decoration: none; cursor: pointer; }
  .card a:hover, .card button:hover { background: #e0e7ff; border-color: #a5b4fc; }
  #map { flex: 1; border-radius: 10px; border: 1px solid #e2e8f0; }
  .hint { font-size: 12px; color: #94a3b8; text-align: center; padding: 4px; }
  .empty { text-align: center; color: #94a3b8; padding: 30px 10px; font-size: 13px; line-height: 1.8; }
  .leaflet-popup-content { font-size: 13px; line-height: 1.7; }
</style>
</head>
<body>
<div class="top">
  <h1>地图助手</h1>
  <span class="sub">地址关键词 → 坐标 → 多平台地图链接</span>
  <form class="search" id="fm" action="/" method="get">
    <input id="q" name="q" placeholder="输入地址关键词，如：天安门 / 深圳北站 / 西湖" value="__Q__" autofocus>
    <input class="city" id="city" name="city" placeholder="城市（可空）" value="__CITY__">
    <button type="submit">检索</button>
  </form>
</div>
<div class="main">
  <div class="panel" id="panel"><div class="empty">输入地址关键词后点击「检索」，左侧显示候选地点，右侧地图可拖拽缩放查看</div></div>
  <div id="map"></div>
</div>
<div class="hint">数据来源：高德地理编码 API｜坐标为 GCJ-02 火星坐标｜本地服务仅本机访问</div>

<script src="/assets/leaflet.js"></script>
<script>
const AMAP_TILE = "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}";
L.Icon.Default.mergeOptions({
  iconUrl: "/assets/marker-icon.png", iconRetinaUrl: "/assets/marker-icon-2x.png",
  shadowUrl: "/assets/marker-shadow.png",
});
L.Icon.Default.imagePath = "";   // 图标 URL 已是绝对路径，禁用自动拼接
const map = L.map("map", { zoomControl: true }).setView([34.8, 108.9], 4);
L.tileLayer(AMAP_TILE, { subdomains: ["1", "2", "3", "4"], maxZoom: 18, attribution: "&copy; 高德地图" }).addTo(map);

function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

let boundaryLayer = null;
async function showRegion(p) {
  // 行政区划：加载全域边界 GeoJSON 并缩放到整个区域
  if (boundaryLayer) { map.removeLayer(boundaryLayer); boundaryLayer = null; }
  const latlng = [p.lat, p.lng];
  if (!p.boundary_url) { map.setView(latlng, 11); return; }
  try {
    const resp = await fetch(p.boundary_url);
    if (!resp.ok) throw new Error(resp.status);
    const gj = await resp.json();
    boundaryLayer = L.geoJSON(gj, {
      style: { color: "#1d4ed8", weight: 2, fillColor: "#3b82f6", fillOpacity: 0.12 },
    }).addTo(map);
    const b = boundaryLayer.getBounds();
    if (b.isValid()) { map.fitBounds(b, { padding: [36, 36] }); }
  } catch (e) {
    map.setView(latlng, 11);
  }
}

function render(results, q) {
  const panel = document.getElementById("panel");
  panel.innerHTML = "";
  if (!results.length) {
    panel.innerHTML = '<div class="empty">未找到与「' + esc(q) + '」相关的地点。\\n试试更完整的地址，如「城市 + 区 + 路名 / 地标名」</div>';
    return;
  }
  const markers = [];
  results.forEach((p, i) => {
    const latlng = [p.lat, p.lng];
    const card = document.createElement("div");
    card.className = "card" + (i === 0 ? " active" : "");
    card.innerHTML =
      '<div class="name">' + esc(p.name) + "</div>" +
      '<div class="meta">' + (p.is_region ? '<span class="badge">行政区划 · 全域</span>' : "") +
      esc(p.province + " " + p.city + " " + p.district) + " · " + esc(p.level) +
      "<br>" + p.lng.toFixed(6) + ", " + p.lat.toFixed(6) + "</div>" +
      '<div class="row">' +
      '<button data-i="' + i + '">' + (p.is_region ? "显示全域" : "定位") + "</button>" +
      '<a href="' + p.links.gaode + '" target="_blank" rel="noopener">高德</a>' +
      '<a href="' + p.links.baidu + '" target="_blank" rel="noopener">百度</a>' +
      '<a href="' + p.links.google + '" target="_blank" rel="noopener">谷歌</a>' +
      "</div>";
    card.querySelector("button").onclick = () => { if (p.is_region) showRegion(p); else zoomTo(i); };
    panel.appendChild(card);
    const mk = L.marker(latlng).addTo(map).bindPopup(
      "<b>" + esc(p.name) + "</b><br>" + p.lng.toFixed(6) + ", " + p.lat.toFixed(6) +
      '<br><a href="' + p.links.gaode + '" target="_blank" rel="noopener">高德 ↗</a> ' +
      '<a href="' + p.links.baidu + '" target="_blank" rel="noopener">百度 ↗</a> ' +
      '<a href="' + p.links.google + '" target="_blank" rel="noopener">谷歌 ↗</a>');
    markers.push({ latlng, mk });
    if (i === 0) {
      if (p.is_region) { showRegion(p); } else { map.setView(latlng, 14); }
    }
    mk.on("click", () => {
      document.querySelectorAll(".card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
    });
  });
  window.__markers = markers;
  window.__zoomTo = zoomTo;
  function zoomTo(i) {
    const m = markers[i];
    if (m) { map.setView(m.latlng, 16); m.mk.openPopup(); }
  }
  if (results.length > 1) {
    map.fitBounds(L.latLngBounds(markers.map(m => m.latlng)).pad(0.15), { maxZoom: 12 });
  }
}

async function doSearch(q, city) {
  if (!q) return;
  const url = "/api/geocode?q=" + encodeURIComponent(q) + (city ? "&city=" + encodeURIComponent(city) : "");
  const panel = document.getElementById("panel");
  panel.innerHTML = '<div class="empty">正在检索「' + esc(q) + '」…</div>';
  try {
    const resp = await fetch(url);
    const data = await resp.json();
    render(data.results, q);
  } catch (e) {
    panel.innerHTML = '<div class="empty">检索失败：' + esc(String(e)) + "</div>";
  }
}

const params = new URLSearchParams(location.search);
const q0 = params.get("q") || "";
const c0 = params.get("city") || "";
document.getElementById("fm").addEventListener("submit", (e) => {
  e.preventDefault();
  doSearch(document.getElementById("q").value.trim(), document.getElementById("city").value.trim());
});
if (q0) doSearch(q0, c0);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# 地理编码
# --------------------------------------------------------------------------- #

def _load_key() -> str:
    key = os.getenv("AMAP_MAPS_API_KEY", "").strip()
    if key:
        return key
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text("utf-8").splitlines():
            line = line.strip()
            if line.startswith("AMAP_MAPS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _ssl_ctx() -> ssl.SSLContext:
    # 本机 venv 的 Python 缺系统 CA 链，用 certifi 构造
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "map-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=20, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_links(name: str, lng: float, lat: float) -> dict:
    """生成各平台地图链接；高德/百度为搜索解析，谷歌按名称检索。"""
    enc = urllib.parse.quote(name)
    return {
        "gaode": f"https://uri.amap.com/marker?position={lng},{lat}&name={enc}",
        "baidu": f"https://map.baidu.com/search/{enc}",
        "google": f"https://www.google.com/maps/search/?api=1&query={enc}",
    }


def _friendly_error(info: str) -> str:
    if "ENGINE_RESPONSE_DATA_ERROR" in info or "INVALID_USER_KEY" in info.upper():
        return ("关键词未被高德识别（过短或过于宽泛），"
                "试试更完整的地址，如「城市 + 区 + 道路/地标名」")
    return f"高德接口返回错误：{info}"


def geocode(address: str, city: str = "", limit: int = 8) -> list[dict]:
    """地址 → 候选地点列表（高德地理编码，GCJ-02 坐标）。"""
    address = (address or "").strip()
    if not address:
        return []
    key = _load_key()
    if not key:
        raise RuntimeError("未配置 AMAP_MAPS_API_KEY（环境变量或 .env）")

    def _call(city_part: str):
        params = {"address": address, "key": key, "output": "json"}
        if city_part.strip():
            params["city"] = city_part.strip()
        return _http_get(GEOCODE_URL + "?" + urllib.parse.urlencode(params))

    data = _call(city)
    # 带城市限定报错时，去掉 city 再试一次（部分地点名在限定城市下无法解析）
    if str(data.get("status")) != "1" and city.strip():
        data = _call("")
    if str(data.get("status")) != "1":
        raise RuntimeError(_friendly_error(str(data.get("info", "未知"))))
    out = []
    for g in data.get("geocodes") or []:
        loc = (g.get("location") or "").split(",")
        if len(loc) != 2:
            continue
        try:
            lng, lat = float(loc[0]), float(loc[1])
        except ValueError:
            continue
        city_raw = g.get("city", "")
        name = (g.get("formatted_address") or g.get("township") or address).strip()
        adcode = str(g.get("adcode", "") or "")
        level = g.get("level", "") or ""
        # 行政区划（省/市/区县）：提供全域边界 GeoJSON（DataV），省市级取 _full（儿童区划），区县级取自身轮廓
        is_region = level in ("省", "市", "区县", "区", "县", "自治州", "地区", "盟")
        boundary_url = ""
        if is_region and len(adcode) == 6:
            suffix = "_full.json" if adcode.endswith("00") else ".json"
            boundary_url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}{suffix}"
        out.append({
            "name": name,
            "lng": lng,
            "lat": lat,
            "province": g.get("province", "") or "",
            "city": city_raw if isinstance(city_raw, str) else (city_raw[0] if city_raw else ""),
            "district": g.get("district", "") or "",
            "adcode": adcode,
            "level": level,
            "is_region": is_region,
            "boundary_url": boundary_url,
            "links": make_links(name, lng, lat),
        })
        if len(out) >= max(1, limit):
            break
    return out


def health() -> dict:
    return {"ok": True, "service": "map-assistant",
            "key": "ready" if _load_key() else "missing"}


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, jsonify, request, send_from_directory

    app = Flask("map_assistant")

    @app.get("/")
    def index():
        q = request.args.get("q", "")
        city = request.args.get("city", "")
        return (PAGE.replace("__Q__", q.replace("'", "").replace('"', ""))
                .replace("__CITY__", city.replace("'", "").replace('"', "")))

    @app.get("/api/geocode")
    def api_geocode():
        q = (request.args.get("q") or "").strip()
        city = (request.args.get("city") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "缺少 q 参数"}), 400
        try:
            results = geocode(q, city)
            return jsonify({"ok": True, "q": q, "count": len(results), "results": results})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "results": []}), 502

    @app.get("/health")
    def h():
        return jsonify(health())

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(ASSETS, filename)

    return app


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="地图助手：地址关键词 → 坐标 → 地图链接")
    ap.add_argument("address", nargs="?", default="", help="地址关键词，如：天安门")
    ap.add_argument("--city", default="", help="限定城市，如：深圳")
    ap.add_argument("--json", action="store_true", help="JSON 输出")
    ap.add_argument("--serve", action="store_true", help="启动网页版")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    if args.serve:
        app = create_app()
        print(f"地图助手：http://{args.host}:{args.port}/?q=天安门")
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if not args.address:
        ap.print_help()
        return

    try:
        results = geocode(args.address, args.city)
    except Exception as exc:
        print(f"检索失败：{exc}", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps({"q": args.address, "count": len(results), "results": results},
                         ensure_ascii=False, indent=2))
        return
    if not results:
        print(f"未找到与「{args.address}」相关的地点，试试更完整的地址（城市 + 区 + 路名/地标）")
        return
    for i, p in enumerate(results, 1):
        print(f"[{i}] {p['name']}")
        print(f"    {p['province']} {p['city']} {p['district']} · {p['level']} · {p['lng']:.6f}, {p['lat']:.6f}")
        for label, url in p["links"].items():
            print(f"    {label}: {url}")
        print()


if __name__ == "__main__":
    main()