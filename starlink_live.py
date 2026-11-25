#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpaceX 星链（Starlink）卫星实时位置统计 + 在线浏览图

数据：CelesTrak 公开 TLE（JSON，GROUP=starlink），抓取时刻的近实时轨道根数。
位置：sgp4 把每颗卫星的 TLE 传播到"当前时刻"（UTC），
      TEME → ECEF（GMST 旋转）→ WGS84 大地坐标，得到星下点纬度/经度 + 轨道高度。

输出（starlink_data/）：
- starlink_live.html          在线浏览图（ECharts，可离线打开）
    · 世界地图散点：约 1.1 万颗星链按轨道高度着色，可缩放/平移/悬停详情
    · 高度分布直方图 + 倾角分布直方图
    · 顶部统计卡片：卫星总数、平均高度、高度/倾角范围、数据时刻
- starlink_live_<time>.json   全量实时位置 + 统计
- tles_starlink_<time>.json   TLE 原始快照（--cached 可离线复用最近一份）

用法：
  python starlink_live.py               # 抓取最新 TLE 并出图
  python starlink_live.py --cached      # 复用本地最新 TLE 快照（离线）
  python starlink_live.py --limit 3000  # 只算前 N 颗（调试/轻量）

依赖：sgp4、numpy；echarts.min.js / world.js 首次运行自动下载到 starlink_data/
      （jsdelivr CDN，失败时回退为在线引用）。
"""
import argparse
import json
import os
import ssl
import urllib.request
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday
try:
    from sgp4.omm import initialize as _omm_init   # CelesTrak JSON = OMM 元素集
except ImportError:                                 # 老版本 sgp4 兜底
    _omm_init = None

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "starlink_data")
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json"
# CelesTrak 限流时的镜像（MTalon63/ReTLEctor 公共实例，同样来自 CelesTrak 上游）
TLE_MIRROR = "https://retlector.eu/starlink/tle"
LATEST_TLE = os.path.join(OUT_DIR, "tles_latest.json")
ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"
WORLD_JS_CDN = "https://cdn.jsdelivr.net/npm/echarts@4.9.0/map/js/world.js"

EARTH_R = 6378.137          # km, WGS84 赤道半径
WGS84_F = 1.0 / 298.257223563
J2000 = 2451545.0


# ---------------- TLE 抓取 ----------------
def _fetch_json(url: str, timeout: int = 60) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return json.loads(r.read().decode())


def _parse_tle_text(text: str) -> list[dict]:
    """解析 TLE 文本（每对象 3 行：名称行 + 1/2 行），转为统一 dict 结构。"""
    objs, name, line1 = [], None, None
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("1 "):
            line1 = line
        elif line.startswith("2 ") and line1 is not None:
            objs.append({
                "OBJECT_NAME": name or f"SAT-{line[2:7].strip()}",
                "NORAD_CAT_ID": line[2:7].strip(),
                "INCLINATION": float(line[8:16]),
                "MEAN_MOTION": float(line[52:63]),
                "TLE_LINE1": line1,
                "TLE_LINE2": line,
            })
            name, line1 = None, None
        elif line.strip():
            name = line.strip()             # 名称行
    return objs


def fetch_tles() -> list[dict]:
    """抓取星链 TLE：优先 CelesTrak JSON，限流时回退到 ReTLEctor 镜像。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    tles, source = None, ""
    try:
        tles = _fetch_json(TLE_URL)
        source = "CelesTrak"
    except Exception as e:
        print(f"CelesTrak 不可用（{type(e).__name__}）→ 回退镜像 {TLE_MIRROR}")
        try:
            req = urllib.request.Request(TLE_MIRROR,
                                        headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120, context=_CTX) as r:
                text = r.read().decode()
            tles = _parse_tle_text(text)
            source = "ReTLEctor(CelesTrak 镜像)"
        except Exception as e2:
            raise SystemExit(f"镜像也失败：{type(e2).__name__} {e2}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(OUT_DIR, f"tles_starlink_{stamp}.json"), "w",
              encoding="utf-8") as f:
        json.dump(tles, f, ensure_ascii=False)
    with open(LATEST_TLE, "w", encoding="utf-8") as f:
        json.dump(tles, f, ensure_ascii=False)
    print(f"TLE 快照已落盘（{len(tles)} 颗，来源 {source}）")
    return tles


def load_cached_tles() -> list[dict]:
    if not os.path.exists(LATEST_TLE):
        raise SystemExit("没有本地 TLE 快照，去掉 --cached 先联网抓取一次")
    with open(LATEST_TLE, encoding="utf-8") as f:
        return json.load(f)


# ---------------- 轨道传播：TLE → 星下点 ----------------
def gmst_rad(jd: float) -> float:
    """格林尼治恒星时（IERS 公式，弧度）。UT1 以 UTC 近似，地图展示精度足够。"""
    d = jd - J2000
    return 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * d) % (2 * np.pi)


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple:
    """ECEF(km) → (纬度°, 经度°, 海拔km)，WGS84 迭代求解。"""
    e2 = WGS84_F * (2 - WGS84_F)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(4):
        n = EARTH_R / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    n = EARTH_R / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)


def _make_satrec(tle: dict) -> Satrec | None:
    """由 TLE 条目构造 Satrec：兼容 3 行 TLE 文本（镜像）与 OMM JSON（CelesTrak）。"""
    if "TLE_LINE1" in tle:
        sat = Satrec.twoline2rv(tle["TLE_LINE1"], tle["TLE_LINE2"])
    elif _omm_init is not None and "MEAN_MOTION" in tle:
        sat = Satrec()
        _omm_init(sat, tle)
    else:
        return None
    return None if sat.error else sat


def propagate(tles: list[dict], when: datetime, limit: int = None) -> tuple:
    """把每颗 TLE 传播到 when，返回 (位置列表, 统计字典)。"""
    jd, fr = jday(when.year, when.month, when.day,
                  when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    rot = gmst_rad(jd + fr)
    c, s = np.cos(rot), np.sin(rot)

    # 批量构建 + 向量化传播（sgp4 2.27 API）
    sats, names, incls, periods, norads = [], [], [], [], []
    for tle in tles[:limit]:
        sat = _make_satrec(tle)
        if sat is None:
            continue
        sats.append(sat)
        names.append(tle.get("OBJECT_NAME", ""))
        incls.append(float(tle.get("INCLINATION", 0)))
        periods.append(1440.0 / float(tle.get("MEAN_MOTION", 15)))
        norads.append(str(tle.get("NORAD_CAT_ID", "")))
    n_sats = len(sats)
    errs, r, _ = SatrecArray(sats).sgp4(np.array([jd]), np.array([fr]))
    # errs: (N,1) 错误码；r: (N,1,3) TEME km
    x = r[:, 0, 0]
    y = r[:, 0, 1]
    z = r[:, 0, 2]
    xe = c * x + s * y          # TEME → ECEF（向量化）
    ye = -s * x + c * y
    ze = z

    pts = []
    for i in range(n_sats):
        if errs[i, 0]:
            continue
        lat, lon, alt = ecef_to_geodetic(xe[i], ye[i], ze[i])
        pts.append({
            "name": names[i],
            "norad": norads[i],
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "alt": round(alt, 1),
            "incl": round(incls[i], 2),
            "period": round(periods[i], 2),
        })
    return pts, {"propagate_errors": int((errs != 0).sum())}


# ---------------- 统计 ----------------
def summarize(pts: list[dict]) -> dict:
    alts = np.array([p["alt"] for p in pts])
    incls = np.array([p["incl"] for p in pts])

    def hist(v, edges):
        counts, _ = np.histogram(v, bins=edges)
        labels = [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)]
        return [{"name": labels[i], "count": int(counts[i])}
                for i in range(len(labels))]

    return {
        "count": len(pts),
        "mean_alt": round(float(alts.mean()), 1),
        "alt_min": round(float(alts.min()), 1),
        "alt_max": round(float(alts.max()), 1),
        "mean_incl": round(float(incls.mean()), 2),
        "incl_min": round(float(incls.min()), 2),
        "incl_max": round(float(incls.max()), 2),
        "alt_hist": hist(alts, [280, 340, 400, 460, 520, 580, 640, 700, 1200]),
        "incl_hist": hist(incls, [40, 44, 48, 52, 56, 60, 66, 74, 85, 100, 120]),
    }


# ---------------- ECharts 在线浏览图 ----------------
def _download_to(url: str, dst_name: str) -> str:
    """下载 js 资源到本地缓存；成功返回本地路径，失败返回原 CDN 地址。"""
    local = os.path.join(OUT_DIR, dst_name)
    if os.path.exists(local):
        return local
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
            data = r.read()
        with open(local, "wb") as f:
            f.write(data)
        return local
    except Exception:
        return url


def ensure_echarts_local() -> str:
    """把 echarts.min.js 缓存到本地（缺失则自动下载），失败回退 CDN 在线引用。"""
    return _download_to(ECHARTS_CDN, "echarts.min.js")


def ensure_world_js() -> str:
    """把世界地图注册脚本 world.js 缓存到本地（缺失则自动下载），失败回退 CDN。"""
    return _download_to(WORLD_JS_CDN, "world.js")


def render_html(pts: list[dict], stats: dict, echarts_src: str, fetched_at: str,
                world_src: str = "world.js") -> str:
    """世界地图散点 + 高度/倾角直方图 + 统计卡片，返回单页 HTML 字符串。

    echarts_src / world_src 传本地缓存路径（如 starlink_data/xxx.js）或在线 CDN 地址。
    """
    scatter = [{"name": p["name"], "value": [p["lon"], p["lat"], p["alt"]],
                "incl": p["incl"], "period": p["period"]} for p in pts]
    data_js = json.dumps({"scatter": scatter, "stats": stats,
                          "fetched_at": fetched_at},
                         ensure_ascii=False).replace("</", "<\\/")

    # 直方图 option（避免 f-string 大括号转义，用占位符替换）
    alt_hist_js = json.dumps({
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 20, "bottom": 30, "top": 30},
        "xAxis": {"type": "category", "data": [h["name"] for h in stats["alt_hist"]]},
        "yAxis": {"type": "value", "name": "卫星数"},
        "series": [{"type": "bar", "data": [h["count"] for h in stats["alt_hist"]],
                    "itemStyle": {"color": "#5470c6"},
                    "label": {"show": True, "position": "top", "fontSize": 10}}],
    }, ensure_ascii=False)
    incl_hist_js = json.dumps({
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 20, "bottom": 30, "top": 30},
        "xAxis": {"type": "category", "data": [h["name"] for h in stats["incl_hist"]]},
        "yAxis": {"type": "value", "name": "卫星数"},
        "series": [{"type": "bar", "data": [h["count"] for h in stats["incl_hist"]],
                    "itemStyle": {"color": "#91cc75"},
                    "label": {"show": True, "position": "top", "fontSize": 10}}],
    }, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8">
<title>SpaceX 星链卫星实时位置</title>
<script src="__ECHARTS__"></script>
<script src="__WORLD__"></script>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 0;
         background: #0d1117; color: #e6edf3; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 20px 16px 50px; }
  h1 { font-size: 22px; margin: 0 0 6px; color: #fff; }
  .meta { color: #8b949e; font-size: 13px; margin-bottom: 16px; }
  .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .card { flex: 1; min-width: 140px; background: #161b22; border: 1px solid #30363d;
          border-radius: 10px; padding: 12px 16px; }
  .card .v { font-size: 22px; font-weight: 700; color: #58a6ff; }
  .card .l { color: #8b949e; font-size: 12px; margin-top: 2px; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
           margin-bottom: 16px; }
  .panel h2 { font-size: 15px; color: #c9d1d9; margin: 12px 16px 0; }
  #map { height: 680px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid2 .panel { margin-bottom: 0; }
  #altHist, #inclHist { height: 320px; }
  @media (max-width: 800px) { .grid2 { grid-template-columns: 1fr; } }
</style></head>
<body><div class="wrap">
  <h1>SpaceX 星链（Starlink）卫星实时位置</h1>
  <div class="meta">数据：CelesTrak TLE（近实时轨道根数）｜ 位置：sgp4 传播至 UTC
      __FETCHED__（本机时刻）｜ 地图：ECharts world</div>
  <div class="cards">
    <div class="card"><div class="v">__COUNT__</div><div class="l">星链卫星数（含退役在轨）</div></div>
    <div class="card"><div class="v">__MEAN_ALT__ km</div><div class="l">平均轨道高度</div></div>
    <div class="card"><div class="v">__ALT_RANGE__ km</div><div class="l">高度范围（近地点-远地点）</div></div>
    <div class="card"><div class="v">__MEAN_INCL__°</div><div class="l">平均轨道倾角</div></div>
    <div class="card"><div class="v">__INCL_RANGE__°</div><div class="l">倾角范围</div></div>
  </div>
  <div class="panel"><h2>实时星下点分布（颜色 = 轨道高度）</h2><div id="map"></div></div>
  <div class="grid2">
    <div class="panel"><h2>轨道高度分布</h2><div id="altHist"></div></div>
    <div class="panel"><h2>轨道倾角分布</h2><div id="inclHist"></div></div>
  </div>
</div>
<script>
const DATA = __DATA__;
var c1 = echarts.init(document.getElementById('map'));
c1.setOption({
  tooltip: { formatter: p => p.seriesType === 'scatter'
      ? `${p.name}<br/>高度 ${p.value[2].toFixed(0)} km ｜ 倾角 ${p.data.incl.toFixed(1)}°`
      : '' },
  visualMap: { min: 280, max: 1200, dimension: 2, calculable: true,
    inRange: { color: ['#4cc9f0', '#4361ee', '#3a0ca3', '#7209b7'] },
    textStyle: { color: '#c9d1d9' }, left: 10, bottom: 20 },
  geo: { map: 'world', roam: true, scaleLimit: { min: 0.6, max: 30 },
    itemStyle: { areaColor: '#0e2233', borderColor: '#2d4a66', borderWidth: 0.6 },
    emphasis: { itemStyle: { areaColor: '#16344d' }, label: { show: false } } },
  series: [{
    name: '星链', type: 'scatter', coordinateSystem: 'geo',
    data: DATA.scatter, symbolSize: 2,
    itemStyle: { opacity: 0.75 },
  }]
});
var c2 = echarts.init(document.getElementById('altHist'));
c2.setOption(__ALTHIST__);
var c3 = echarts.init(document.getElementById('inclHist'));
c3.setOption(__INCLHIST__);
window.addEventListener('resize', () => { c1.resize(); c2.resize(); c3.resize(); });
</script>
</body></html>
"""
    out = html.replace("__ECHARTS__", echarts_src)
    out = out.replace("__WORLD__", world_src)
    out = out.replace("__DATA__", data_js)
    out = out.replace("__ALTHIST__", alt_hist_js)
    out = out.replace("__INCLHIST__", incl_hist_js)
    out = out.replace("__FETCHED__", fetched_at)
    out = out.replace("__COUNT__", f"{stats['count']:,}")
    out = out.replace("__MEAN_ALT__", f"{stats['mean_alt']:,.0f}")
    out = out.replace("__ALT_RANGE__", f"{stats['alt_min']:,.0f}-{stats['alt_max']:,.0f}")
    out = out.replace("__MEAN_INCL__", f"{stats['mean_incl']:.1f}")
    out = out.replace("__INCL_RANGE__", f"{stats['incl_min']:.1f}-{stats['incl_max']:.1f}")
    return out


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="星链卫星实时位置统计 + 在线浏览图")
    ap.add_argument("--cached", action="store_true", help="复用本地最新 TLE（离线）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 颗卫星")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tles = load_cached_tles() if args.cached else fetch_tles()
    now = datetime.now(timezone.utc)
    pts, extra = propagate(tles, now, args.limit)
    stats = summarize(pts)
    stats.update(extra)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {"fetched_at": now.isoformat(), "stats": stats, "points": pts}
    with open(os.path.join(OUT_DIR, f"starlink_live_{stamp}.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"\n星链卫星：{stats['count']:,} 颗（传播失败 {stats.get('propagate_errors', 0)}）")
    print(f"平均高度 {stats['mean_alt']:.0f} km（范围 {stats['alt_min']:.0f}-{stats['alt_max']:.0f} km）")
    print(f"平均倾角 {stats['mean_incl']:.1f}°（范围 {stats['incl_min']:.1f}-{stats['incl_max']:.1f}°）")

    src = ensure_echarts_local()
    world = ensure_world_js()
    html = render_html(pts, stats, src, now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                       world_src=world)
    html_path = os.path.join(OUT_DIR, "starlink_live.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"JSON 报告 → {os.path.join(OUT_DIR, f'starlink_live_{stamp}.json')}")
    print(f"在线浏览图 → {html_path}")


if __name__ == "__main__":
    main()
