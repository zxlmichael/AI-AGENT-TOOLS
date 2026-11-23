#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""天气预报助手 · 数据层（纯标准库）

数据源：Open-Meteo 免费 API（无需 key，forecast_days 最大 16，
本设计取 10 天）。全国主要城市（直辖市+省会+计划单列市+重要城市）
硬编码经纬度表。

设计文档：无（直接实现，参考同工作区 stock_data.py 模式）
"""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
DAYS = 10  # 预测天数

# 全国主要城市：name -> (lat, lon, 省份/区域)
CITIES: dict[str, tuple[float, float, str]] = {
    # 直辖市
    "北京": (39.9042, 116.4074, "北京"), "上海": (31.2304, 121.4737, "上海"),
    "天津": (39.3434, 117.3616, "天津"), "重庆": (29.5630, 106.5516, "重庆"),
    # 省会 / 首府
    "石家庄": (38.0428, 114.5149, "河北"), "太原": (37.8706, 112.5489, "山西"),
    "呼和浩特": (40.8414, 111.7519, "内蒙古"), "沈阳": (41.8057, 123.4315, "辽宁"),
    "长春": (43.8171, 125.3235, "吉林"), "哈尔滨": (45.8038, 126.5350, "黑龙江"),
    "南京": (32.0603, 118.7969, "江苏"), "杭州": (30.2741, 120.1551, "浙江"),
    "合肥": (31.8206, 117.2272, "安徽"), "福州": (26.0745, 119.2965, "福建"),
    "南昌": (28.6820, 115.8579, "江西"), "济南": (36.6512, 117.1201, "山东"),
    "郑州": (34.7466, 113.6254, "河南"), "武汉": (30.5928, 114.3055, "湖北"),
    "长沙": (28.2282, 112.9388, "湖南"), "广州": (23.1291, 113.2644, "广东"),
    "南宁": (22.8170, 108.3669, "广西"), "海口": (20.0444, 110.1999, "海南"),
    "成都": (30.5728, 104.0668, "四川"), "贵阳": (26.6470, 106.6302, "贵州"),
    "昆明": (24.8801, 102.8329, "云南"), "拉萨": (29.6520, 91.1721, "西藏"),
    "西安": (34.3416, 108.9398, "陕西"), "兰州": (36.0611, 103.8343, "甘肃"),
    "西宁": (36.6171, 101.7782, "青海"), "银川": (38.4872, 106.2309, "宁夏"),
    "乌鲁木齐": (43.8256, 87.6168, "新疆"),
    # 计划单列市 / 重要城市
    "深圳": (22.5431, 114.0579, "广东"), "青岛": (36.0671, 120.3826, "山东"),
    "大连": (38.9140, 121.6147, "辽宁"), "宁波": (29.8683, 121.5440, "浙江"),
    "厦门": (24.4798, 118.0894, "福建"), "苏州": (31.2989, 120.5853, "江苏"),
    "无锡": (31.4912, 120.3119, "江苏"), "佛山": (23.0218, 113.1219, "广东"),
    "东莞": (23.0207, 113.7518, "广东"), "三亚": (18.2528, 109.5119, "海南"),
    "珠海": (22.2707, 113.5767, "广东"), "洛阳": (34.6197, 112.4540, "河南"),
    "桂林": (25.2736, 110.2900, "广西"), "大理": (25.6065, 100.2676, "云南"),
    "丽江": (26.8721, 100.2299, "云南"), "张家界": (29.1170, 110.4790, "湖南"),
    "烟台": (37.4638, 121.4479, "山东"), "温州": (27.9938, 120.6994, "浙江"),
    "徐州": (34.2044, 117.2858, "江苏"), "唐山": (39.6305, 118.1802, "河北"),
}

# WMO 天气码 -> 中文描述
WMO = {
    0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷暴伴冰雹",
}


def _ssl_ctx() -> ssl.SSLContext | None:
    """系统 Python 缺少 CA 链，改用 certifi 的证书。"""
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def weather_code_cn(code) -> str:
    try:
        return WMO.get(int(code), f"未知({code})")
    except (TypeError, ValueError):
        return "--"


def find_city(query: str) -> list[dict]:
    """按名称/省份模糊搜索城市（子串匹配，空串返回全国列表前 20 个）。"""
    q = (query or "").strip()
    if not q:
        return [{"name": n, "province": p, "lat": lat, "lon": lon}
                for n, (lat, lon, p) in list(CITIES.items())[:20]]
    hits = [{"name": n, "province": p, "lat": lat, "lon": lon}
            for n, (lat, lon, p) in CITIES.items() if q in n or q in p]
    if not hits:  # 无命中时做首字匹配（如 "北" -> 北京）
        hits = [{"name": n, "province": p, "lat": lat, "lon": lon}
                for n, (lat, lon, p) in CITIES.items() if n.startswith(q)]
    return hits


def forecast(city: str, days: int = DAYS) -> dict:
    """某城市未来 days 天预报。

    返回 {city, province, days: [{date, weekday, code, desc, tmax, tmin, precip_prob}],
          updated}；城市不存在抛 ValueError。
    """
    city = (city or "").strip()
    if city not in CITIES:
        raise ValueError(f"暂不支持城市「{city}」，支持城市见 find_cities")
    lat, lon, prov = CITIES[city]
    days = max(1, min(int(days), 16))
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                 "precipitation_probability_max",
        "timezone": "Asia/Shanghai", "forecast_days": days})
    data = _fetch_json(f"{FORECAST_API}?{params}")
    if "daily" not in data:
        raise RuntimeError(f"接口异常：{json.dumps(data, ensure_ascii=False)[:200]}")
    d = data["daily"]
    wk = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    out = []
    for i, date in enumerate(d["time"]):
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            weekday = "今天" if i == 0 else ("明天" if i == 1 else wk[dt.weekday()])
        except ValueError:
            dt, weekday = None, ""
        code = d["weather_code"][i] if i < len(d["weather_code"]) else None
        out.append({
            "date": date,
            "weekday": weekday,
            "code": code,
            "desc": weather_code_cn(code),
            "tmax": d["temperature_2m_max"][i] if i < len(d["temperature_2m_max"]) else None,
            "tmin": d["temperature_2m_min"][i] if i < len(d["temperature_2m_min"]) else None,
            "precip_prob": (d["precipitation_probability_max"][i]
                            if "precipitation_probability_max" in d
                            and i < len(d["precipitation_probability_max"]) else None),
        })
    return {"city": city, "province": prov, "days": out,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def multi_forecast(cities: list[str], days: int = DAYS) -> list[dict]:
    """多城市预报（逐个请求，容错跳过失败城市）。"""
    res = []
    for c in cities:
        try:
            res.append(forecast(c, days))
        except Exception:
            continue
    return res


def cities_info() -> dict:
    """城市列表（名称+省份）供页面下拉候选。"""
    return {"cities": [{"name": n, "province": p} for n, (_, _, p) in CITIES.items()]}


if __name__ == "__main__":
    for city in ("北京", "上海", "广州", "杭州"):
        f = forecast(city)
        d = f["days"][0]
        print(f"{city}({f['province']}) {d['date']} {d['weekday']} {d['desc']} "
              f"{d['tmin']}~{d['tmax']}℃ 降水{d['precip_prob']}%")
        # 一周概览
        week = [f"{x['date'][5:]}{x['desc']}{x['tmin']}/{x['tmax']}" for x in f["days"][:5]]
        print("  ", " | ".join(week))
    print("find_city('广'):", [c["name"] for c in find_city("广")])