#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股票助手数据层：A股 / 港股 / 美股 行情、涨幅榜与科技股观察清单

数据源（本网络实测可用，2026-11-13）：
  - 新浪行情 hq.sinajs.cn（GBK 编码，需 Referer）：
      A股 list=sh600519,sz300750 ｜ 港股 rt_hk00700,rt_hkHSI ｜ 美股 gb_aapl,gb_$dji
  - 新浪榜单 Market_Center.getHQNodeData（A股 hs_a 按涨幅排序）、getHKStockData（港股）
  - 雅虎 query1.finance.yahoo.com v1/finance/screener day_gainers（美股涨幅榜）
  - 注意：东方财富 push2 的 clist/kline 路径在本网络被拦（000），故榜单走新浪/雅虎

字段约定：价格单位元/港元/美元；涨跌幅 %；成交额 A股/港股为对应货币，美股为美元。
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

SINA_QUOTE = "https://hq.sinajs.cn/list="
SINA_RANK_A = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               "Market_Center.getHQNodeData")
SINA_RANK_HK = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                "Market_Center.getHKStockData")
YAHOO_SCREENER = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                  "?formatted=true&scrIds=day_gainers&count={n}")

INDEX_CODES = [("s_sh000001", "上证指数"), ("s_sz399001", "深证成指"),
               ("s_sz399006", "创业板指"), ("rt_hkHSI", "恒生指数"),
               ("gb_$dji", "道琼斯"), ("gb_$ixic", "纳斯达克"), ("gb_$inx", "标普500")]

# 科技股观察清单（各市场主流科技龙头；代码可直接用于对应行情接口）
WATCHLIST = {
    "a": [("sh688981", "中芯国际"), ("sz002371", "北方华创"), ("sh603501", "韦尔股份"),
          ("sh688041", "海光信息"), ("sh688256", "寒武纪"), ("sh688111", "金山办公"),
          ("sz002475", "立讯精密"), ("sh601138", "工业富联"), ("sz300308", "中际旭创"),
          ("sh688008", "澜起科技"), ("sh603986", "兆易创新"), ("sz002463", "沪电股份"),
          ("sh603160", "汇顶科技"), ("sh600588", "用友网络"), ("sh688012", "中微公司")],
    "hk": [("00700", "腾讯控股"), ("09988", "阿里巴巴-W"), ("03690", "美团-W"),
           ("01810", "小米集团-W"), ("01024", "快手-W"), ("09618", "京东集团-SW"),
           ("00981", "中芯国际"), ("01347", "华虹半导体"), ("03888", "金山软件"),
           ("00992", "联想集团"), ("09888", "百度集团-SW"), ("09999", "网易-S")],
    "us": [("NVDA", "英伟达"), ("AAPL", "苹果"), ("MSFT", "微软"), ("GOOGL", "谷歌"),
           ("META", "Meta"), ("AMZN", "亚马逊"), ("TSLA", "特斯拉"), ("AVGO", "博通"),
           ("AMD", "超威半导体"), ("INTC", "英特尔"), ("TSM", "台积电"), ("QCOM", "高通"),
           ("ORCL", "甲骨文"), ("CRM", "赛富时"), ("MU", "美光科技")],
}
MARKET_NAMES = {"a": "A股", "hk": "港股", "us": "美股"}


# --------------------------------------------------------------------------- #
# 底层请求
# --------------------------------------------------------------------------- #

def _ssl_ctx() -> ssl.SSLContext | None:
    """系统 Python 缺少 CA 链，改用 certifi 的证书；装不了 certifi 就退回默认。"""
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _fetch(url: str, referer: str = "https://finance.sina.com.cn") -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer": referer,
    })
    ctx = _ssl_ctx()
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        raw = resp.read()
    return raw.decode("gbk", errors="replace") if "hq.sinajs" in url else raw.decode("utf-8", errors="replace")


def _fetch_json(url: str) -> list:
    text = _fetch(url)
    text = text.strip()
    if not text or text == "null":
        return []
    return json.loads(text)


def _pct(price: float, prev: float):
    if prev:
        return round((price - prev) / prev * 100, 2)
    return None


# --------------------------------------------------------------------------- #
# 行情
# --------------------------------------------------------------------------- #

def quote_a(codes: list[str]) -> list[dict]:
    """A股批量行情。格式：名称,今开,昨收,现价,最高,最低,买一,卖一,量,额,日期,时间"""
    out = []
    text = _fetch(SINA_QUOTE + ",".join(codes))
    for line in text.splitlines():
        if "=" not in line or '""' in line:
            continue
        code = line.split("hq_str_", 1)[1].split("=", 1)[0]
        f = line.split('"')[1].split(",")
        if len(f) < 10:
            continue
        try:
            price, prev = float(f[3]), float(f[2])
        except ValueError:
            continue
        out.append({"code": code, "name": f[0], "price": price, "prev": prev,
                    "change": round(price - prev, 3), "change_pct": _pct(price, prev),
                    "open": float(f[1]), "high": float(f[4]), "low": float(f[5]),
                    "volume": float(f[8]), "amount": float(f[9]),
                    "date": f[10], "time": f[11]})
    return out


def quote_hk(codes: list[str]) -> list[dict]:
    """港股批量行情。格式：EN名,CN名,开,昨收,高,低,现价,涨跌,涨跌%,买,卖,额,量,..."""
    out = []
    text = _fetch(SINA_QUOTE + ",".join("rt_hk" + c for c in codes))
    for line in text.splitlines():
        if "=" not in line or '""' in line:
            continue
        code = line.split("hq_str_rt_hk", 1)[1].split("=", 1)[0]
        f = line.split('"')[1].split(",")
        if len(f) < 13:
            continue
        try:
            price = float(f[6])
        except ValueError:
            continue
        out.append({"code": code, "name": f[1], "price": price,
                    "prev": float(f[3]), "open": float(f[2]), "high": float(f[4]),
                    "low": float(f[5]), "change": float(f[7]) if f[7] else None,
                    "change_pct": float(f[8]) if f[8] else None,
                    "amount": float(f[11]), "volume": float(f[12]),
                    "date": f[17] if len(f) > 17 else ""})
    return out


def quote_us(codes: list[str]) -> list[dict]:
    """美股批量行情。格式：名称,现价,涨跌%,时间,涨跌,开,高,低,52高,52低,量,额,市值,PE,..."""
    out = []
    text = _fetch(SINA_QUOTE + ",".join("gb_" + c.lower() for c in codes))
    for line in text.splitlines():
        if "=" not in line or '""' in line:
            continue
        code = line.split("hq_str_gb_", 1)[1].split("=", 1)[0]
        f = line.split('"')[1].split(",")
        if len(f) < 7:
            continue
        try:
            price = float(f[1])
        except ValueError:
            continue
        out.append({"code": code.upper(), "name": f[0], "price": price,
                    "change": float(f[4]) if f[4] else None,
                    "change_pct": float(f[2]) if f[2] else None,
                    "open": float(f[5]) if f[5] else None,
                    "high": float(f[6]) if f[6] else None,
                    "low": float(f[7]) if len(f) > 7 and f[7] else None,
                    "high52": float(f[8]) if len(f) > 8 and f[8] else None,
                    "low52": float(f[9]) if len(f) > 9 and f[9] else None,
                    "volume": float(f[10]) if len(f) > 10 and f[10] else None,
                    "mktcap": float(f[12]) if len(f) > 12 and f[12] else None,
                    "datetime": f[3]})
    return out


def indices() -> list[dict]:
    """主要指数快照（RN/恒指/道指/纳指/标普）。"""
    text = _fetch(SINA_QUOTE + ",".join(c for c, _ in INDEX_CODES))
    out = []
    for line in text.splitlines():
        if "=" not in line or '""' in line:
            continue
        key = line.split("hq_str_", 1)[1].split("=", 1)[0]
        f = line.split('"')[1].split(",")
        name = dict(INDEX_CODES).get(key, key)
        try:
            if key.startswith("s_"):
                # 简版：名称,现价,涨跌,涨跌%,量,额
                out.append({"name": name, "price": float(f[1]),
                            "change": float(f[2]) if f[2] else None,
                            "change_pct": float(f[3]) if f[3] else None})
            elif key.startswith("rt_hk"):
                # 恒指：...,开,昨收,高,低,现价,涨跌,涨跌%,...
                out.append({"name": name, "price": float(f[6]),
                            "change": float(f[7]) if f[7] else None,
                            "change_pct": float(f[8]) if f[8] else None})
            else:
                # 美股指数：名称,现价,涨跌%,时间,涨跌,...
                out.append({"name": name, "price": float(f[1]),
                            "change": float(f[4]) if f[4] else None,
                            "change_pct": float(f[2]) if f[2] else None})
        except (ValueError, IndexError):
            continue
    return out


# --------------------------------------------------------------------------- #
# 榜单
# --------------------------------------------------------------------------- #

def top_gainers(market: str, n: int = 10) -> list[dict]:
    """各市场当日涨幅榜。"""
    n = max(1, min(int(n), 50))
    if market == "a":
        rows = _fetch_json(SINA_RANK_A + "?" + urllib.parse.urlencode(
            {"page": 1, "num": n, "sort": "changepercent", "asc": 0, "node": "hs_a"}))
        return [{"code": r.get("code", ""), "name": r.get("name", ""),
                 "price": r.get("trade"), "change_pct": r.get("changepercent"),
                 "amount": r.get("amount"), "turnover": r.get("turnoverratio"),
                 "mktcap": r.get("mktcap")} for r in rows][:n]
    if market == "hk":
        rows = _fetch_json(SINA_RANK_HK + "?" + urllib.parse.urlencode(
            {"page": 1, "num": n, "sort": "changepercent", "asc": 0, "node": "qbgg_hk"}))
        return [{"code": r.get("symbol", ""), "name": r.get("name", ""),
                 "price": _num(r.get("lasttrade")), "change_pct": _num(r.get("changepercent")),
                 "amount": _num(r.get("amount")), "turnover": _num(r.get("turnoverratio"))}
                for r in rows][:n]
    if market == "us":
        text = _fetch(YAHOO_SCREENER.format(n=n), referer="https://finance.yahoo.com/")
        data = json.loads(text)
        quotes = (data.get("finance", {}).get("result") or [{}])[0].get("quotes") or []
        # Yahoo formatted=true 返回 {"raw": x, "fmt": "..."}，统一拍平成数值
        return [{"code": q.get("symbol", ""), "name": q.get("shortName") or q.get("longName", ""),
                 "price": _num(q.get("regularMarketPrice")),
                 "change_pct": _num(q.get("regularMarketChangePercent")),
                 "volume": _num(q.get("regularMarketVolume"))} for q in quotes][:n]
    return []


def _num(v):
    """Yahoo formatted 字段可能是 {"raw":..,"fmt":..}，统一取 raw 数值。"""
    if isinstance(v, dict):
        v = v.get("raw")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 科技股观察清单
# --------------------------------------------------------------------------- #

def tech_watchlist(market: str) -> list[dict]:
    """科技股观察清单行情（按清单代码批量拉取，附涨跌幅）。"""
    pairs = WATCHLIST.get(market, [])
    if not pairs:
        return []
    if market == "a":
        quotes = {q["code"]: q for q in quote_a([c for c, _ in pairs])}
    elif market == "hk":
        quotes = {q["code"]: q for q in quote_hk([c for c, _ in pairs])}
    else:
        quotes = {q["code"]: q for q in quote_us([c for c, _ in pairs])}
    out = []
    for code, name in pairs:
        q = quotes.get(code)
        if not q or not q.get("price"):
            continue
        out.append({"code": code, "name": name, "price": q["price"],
                    "change_pct": q.get("change_pct"), "change": q.get("change")})
    out.sort(key=lambda x: (x["change_pct"] is None, -(x["change_pct"] or 0)))
    return out


def market_overview() -> dict:
    """全市场概览：指数 + 各市场涨幅榜前 5。"""
    return {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indices": indices(),
        "top": {m: top_gainers(m, 5) for m in ("a", "hk", "us")},
        "watchlist": {m: tech_watchlist(m) for m in ("a", "hk", "us")},
    }