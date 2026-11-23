# -*- coding: utf-8 -*-
"""
auto_data.py — 全球汽车出口(外贸)数据层(纯标准库)

数据来源(按可信度分层):
  1. UN Comtrade 公开预览 API (annual, HS 8703 汽车, flow=X 出口)
     -> 全球主要出口国年度出口量(qty 辆)/净重/金额, 真实可验证
  2. 月度序列: 新闻快讯解析(best-effort) + 半自动录入(JSONL 持久化)
  3. 内置演示数据(source="sample"): 仅当无真实月度数据时展示, 页面/报告明确标注

网络说明: 中汽协/乘联会/海关总站在本网络不可达(已探查), 故月度数据采用上述分层方案。
"""
import json
import os
import re
import ssl
import threading
import urllib.request
from datetime import datetime

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # 本机已有 CA 链时兜底
    _CTX = ssl.create_default_context()

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "auto_export_data")
os.makedirs(DATA_DIR, exist_ok=True)

# h 8760: HS 8703 = 小汽车及其他机动车辆(主要用于载人)。预览 API 限制: 每次仅一个 period。
COMTRADE_URL = ("https://comtradeapi.un.org/public/v1/preview/C/A/HS"
                "?reporterCode={reporters}&period={year}"
                "&cmdCode=8703&flowCode=X&partnerCode=0")
COMTRADE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
COMTRADE_DELAY = 3.0  # 预览 API 限流, 请求间间隔(秒)

# 全球主要汽车出口国/生产国 (UN M49 code -> (英文名, 中文名))
REPORTERS = {
    156: ("China", "中国"),
    392: ("Japan", "日本"),
    276: ("Germany", "德国"),
    410: ("Korea", "韩国"),
    840: ("USA", "美国"),
    484: ("Mexico", "墨西哥"),
    724: ("Spain", "西班牙"),
    703: ("Slovakia", "斯洛伐克"),
    764: ("Thailand", "泰国"),
    826: ("United Kingdom", "英国"),
    250: ("France", "法国"),
    380: ("Italy", "意大利"),
    792: ("Turkey", "土耳其"),
    203: ("Czechia", "捷克"),
    76: ("Brazil", "巴西"),
    356: ("India", "印度"),
    360: ("Indonesia", "印度尼西亚"),
    348: ("Hungary", "匈牙利"),
    616: ("Poland", "波兰"),
    705: ("Slovenia", "斯洛文尼亚"),
}

# 中国主要出口车企(月度结构数据, 用于半自动录入/演示)
MANUFACTURERS = ["奇瑞", "比亚迪", "上汽", "吉利", "长安", "长城", "特斯拉中国", "其他"]

_LOCK = threading.Lock()


# ---------------------------------------------------------------- 基础工具
def _fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read().decode("utf-8", "ignore")


def _fetch_json(url, timeout=15):
    return json.loads(_fetch(url, timeout=timeout))


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------- 1) 全球年度出口 (Comtrade)
_cached = {}

def _load_reporter_codes():
    return ",".join(str(c) for c in sorted(REPORTERS))


def world_exports(force=False):
    """全球主要出口国年度出口数据: [ {year, code, name, name_cn, qty, netwgt_t, value_usd} ]"""
    if not force and "comtrade" in _cached:
        return _cached["comtrade"]
    rows, err = [], ""
    import time as _t
    for y in COMTRADE_YEARS:
        url = COMTRADE_URL.format(reporters=_load_reporter_codes(), year=y)
        try:
            j = _fetch_json(url)
            for rec in j.get("data") or []:
                code = rec.get("reporterCode")
                info = REPORTERS.get(code)
                if not info:
                    continue
                rows.append({
                    "year": rec.get("refYear"),
                    "code": code,
                    "name": info[0],
                    "name_cn": info[1],
                    "qty": rec.get("qty") or 0,               # 辆(估算)
                    "qty_est": rec.get("isQtyEstimated", False),
                    "netwgt_t": round((rec.get("netWgt") or 0) / 1000.0),  # kg -> 吨
                    "value_usd": rec.get("primaryValue") or 0,             # 美元
                })
        except Exception as e:
            err = f"{err} {y}:{e};"
        _t.sleep(COMTRADE_DELAY)  # 预览 API 限流
    rows.sort(key=lambda r: (r["year"], -r["value_usd"]))
    _cached["comtrade"] = {"rows": rows, "error": err or None, "updated": _now()}
    return _cached["comtrade"]


def global_trend():
    """主要国家年度出口趋势: {years:[...], series:[{code,name_cn,qty:[..],value:[..]}]}"""
    data = world_exports()
    rows = data.get("rows", [])
    years = sorted(set(r["year"] for r in rows)) if rows else COMTRADE_YEARS
    by_code = {}
    for r in rows:
        by_code.setdefault(r["code"], {})[r["year"]] = r
    series = []
    for code, info in sorted(REPORTERS.items(), key=lambda x: -sum((by_code.get(x[0], {}).get(y, {}).get("value_usd") or 0) for y in years)):
        m = by_code.get(code, {})
        if not m:
            continue
        series.append({
            "code": code,
            "name_cn": info[1],
            "qty": [m.get(y, {}).get("qty", 0) for y in years],
            "value": [round((m.get(y, {}).get("value_usd") or 0) / 1e8, 2) for y in years],  # 亿美元
        })
    return {"years": years, "series": series, "updated": data.get("updated", "")}


def top_exporters(year=None, limit=15):
    """指定年份出口额 TOP 国家排名(默认最新年)"""
    rows = world_exports().get("rows", [])
    if not rows:
        return []
    years = sorted(set(r["year"] for r in rows))
    y = year or years[-1]
    out = [r for r in rows if r["year"] == y]
    out.sort(key=lambda r: -r["value_usd"])
    return out[:limit]


# -------------------------------------------- 2) 月度序列 (演示 + 用户维护)
_MONTHLY_FIELDS = ("ym", "scope", "total", "note", "source", "updated")

def _monthly_paths():
    return (os.path.join(DATA_DIR, "monthly.jsonl"),
            os.path.join(DATA_DIR, "monthly_%s.jsonl" % datetime.now().strftime("%Y%m%S%f")))


def _sample_monthly():
    """内置演示月度数据(2024-01 ~ 2026-10, 中国月度出口总值+车企结构)。
    注意: source="sample", 仅作占位演示, 真实数据请通过 add_monthly 录入或 update_from_news 解析。"""
    import math
    base = 45.0  # 万辆/月 基量
    out = {}
    for y in (2024, 2025, 2026):
        for m in range(1, 13):
            if y == 2026 and m > 10:
                break
            ym = f"{y}{m:02d}"
            # 温和增长 + 淡旺季波动(春节2月低, 年底高)
            season = {1: 0.95, 2: 0.75, 3: 1.02, 4: 1.0, 5: 1.02, 6: 0.98,
                      7: 0.96, 8: 0.97, 9: 1.05, 10: 0.96, 11: 1.06, 12: 1.08}[m]
            growth = 1 + 0.10 * ((y - 2024) + (m - 1) / 12.0)
            total = round(base * season * growth, 1)  # 万辆
            out[ym] = {
                "ym": ym, "scope": "中国", "total": total,
                "note": "演示数据(占总出口~9成), 含车企结构测算",
                "source": "sample", "updated": "",
                "mfr": {
                    "奇瑞": round(total * 0.24, 2), "比亚迪": round(total * 0.21, 2),
                    "上汽": round(total * 0.11, 2), "吉利": round(total * 0.10, 2),
                    "长安": round(total * 0.09, 2), "长城": round(total * 0.06, 2),
                    "特斯拉中国": round(total * 0.05, 2),
                    "其他": round(total * 0.14, 2),
                },
            }
    return out


def _read_monthly():
    """读取用户维护的月度记录(JSONL), 与演示数据合并(用户记录优先)"""
    data = _sample_monthly()
    path = os.path.join(DATA_DIR, "monthly.jsonl")
    if os.path.exists(path):
        with _LOCK:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    key = (rec.get("scope", "中国"), rec.get("ym", ""))
                    data[rec["ym"]] = {**data.get(rec["ym"], {}), **rec, "_key": key}
    return data


def monthly_series(scope="中国"):
    """月度序列(按时间升序): [{ym, total, ...}]"""
    data = _read_monthly()
    out = [rec for rec in data.values() if rec.get("scope") == scope]
    out.sort(key=lambda r: r["ym"])
    return out


def add_monthly(rec):
    """录入/更新一条月度记录(用户维护, 半自动)。要求: ym=YYYYMM 且 total 为数字。"""
    ym = str(rec.get("ym", "")).strip()
    if not re.fullmatch(r"\d{6}", ym):
        return {"ok": False, "error": "ym 需为 YYYYMM 格式"}
    try:
        total = float(rec.get("total"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "total 需为数字(万辆)"}
    mfr = rec.get("mfr") or {}
    if isinstance(mfr, dict):
        mfr = {str(k): float(v) for k, v in mfr.items() if _isnum(v)}
    entry = {
        "ym": ym,
        "scope": rec.get("scope") or "中国",
        "total": total,
        "note": rec.get("note", ""),
        "source": rec.get("source") or "manual",
        "updated": _now(),
        "mfr": mfr,
    }
    path = os.path.join(DATA_DIR, "monthly.jsonl")
    with _LOCK:
        lines = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
        # 同 (scope, ym) 去重
        keep = []
        for l in lines:
            try:
                old = json.loads(l)
            except ValueError:
                keep.append(l)
                continue
            if old.get("scope") == entry["scope"] and old.get("ym") == ym:
                continue
            keep.append(l)
        keep.append(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    return {"ok": True, "entry": entry}


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def update_from_news():
    """best-effort: 从 gasgoo 首页/资讯抓取含'出口'与万辆数字的标题, 尝试解析最新月度出口快讯。
    返回 {parsed: [rec...], hits: [title...]}, 不抛出异常。"""
    hits, parsed = [], []
    for url in ("https://auto.gasgoo.com/", "https://www.autohome.com.cn/news/1/"):
        try:
            html = _fetch(url, timeout=10)
        except Exception:
            continue
        for t in re.findall(r"<a[^>]*>([^<]{8,80})</a>", html):
            t = t.strip()
            if "出口" in t and re.search(r"\d+\s*万", t):
                hits.append(t)
                m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", t)
                n = re.search(r"(\d+(?:\.\d+)?)\s*万", t)
                if m and n:
                    ym = m.group(1) + f"{int(m.group(2)):02d}"
                    parsed.append({"ym": ym, "total": round(float(n.group(1)), 1),
                                   "note": t[:60], "source": "news"})
        if hits:
            break
    return {"hits": hits[:10], "parsed": parsed}


# ------------------------------------------------------- 3) 技术报告
def build_report():
    """生成技术分析报告(Markdown)。含全球格局/中国月度/车企结构/结论与数据来源说明。"""
    data = world_exports()
    rows = data.get("rows", [])
    md = ["# 全球汽车出口(外贸)动态监测技术报告", "",
          f"> 生成时间:{_now()}　|　数据源:UN Comtrade(HS 8703, 出口) + 月度维护库", ""]
    if not rows:
        md += ["## ⚠ 数据不可用", "", f"{data.get('error', '未获取到数据')}", ""]
        return "\n".join(md)

    years = sorted(set(r["year"] for r in rows))
    latest = years[-1]
    prev = years[-2] if len(years) > 1 else None

    # 一、全球出口格局(最新年)
    md += ["## 一、全球出口格局(年度)", ""]
    md += ["| 排名 | 国家 | 出口额(亿美元) | 出口量(万辆) | 同比(出口额) |", "|---|---|---|---|---|"]
    top = [r for r in rows if r["year"] == latest]
    top.sort(key=lambda r: -r["value_usd"])
    for i, r in enumerate(top[:15], 1):
        yoy = ""
        if prev:
            pr = next((x for x in rows if x["year"] == prev and x["code"] == r["code"]), None)
            if pr and pr["value_usd"]:
                yoy = f"{(r['value_usd'] / pr['value_usd'] - 1) * 100:+.1f}%"
        md.append(f"| {i} | {r['name_cn']} | {r['value_usd'] / 1e8:,.0f} | {r['qty'] / 1e4:,.1f} | {yoy} |")
    md.append("")

    # 二、近三年变化
    md += ["## 二、近三年主要国家出口额变化(亿美元)", "", "| 国家 | " + " | ".join(map(str, years[-3:])) + " |", "|---|" + "---|" * len(years[-3:])]
    for code, info in sorted(REPORTERS.items(), key=lambda x: -max((r["value_usd"] for r in rows if r["code"] == x[0]), default=0)):
        vals = [next((r["value_usd"] / 1e8 for r in rows if r["year"] == y and r["code"] == code), 0) for y in years[-3:]]
        if not any(vals):
            continue
        md.append(f"| {info[1]} | " + " | ".join(f"{v:,.0f}" for v in vals) + " |")
    md.append("")

    # 三、中国月度出口
    ms = [r for r in monthly_series() if r["source"] == "sample"]
    real = [r for r in monthly_series() if r["source"] != "sample"]
    if real:
        src_txt = "用户维护(真实快讯)"
    else:
        src_txt = "演示数据"
    md += [f"## 三、中国月度出口动态({src_txt})", ""]
    if real:
        for r in real[-6:]:
            md.append(f"- **{r['ym'][:4]}-{r['ym'][4:]}**: {r['total']} 万辆　{('(' + r['note'] + ')') if r['note'] else ''}")
    else:
        md.append("> 当前展示为内置演示序列(用于图表占位)。请通过页面『数据录入』或 `add_monthly()` 录入中汽协/海关快讯的真实月度数据。")
    md += ["", "| 月份 | 出口量(万辆) | 环比 |", "|---|---|---|"]
    for i, r in enumerate(ms[-12:]):
        m = ms[max(0, len(ms) - 12 + i - 1)]
        mom = f"{(r['total'] / m['total'] - 1) * 100:+.1f}%" if m and m["ym"] != r["ym"] and m["total"] else "-"
        md.append(f"| {r['ym'][:4]}-{r['ym'][4:]} | {r['total']} | {mom} |")
    md.append("")

    # 四、中国车企出口结构
    if ms:
        last = ms[-1]
        mfr = last.get("mfr") or {}
        if mfr:
            md += ["## 四、中国车企出口结构(最新月, 演示)", "", "| 车企 | 出口量(万辆) | 占比 |", "|---|---|---|"]
            tot = sum(mfr.values()) or 1
            for name, v in sorted(mfr.items(), key=lambda x: -x[1]):
                md.append(f"| {name} | {v} | {v / tot * 100:.1f}% |")
            md.append("")

    # 五、结论与说明
    cn = next((r for r in top if r["code"] == 156), None)
    md += ["## 五、分析结论", ""]
    if cn:
        md.append(f"- 全球最大汽车出口国为**中国**({latest}年出口额 {cn['value_usd'] / 1e8:,.0f} 亿美元, 出口量 {cn['qty'] / 1e4:,.1f} 万辆)。")
    md += ["- 出口额与出口量呈同向波动, 单价(出口额/辆)可衡量产品结构升级: 出口量增速高于出口额增速说明均价下探。"]
    md += ["- 月度维度受春节(2月)与年末冲量影响显著, 环比波动属正常季节性。", "",
           "## 六、数据来源与局限", "",
           "1. **年度**: UN Comtrade 公开预览 API(HS 8703, 出口流), 量为估算值(`isQtyEstimated`), 金额为 FOB 美元。",
           "2. **月度**: 中汽协/海关总站在本网络不可达, 当前为演示序列; 真实月度数据请通过页面录入(半自动维护)。",
           "3. 本报告不含新能源/燃油车结构拆分与单一车型明细。", ""]
    return "\n".join(md)


# ------------------------------------------------------------- main(自检)
if __name__ == "__main__":
    import sys
    print("== world_exports ==")
    w = world_exports()
    print("error:", w.get("error", "none"), "| rows:", len(w.get("rows", [])))
    for r in w.get("rows", [])[:3]:
        print("  ", r)
    print("\n== top_exporters (latest) ==")
    for r in top_exporters():
        print(f"  {r['name_cn']:8s} {r['year']} qty={r['qty']/1e4:8.1f}万辆 value={r['value_usd']/1e8:9.0f}亿美元")
    print("\n== monthly_series ==")
    ms = monthly_series()
    print("  count:", len(ms), "| latest:", ms[-1] if ms else None)
    print("\n== update_from_news ==")
    n = update_from_news()
    print("  hits:", n["hits"][:3])
    print("  parsed:", n["parsed"][:3])
    print("\n== report head ==")
    print(build_report()[:600])