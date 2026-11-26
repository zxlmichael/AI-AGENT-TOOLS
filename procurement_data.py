#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""政府招采地图需求监测 · 数据层：多源抓取 → 关键词筛选 → SQLite 增量入库 → 统计派生

数据源（全部免费公开，多源容错）：
- 中国政府采购网 search.ccgp.gov.cn（全国招标公告 + 采购意向；有 IP 频控，单轮限页 + 详情页节流 + 退避）
- 北京市公共资源交易服务平台 ggzyfw.beijing.gov.cn（招标公告 + 采购需求公示，纯静态列表可解析）
- 上海市公共资源交易平台 / 广东省政府采购网：JS 渲染入口预留（playwright 子进程降级），
  当前网络环境被反爬拦截时记入 errors 不阻塞其他源

统计输出（供 ECharts 可视化与问答/导出）：
- by_region / by_type / by_source / per_day / top_keywords / budget_range
"""

from __future__ import annotations

import html
import os
import re
import sqlite3
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import certifi

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "procurement_data"
DB_PATH = DATA_DIR / "procurement.db"

try:
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = None

_CACHE_TTL = 1800          # 抓取间隔 30 分钟（政务网站低频率礼貌访问）
_FETCH_GAP = 1.0           # 源间礼貌间隔（秒）
_DETAIL_GAP = 0.6          # 详情页间隔（秒）
_WINDOW_DAYS = 30          # 近一个月窗口
_cache = {"lock": threading.Lock(), "t": 0.0, "meta": {}}

# 地图需求关键词（标题/命中即候选；"影像"必须限遥感/航拍类，避免医学影像误报）
KEYWORDS = [
    "地图", "GIS", "地理信息", "测绘", "遥感", "北斗", "一张图", "国土空间",
    "三维实景", "实景三维", "智慧城市", "河湖划界", "确权", "航拍",
    "遥感影像", "卫星影像", "正射影像", "影像数据", "影像图",
    "坐标", "空间数据", "geospatial", "mapping", "地形图", "地籍", "倾斜摄影",
]

# 行政区简称 → 省级地区（标题/正文匹配用）
_REGIONS = ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
            "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
            "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
            "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "全国"]

# ccgp 搜索接口地区筛选参数值(全称, 入库 region 简称)
CCGP_PROVINCES = [
    ("北京市", "北京"), ("天津市", "天津"), ("河北省", "河北"), ("山西省", "山西"),
    ("内蒙古自治区", "内蒙古"), ("辽宁省", "辽宁"), ("吉林省", "吉林"), ("黑龙江省", "黑龙江"),
    ("上海市", "上海"), ("江苏省", "江苏"), ("浙江省", "浙江"), ("安徽省", "安徽"),
    ("福建省", "福建"), ("江西省", "江西"), ("山东省", "山东"), ("河南省", "河南"),
    ("湖北省", "湖北"), ("湖南省", "湖南"), ("广东省", "广东"), ("广西壮族自治区", "广西"),
    ("海南省", "海南"), ("重庆市", "重庆"), ("四川省", "四川"), ("贵州省", "贵州"),
    ("云南省", "云南"), ("西藏自治区", "西藏"), ("陕西省", "陕西"), ("甘肃省", "甘肃"),
    ("青海省", "青海"), ("宁夏回族自治区", "宁夏"), ("新疆维吾尔自治区", "新疆"),
]
_CCGP_GAP = 3.0        # 省级间礼貌间隔（秒），降低触发 IP 频控的概率
_CCGP_GROUPS = 4       # 增量轮换组数（每轮一组约 8 省，约 4 天轮完）

_BLOCK_HINTS = ("访问过于频繁", "频繁访问", "请稍后再试", "too many requests", "verify")

SOURCES_META = [
    ("ccgp", "中国政府采购网", "招标公告/采购意向"),
    ("bj", "北京公共资源交易", "招标公告/采购需求"),
    ("sh", "上海公共资源交易", "适配中"),
    ("gd", "广东省政府采购", "适配中"),
]


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            pub_type TEXT,
            pub_date TEXT,
            region TEXT,
            buyer TEXT,
            budget REAL,
            keywords TEXT,
            desc TEXT,
            first_seen REAL
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_region ON projects(region)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pub ON projects(pub_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen ON projects(first_seen)")
    conn.commit()


# ---------- 抓取基础 ----------

def _fetch(url: str, timeout: int = 15, headers: dict | None = None) -> bytes:
    hdrs = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read(1200000)


def _strip_html(s: str) -> str:
    s = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", s or "", flags=re.S)
    s = re.sub(r"<[^>]+>", "\n", s)
    s = re.sub(r"[\t\xa0 ]+", " ", s)
    return html.unescape(re.sub(r"\n\s*\n+", "\n", s)).strip()


def _clean_desc(body: str, title: str = "") -> str:
    """详情正文 → 摘要：去 BOM/导航碎行，从标题或内容锚点起截取。"""
    s = (body or "").replace("\ufeff", "")
    pos = -1
    if title:
        p = s.find(title)
        if p != -1:
            pos = p
    if pos == -1:
        for a in ("项目概况", "项目基本情况", "采购需求", "一、项目"):
            p = s.find(a)
            if p != -1 and (pos == -1 or p < pos):
                pos = p
    if pos > 0:
        s = s[pos:]
    lines = []
    for ln in s.split("\n"):
        ln = ln.strip()
        if len(ln) < 4 or ">" in ln or ln in ("首页", "公告公示", "政府采购"):
            continue
        lines.append(ln)
    return re.sub(r"\s{2,}", " ", " ".join(lines))[:280].strip()


def _is_blocked(text: str) -> bool:
    return any(h in text for h in _BLOCK_HINTS)


def _norm_date(s: str) -> str | None:
    """解析 '2026-11-26' / '2026年11月26日' 等 → ISO；失败 None。"""
    s = (s or "").strip()
    m = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            return None
    return None


def _extract_budget(text: str) -> float | None:
    m = re.search(r"预算金额[：:\s]*([\d,]+(?:\.\d+)?)\s*(万元|万)", text)
    if m:
        v = float(m.group(1).replace(",", ""))
        return v if m.group(2) == "万元" else v / 10000
    m = re.search(r"预算[：:\s]*([\d,]+(?:\.\d+)?)\s*(万元|万|元)", text)
    if m:
        v = float(m.group(1).replace(",", ""))
        unit = m.group(2)
        return v if unit in ("万元", "万") else v / 10000
    return None


def _judge_region(title: str, desc: str) -> str:
    for r in _REGIONS:
        if r in title:
            return r
    for r in _REGIONS:
        if r in desc:
            return r
    return ""


def match_keywords(text: str) -> list[str]:
    return [k for k in KEYWORDS if k.lower() in text.lower()]


def _in_window(pub_date: str | None, first_seen: float) -> bool:
    now = time.time()
    if pub_date:
        d = datetime.strptime(pub_date, "%Y-%m-%d")
        if (datetime.now() - d).days > _WINDOW_DAYS:
            return False
        return True
    return now - first_seen < _WINDOW_DAYS * 86400


def _insert(conn: sqlite3.Connection, rec: dict) -> bool:
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO projects
               (link, title, source, pub_type, pub_date, region, buyer, budget,
                keywords, desc, first_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (rec["link"], rec["title"][:300], rec["source"], rec.get("pub_type"),
             rec.get("pub_date"), rec.get("region"), rec.get("buyer"), rec.get("budget"),
             ",".join(rec.get("keywords", [])), rec.get("desc", "")[:500],
             rec.get("first_seen", time.time())))
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error:
        conn.rollback()
        return False


# ---------- 源适配器：中国政府采购网 ----------

CCGP_BID_TYPES = [("1", "招标公告"), ("4", "采购意向")]   # bidType 映射见下（真值以探测为准）


def _ccgp_list_url(kw: str, page: int, bid_type: str, start: str, end: str,
                   province: str = "") -> str:
    q = urllib.parse.urlencode({
        "searchtype": "1", "page_index": page, "bidSort": "0", "pinMu": "0",
        "bidType": bid_type, "kw": kw, "start_time": start, "end_time": end,
        "timeType": "6", "province": province,
    })
    return f"https://search.ccgp.gov.cn/bxsearch?{q}"


def _ccgp_parse_list(text: str) -> list[tuple[str, str]]:
    """列表页 → [(详情URL, 标题)]。"""
    items = []
    for m in re.finditer(
            r'<a[^>]+href="(https?://\S*?ccgp\.gov\.cn/\S*?/\d{8}/\d+\.html|'
            r'https?://\S*?ccgp\.gov\.cn/\S*?bd_\d+\.htm)"[^>]*>(.*?)</a>',
            text, re.S):
        url, title = m.group(1), re.sub(r"<[^>]+>|\s+", " ", m.group(2)).strip()
        if url and title:
            items.append((html.unescape(url), html.unescape(title)))
    return items


def _ccgp_parse_detail(text: str, link: str, pub_type: str) -> dict | None:
    """详情页 → 统一记录。"""
    body = _strip_html(text)
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    heading = title.group(1).split("-")[0].strip() if title else ""
    m = re.search(r"发布时间[：:]\s*([\d\-/年]+)", body)
    pub = _norm_date(m.group(1)) if m else None
    region = _judge_region(body[:2000], body)
    buyer = ""
    mb = re.search(r"采购人[（(]?信息[）)]?[：:]\s*([^\n]{2,40})", body)
    if mb:
        buyer = mb.group(1).strip()
    if not buyer:
        mb = re.search(r"采购人[：:]\s*([^\n，。]{2,40})", body)
        if mb:
            buyer = mb.group(1).strip()
    budget = _extract_budget(body)
    kw = match_keywords(body[:3000] + " " + heading)
    return {"link": link, "title": heading or body[:60], "source": "ccgp",
            "pub_type": pub_type, "pub_date": pub, "region": region,
            "buyer": buyer, "budget": budget, "keywords": kw,
            "desc": _clean_desc(body, heading)}


def fetch_ccgp(kw: str = "地图", pages: int = 3, limit_details: int = 40,
               now_iso: str | None = None,
               provinces: list[tuple[str, str]] | None = None) -> tuple[list[dict], list[str]]:
    """抓 ccgp 搜索结果列表 + 详情。

    provinces=None → 全站检索；[(全称,简称), ...] → 按省滚动检索；
    单省频控只跳过该省；连续 3 省频控则终止本轮（避免把抓取预算耗在等待上）。
    """
    now = datetime.now()
    end = now.date().isoformat()
    start = (now - timedelta(days=_WINDOW_DAYS)).date().isoformat()
    items, errors, seen = [], [], set()
    blocked_run = 0
    scope = list(provinces) if provinces is not None else [("", "")]
    for prov_full, prov_short in scope:
        if blocked_run >= 3:
            errors.append("ccgp: 多省连续频控,本轮提前结束")
            break
        for bid_type, tname in CCGP_BID_TYPES:
            if blocked_run >= 3:
                break
            for page in range(1, pages + 1):
                try:
                    text = _fetch(_ccgp_list_url(kw, page, bid_type, start, end,
                                                 prov_full)).decode("utf-8", "ignore")
                    if _is_blocked(text):
                        errors.append(f"ccgp: 频控({prov_short or '全站'} {bid_type} p{page})")
                        blocked_run += 1
                        break
                    links = _ccgp_parse_list(text)
                    if not links:
                        break
                    for link, title in links:
                        if link in seen:
                            continue
                        seen.add(link)
                        if not match_keywords(title + " " + link):
                            continue            # 标题不含地图关键词，跳过详情
                        time.sleep(_DETAIL_GAP)
                        try:
                            det = _fetch(link).decode("utf-8", "ignore")
                            if _is_blocked(det):
                                errors.append("ccgp: 详情频控")
                                continue
                            rec = _ccgp_parse_detail(det, link, tname)
                            if rec and match_keywords(rec["title"] + " " + rec["desc"]):
                                if prov_short and not rec["region"]:
                                    rec["region"] = prov_short
                                items.append(rec)
                        except Exception as exc:
                            errors.append(f"ccgp 详情: {type(exc).__name__}")
                        if len(items) >= limit_details:
                            return items, errors
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        break                     # 频道没有该页 → 翻页到底
                    errors.append(f"ccgp {prov_short or '全站'} {bid_type} p{page}: HTTP{exc.code}")
                    break
                except Exception as exc:
                    errors.append(f"ccgp {prov_short or '全站'} {bid_type} p{page}: {type(exc).__name__}")
                    break
                time.sleep(_FETCH_GAP)
        time.sleep(_CCGP_GAP if prov_full else 0)
    if provinces and not items and not any("频控" in e for e in errors):
        errors.append("ccgp: 省份检索 0 结果且无频控,可能 province 参数未生效,"
                      "请运行 --probe 验证")
    return items, errors


def _ccgp_province_pick() -> list[tuple[str, str]]:
    """首跑（库空）全 31 省回填；之后按日期交错分 4 组轮换（每轮一组约 8 省），约 4 天轮完。"""
    conn = _connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    finally:
        conn.close()
    if n == 0:
        return list(CCGP_PROVINCES)
    day = int(datetime.now().strftime("%j"))          # 一年中的第几天
    groups = [CCGP_PROVINCES[i::_CCGP_GROUPS] for i in range(_CCGP_GROUPS)]
    return groups[day % _CCGP_GROUPS]


# ---------- 源适配器：北京公共资源交易 ----------

BJ_CHANNELS = [
    ("jyxxcggg", "招标公告"),      # 政府采购 > 采购公告
    ("xmcgxqgg", "采购意向"),      # 采购需求公示
]


def _bj_list_url(channel: str, page: int) -> str:
    if page <= 1:
        return f"https://ggzyfw.beijing.gov.cn/{channel}/index.html"
    return f"https://ggzyfw.beijing.gov.cn/{channel}/index_{page}.html"


def _bj_parse_list(text: str, channel: str) -> list[tuple[str, str]]:
    items = []
    for m in re.finditer(
            rf'<a[^>]+href="(/{channel}/\d{{8}}/\d+\.html)"[^>]*>(.*?)</a>', text, re.S):
        title = re.sub(r"<[^>]+>|\s+", " ", m.group(2)).strip()
        if title:
            items.append((f"https://ggzyfw.beijing.gov.cn{m.group(1)}",
                          html.unescape(title)))
    return items


def _bj_parse_detail(text: str, link: str, pub_type: str, list_title: str) -> dict | None:
    body = _strip_html(text)
    m = re.search(r"发布时间[：:]\s*([\d\-/年]+)", body)
    pub = _norm_date(m.group(1)) if m else None
    m = re.search(r"([^\n]{6,120}?(?:公开招标公告|采购需求公示|采购需求公告|招标公告|询价公告|竞争性磋商公告|更正公告))", body)
    title = m.group(1).strip() if m else list_title
    mb = re.search(r"采购人[：:]\s*([^\n，。]{2,40})", body)
    buyer = mb.group(1).strip() if mb else ""
    region = _judge_region(body[:1500], body)
    budget = _extract_budget(body)
    kw = match_keywords(body[:3000] + " " + title)
    return {"link": link, "title": title[:120], "source": "bj",
            "pub_type": pub_type, "pub_date": pub, "region": region,
            "buyer": buyer, "budget": budget, "keywords": kw,
            "desc": _clean_desc(body, title)}


def fetch_beijing(pages: int | None = None, limit_details: int = 80) -> tuple[list[dict], list[str]]:
    """pages=None 时自动选择：库空（首次回填 30 天窗口）→ 100 页，否则 30 页增量。"""
    if pages is None:
        conn = _connect()
        try:
            _init_db(conn)
            empty = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0
        finally:
            conn.close()
        pages = 100 if empty else 30
    items, errors, seen = [], [], set()
    for channel, tname in BJ_CHANNELS:
        for page in range(1, pages + 1):
            try:
                text = _fetch(_bj_list_url(channel, page)).decode("utf-8", "ignore")
                if _is_blocked(text):
                    errors.append(f"bj: 频控({channel} p{page})")
                    break
                links = _bj_parse_list(text, channel)
                if not links:
                    break
                for link, title in links:
                    if link in seen or not match_keywords(title):
                        continue
                    seen.add(link)
                    time.sleep(_DETAIL_GAP)
                    try:
                        det = _fetch(link).decode("utf-8", "ignore")
                        rec = _bj_parse_detail(det, link, tname, title)
                        if rec and match_keywords(rec["title"] + " " + rec["desc"]):
                            items.append(rec)
                    except Exception as exc:
                        errors.append(f"bj 详情: {type(exc).__name__}")
                    if len(items) >= limit_details:
                        return items, errors
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    break                     # 频道没有该页 → 翻页到底
                errors.append(f"bj {channel} p{page}: HTTP{exc.code}")
                break
            except Exception as exc:
                errors.append(f"bj {channel} p{page}: {type(exc).__name__}")
                break
            time.sleep(_FETCH_GAP)
    return items, errors


# ---------- 源适配器：上海 / 广东（JS 渲染预留） ----------

SH_GUANGDONG_ADAPTERS = [
    ("sh", "https://www.shggzy.com/jyxx.jhtml"),
    ("gd", "https://gdgpo.czt.gd.gov.cn/"),
]


def fetch_render_adapters(exe_py: str | None = None) -> tuple[list[dict], list[str]]:
    """playwright 子进程渲染入口（当前环境被反爬拦截则记入 errors）。
    exe_py: 渲染脚本解释器（默认 ~/OpenManus/.venv/bin/python 带 playwright）。"""
    render_script = DATA_DIR / "render_page.py"
    items, errors = [], []
    if not render_script.exists():
        errors.append("render_page.py 缺失")
        return items, errors
    py = exe_py or str(Path.home() / "OpenManus" / ".venv" / "bin" / "python")
    if not Path(py).exists():
        errors.append("playwright 解释器不可用(需安装 playwright)")
        return items, errors
    for code, url in SH_GUANGDONG_ADAPTERS:
        try:
            import subprocess
            cp = subprocess.run([py, str(render_script), url],
                                capture_output=True, text=True, timeout=45)
            if cp.returncode != 0:
                errors.append(f"{code}: 渲染失败")
                continue
            text = cp.stdout
            if not match_keywords(text):
                errors.append(f"{code}: 渲染无内容或无可解析条目")
                continue
            for m in re.finditer(r"(.{8,120}?(?:地图|GIS|测绘|遥感|北斗|地理信息).{0,60})", text):
                items.append({"link": f"{code}-{hash(m.group(1))}", "title": m.group(1)[:100],
                              "source": code, "pub_type": "渲染抓取", "pub_date": None,
                              "region": "", "buyer": "", "budget": None,
                              "keywords": match_keywords(m.group(1)), "desc": m.group(1)[:500]})
        except Exception as exc:
            errors.append(f"{code}: {type(exc).__name__}")
        time.sleep(_FETCH_GAP)
    return items, errors


# ---------- 汇总抓取 ----------

def fetch(force: bool = False) -> dict:
    """一轮抓取：各源独立容错 → 关键词筛选 → 入库 → 返回元信息。"""
    with _cache["lock"]:
        if not force and _cache["meta"] and time.time() - _cache["t"] < _CACHE_TTL:
            return dict(_cache["meta"])
        conn = _connect()
        _init_db(conn)
        fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
        added, errors = 0, []
        fetchers = [
            (fetch_ccgp, {"provinces": _ccgp_province_pick()}),
            (fetch_beijing, {}),
        ]
        for fetcher, kwargs in fetchers:
            try:
                items, errs = fetcher(**kwargs)
                errors.extend(errs)
                for rec in items:
                    rec["first_seen"] = time.time()
                    if not _in_window(rec["pub_date"], rec["first_seen"]):
                        continue
                    if _insert(conn, rec):
                        added += 1
            except Exception as exc:
                errors.append(f"{fetcher.__name__}: {type(exc).__name__}")
            time.sleep(_FETCH_GAP)
        # 广东/上海适配器（独立目录，不影响主流程）
        src_items, src_errs = fetch_render_adapters()
        errors.extend(src_errs)
        for rec in src_items:
            rec["first_seen"] = time.time()
            if _insert(conn, rec):
                added += 1
        total = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        conn.close()
        meta = {"ok": True, "added": added, "total": total, "errors": errors,
                "fetched_at": fetched_at, "sources": [s[0] for s in SOURCES_META]}
        _cache.update({"t": time.time(), "meta": meta})
        return dict(meta)


# ---------- 统计与查询 ----------

def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["keywords"] = (d.get("keywords") or "").split(",")
    return d


def connected() -> sqlite3.Connection:
    conn = _connect()
    _init_db(conn)
    return conn


def stats(conn: sqlite3.Connection | None = None) -> dict:
    own = conn is None
    conn = conn or connected()
    now = time.time()
    rows = conn.execute("SELECT * FROM projects").fetchall()
    by_region = Counter((r["region"] or "未识别") for r in rows)
    by_type = Counter((r["pub_type"] or "未分类") for r in rows)
    by_source = Counter(r["source"] for r in rows)
    per_day = Counter()
    for r in rows:
        d = r["pub_date"] or datetime.fromtimestamp(r["first_seen"]).date().isoformat()
        per_day[d[:10]] += 1
    kw_counter = Counter()
    for r in rows:
        for k in (r["keywords"] or "").split(","):
            if k:
                kw_counter[k] += 1
    budgets = sorted(r["budget"] for r in rows if r["budget"] is not None)
    budget_range = None
    if len(budgets) >= 10:
        buckets = [("0-100", 0), ("100-500", 0), ("500-1000", 0), ("1000-5000", 0), ("5000+", 0)]
        for b in budgets:
            if b < 100:
                buckets[0] = (buckets[0][0], buckets[0][1] + 1)
            elif b < 500:
                buckets[1] = (buckets[1][0], buckets[1][1] + 1)
            elif b < 1000:
                buckets[2] = (buckets[2][0], buckets[2][1] + 1)
            elif b < 5000:
                buckets[3] = (buckets[3][0], buckets[3][1] + 1)
            else:
                buckets[4] = (buckets[4][0], buckets[4][1] + 1)
        budget_range = [{"name": n, "count": c} for n, c in buckets]
    days = sorted(per_day)
    out = {
        "total": len(rows),
        "by_region": [{"name": k, "count": v} for k, v in by_region.most_common()],
        "by_type": [{"name": k, "count": v} for k, v in by_type.most_common()],
        "by_source": [{"name": k, "count": v} for k, v in by_source.most_common()],
        "per_day": [{"name": d, "count": per_day[d]} for d in days[-30:]],
        "top_keywords": [{"name": k, "count": v} for k, v in kw_counter.most_common(12)],
        "budget_range": budget_range,
        "errors": _cache["meta"].get("errors", []),
        "fetched_at": _cache["meta"].get("fetched_at"),
        "source_status": [{"code": s[0], "name": s[1], "note": s[2]} for s in SOURCES_META],
    }
    if own:
        conn.close()
    return out


def projects(q: str | None = None, region: str | None = None, limit: int = 200) -> list[dict]:
    conn = connected()
    sql, args = "SELECT * FROM projects WHERE 1=1", []
    if q:
        sql += " AND (title LIKE ? OR desc LIKE ?)"
        args += ["%" + q + "%"] * 2
    if region:
        sql += " AND region = ?"
        args.append(region)
    sql += " ORDER BY pub_date DESC, first_seen DESC LIMIT ?"
    args.append(limit)
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def deepdive_input(n: int = 30) -> str:
    conn = connected()
    rows = conn.execute(
        "SELECT title, region, budget, pub_date, source, link FROM projects "
        "ORDER BY pub_date DESC LIMIT ?", (n,)).fetchall()
    conn.close()
    lines = []
    for r in rows:
        budget = f"{r['budget']:.0f}万" if r["budget"] else "预算未知"
        d = r["pub_date"] or "日期未知"
        reg = r["region"] or ""
        lines.append(f"- [{d}]{reg} "
                     f"[{budget}] {r['title']} ({r['source']}) {r['link']}")
    return "\n".join(lines)


def answer_local(q: str) -> str:
    """无 OpenManus 时的确定性规则分析（按地区/类型/预算排序的 Markdown 结论）。"""
    s = stats()
    lines = [f"## 招采地图需求监测 · 快速概览",
             f"\n当前库内共 **{s['total']}** 条地图需求项目（近 30 天），"
             f"数据抓取于 {s['fetched_at'] or '—'}。\n"]
    if s["by_region"]:
        top = s["by_region"][:5]
        lines.append("### 地区 Top5")
        lines += [f"- {r['name']}: {r['count']} 条" for r in top]
    if s["by_type"]:
        lines.append("\n### 项目类型")
        lines += [f"- {t['name']}: {t['count']} 条" for t in s["by_type"]]
    if s["top_keywords"]:
        lines.append("\n### 需求关键词")
        lines += [f"- {k['name']}: {k['count']} 次" for k in s["top_keywords"][:8]]
    rows = projects(limit=5)
    if rows:
        lines.append("\n### 最新项目")
        for r in rows:
            b = f"{r['budget']:.0f}万" if r["budget"] else "预算未知"
            lines.append(f"- {r['pub_date'] or ''} {r['title']} （{r['region'] or '地区未识别'}，{b}）")
    return "\n".join(lines)


def probe_ccgp(provs: list[tuple[str, str]] | None = None) -> str:
    """验证 ccgp province 参数是否生效：抓代表省份列表页，打印命中/频控状态。"""
    now = datetime.now()
    end = now.date().isoformat()
    start = (now - timedelta(days=_WINDOW_DAYS)).date().isoformat()
    provs = provs or CCGP_PROVINCES[:5] + CCGP_PROVINCES[-2:]
    lines = []
    for prov_full, prov_short in provs:
        url = _ccgp_list_url("地图", 1, "1", start, end, prov_full)
        try:
            text = _fetch(url).decode("utf-8", "ignore")
            if _is_blocked(text):
                lines.append(f"{prov_short}: 频控拦截")
                continue
            n = len(_ccgp_parse_list(text))
            lines.append(f"{prov_short}: 列表 {n} 条 {'(可能未按省过滤,值得怀疑)' if n > 30 else ''}")
        except Exception as exc:
            lines.append(f"{prov_short}: {type(exc).__name__}")
        time.sleep(_CCGP_GAP)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="招采地图数据层")
    ap.add_argument("--probe", action="store_true", help="验证 ccgp province 参数是否生效")
    args = ap.parse_args()
    if args.probe:
        print(probe_ccgp())
    else:
        import json
        meta = fetch(force=True)
        print(json.dumps(meta, ensure_ascii=False, indent=1))
        if meta["total"]:
            print("\n最新 8 条：")
            for p in projects(limit=8):
                print(f"  [{p['region'] or '?'}/{p['pub_type'] or '?'}] {p['title'][:46]} "
                      f"@{p['source']} 预算={p['budget']}")