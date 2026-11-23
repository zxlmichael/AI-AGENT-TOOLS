#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""衣物直播商品识别助手 · 数据层（纯标准库，可独立测试）

输入商品文本/JSON，纯规则判定：①是否为衣物类（服装/鞋靴/箱包，多字词表）
②定价是否 >= 1000 元。记录写入 records.jsonl（线程安全，5000 行滚动归档）。

设计文档：.comate/specs/live-detect/doc.md
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "live_detect_data"
RECORDS_PATH = DATA_DIR / "records.jsonl"

PRICE_THRESHOLD = 1000
MAX_RECORDS = 5000  # 活动文件行数上限，超出滚动归档（测试可注入调小）

# 衣物类目多字词表（按主类目归类；命中即 is_cloth=True）
CLOTH_KEYWORDS = {
    "服装": ["羽绒服", "大衣", "风衣", "卫衣", "毛衣", "针织衫", "羊绒衫", "羊毛衫",
             "连衣裙", "衬衫", "西装", "外套", "夹克", "马甲", "皮衣", "牛仔裤",
             "休闲裤", "阔腿裤", "半身裙", "百褶裙", "套装", "内衣", "家居服",
             "打底衫", "POLO衫", "T恤", "短袖", "长袖", "羽绒马甲", "冲锋衣",
             "西装外套", "风衣外套", "卫裤", "运动裤", "工装裤"],
    "鞋靴": ["运动鞋", "板鞋", "皮鞋", "靴子", "雪地靴", "马丁靴", "高跟鞋",
             "帆布鞋", "拖鞋", "乐福鞋", "老爹鞋", "小白鞋", "登山鞋", "短靴",
             "长靴", "切尔西靴", "豆豆鞋"],
    "箱包": ["双肩包", "手提包", "单肩包", "斜挎包", "钱包", "背包", "行李箱",
             "胸包", "腰包", "通勤包", "托特包", "腋下包", "水桶包", "链条包",
             "妈咪包", "旅行箱", "皮包", "真皮包", "羊皮包", "牛皮包"],
}
# 显式排除词：命中时不做「包」系列判定（防"包邮"等误判为箱包）
EXCLUDE_WORDS = ["包邮", "包退", "包换", "包赔", "包税", "面包", "包装", "包打听"]

# 价格修饰词（后跟的价格优先于裸价格）：券后/到手/现价/秒杀价/活动价/到手价
_STRONG_PRICE = re.compile(
    r"(?:券后|到手|现价|秒杀|活动|折后|特惠|促销|拼团|团购|直播价|福利价)"
    r"价?\s*[：:为是]?\s*[¥￥]?\s*(\d{1,7}(?:\.\d{1,2})?)", re.I)
_RAW_PRICE = re.compile(r"[¥￥]\s*(\d{1,7}(?:\.\d{1,2})?)|(\d{1,7}(?:\.\d{1,2})?)\s*(?:元|块|rmb|RMB)")
_JSON_FIELD = re.compile(r'"(?:price|现价|到手价|活动价|售价|券后价)"\s*[:=]\s*"?(\d{1,7}(?:\.\d{1,2})?)"?')

_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

def _clean_text(text: str) -> str:
    """去掉链接/HTML 标签/多余空白，保留可读文本。"""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_price(text: str):
    """提取价格：JSON 字段 > 强修饰词（券后/到手/现价…） > 裸价格取最小候选。"""
    m = _JSON_FIELD.search(text)
    if m:
        return float(m.group(1)), "json字段"
    m = _STRONG_PRICE.search(text)
    if m:
        return float(m.group(1)), "修饰词(券后/到手等)"
    cands = [float(a or b) for a, b in _RAW_PRICE.findall(text)]
    if cands:
        return min(cands), "裸价格取最小候选"
    return None, "未提取到价格"


def parse_input(text: str) -> dict:
    """从自由文本/JSON 提取 {name, price, category, desc}；提取失败字段为 None。"""
    text = (text or "").strip()
    if not text:
        return {"name": None, "price": None, "category": None, "desc": ""}
    # 尝试 JSON 对象（如 {"name": "...", "price": 1288}）
    name, category, price = None, None, None
    price_why = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            name = data.get("name") or data.get("title") or data.get("商品名")
            category = data.get("category") or data.get("类目")
            raw_price = data.get("price") or data.get("现价") or data.get("到手价")
            if raw_price is not None:
                try:
                    price = float(str(raw_price).replace(",", ""))
                    price_why = "json字段"
                except ValueError:
                    pass
    except (json.JSONDecodeError, TypeError):
        pass
    if name is None:
        # 取第一个双引号/书名号内片段或整行首句作为商品名候选
        m = re.search(r"[「『\"“]([^」』\"”]{2,60})[」』\"”]", text)
        if not m:
            m = re.search(r"^([^,，。;；|]{2,60})", text)
        name = (m.group(1).strip() if m else text[:60])
    if price is None:
        price, price_why = _extract_price(text)
    return {"name": name, "price": price, "category": category,
            "desc": _clean_text(text)}


# --------------------------------------------------------------------------- #
# 规则判定
# --------------------------------------------------------------------------- #

def _match_category(name: str) -> tuple[bool, str | None, str | None]:
    """返回 (is_cloth, category, matched_keyword)；命中排除词时跳过「包」系列。"""
    if not name:
        return False, None, None
    for w in EXCLUDE_WORDS:
        if w in name:
            name = name.replace(w, " ")
    for cat, words in CLOTH_KEYWORDS.items():
        for w in words:
            if w in name:
                return True, cat, w
    return False, None, None


def classify_product(name: str, price=None, desc: str = "") -> dict:
    """纯规则判定。返回 {is_cloth, category, price, premium, reason, method}"""
    name = (name or "").strip()
    is_cloth, category, kw = _match_category(name)
    if price is not None:
        try:
            price = round(float(price), 2)
        except (TypeError, ValueError):
            price = None
    premium = bool(is_cloth and price is not None and price >= PRICE_THRESHOLD)
    if is_cloth:
        reason = f"词表命中「{kw}」→ {category}"
    else:
        reason = "未命中文物类目词表" if name else "缺少商品名称"
    if price is None:
        reason += "；未提取到价格"
    elif not is_cloth:
        pass
    else:
        reason += f"；价格 {price:g} 元" + ("，达标(≥1000元)" if premium else "，未达标(<1000元)")
    return {"is_cloth": bool(is_cloth), "category": category, "price": price,
            "premium": bool(premium), "reason": reason, "method": "rule"}


# --------------------------------------------------------------------------- #
# 存储（线程安全 + 滚动归档）
# --------------------------------------------------------------------------- #

def _read_all() -> list[dict]:
    """读取活动文件 + 全部归档文件（按时间序合并），坏行跳过。"""
    if not RECORDS_PATH.exists():
        return []
    files = sorted(DATA_DIR.glob("records_*.jsonl")) + [RECORDS_PATH]
    rows = []
    for fp in files:
        try:
            for line in fp.read_text("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return rows


def _rollover_if_needed():
    """活动文件超过 MAX_RECORDS 时归档（带毫秒时间戳防重名），保留最后 100 行。"""
    if not RECORDS_PATH.exists():
        return
    lines = [l for l in RECORDS_PATH.read_text("utf-8", errors="replace")
             .splitlines() if l.strip()]
    if len(lines) <= MAX_RECORDS:
        return
    name = f"records_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jsonl"
    arch = DATA_DIR / name
    arch.write_text("\n".join(lines[:-100]) + "\n", "utf-8")
    RECORDS_PATH.write_text("\n".join(lines[-100:]) + "\n", "utf-8")


def _append(record: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        _rollover_if_needed()
        with open(RECORDS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def detect(text: str) -> dict:
    """解析+识别+入库。返回 {ok, results, written, duplicate, records_total}"""
    parsed = parse_input(text)
    result = classify_product(parsed["name"], parsed["price"], parsed["desc"])
    record = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "source": "link" if re.search(r"https?://", text or "") else "text",
              "platform": "other" if re.search(r"douyin|live\.|taobao|kuaishou",
                                               text or "", re.I) else "none",
              "name": parsed["name"], "category": result["category"],
              "price": result["price"], "is_cloth": result["is_cloth"],
              "premium": result["premium"], "method": result["method"],
              "reason": result["reason"]}
    # 去重：与最近 50 条比较 (name, price)
    duplicate = False
    with _lock:
        for rec in _read_all()[-50:]:
            if rec.get("name") == record["name"] and rec.get("price") == record["price"]:
                duplicate = True
                break
    if not duplicate:
        _append(record)
    total = len(_read_all())
    result["name"] = parsed["name"]  # 供调用方直接展示
    return {"ok": True, "results": [result], "written": 0 if duplicate else 1,
            "duplicate": duplicate, "records_total": total}


def list_records(kind: str = "all", n: int = 30) -> list[dict]:
    """kind: all|cloth|premium；返回最近 n 条，按时间倒序。"""
    rows = _read_all()
    if kind == "cloth":
        rows = [r for r in rows if r.get("is_cloth")]
    elif kind == "premium":
        rows = [r for r in rows if r.get("premium")]
    return list(reversed(rows[-max(1, min(int(n), 500)):]))


def stats() -> dict:
    """{total, cloth, premium, by_category, price_bins, updated}，统计口径=全部记录。"""
    rows = _read_all()
    by_category: dict = {}
    bins = {"1000-2000": 0, "2000-5000": 0, "5000-10000": 0, "10000+": 0}
    for r in rows:
        if r.get("category"):
            by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        p = r.get("price")
        if p is not None:
            if p < 2000:
                bins["1000-2000"] += 1
            elif p < 5000:
                bins["2000-5000"] += 1
            elif p < 10000:
                bins["5000-10000"] += 1
            else:
                bins["10000+"] += 1
    return {"total": len(rows),
            "cloth": sum(1 for r in rows if r.get("is_cloth")),
            "premium": sum(1 for r in rows if r.get("premium")),
            "by_category": by_category, "price_bins": bins}
    # updated 语义：最后写入时间，由调用方从 records 取


def updated_at() -> str:
    rows = _read_all()
    return rows[-1]["ts"] if rows else ""


# --------------------------------------------------------------------------- #
# 预留接口位（v1 恒返回 None；后续接入平台抓取/第三方 API 时实现）
# --------------------------------------------------------------------------- #

def fetch_goods(url: str) -> list[dict] | None:
    """平台商品抓取适配器。v1 未实现（平台接口需签名/登录态），恒返回 None。"""
    return None


if __name__ == "__main__":
    # 数据层自测
    cases = [
        "【新品】2026秋冬 羊绒大衣 女 中长款 券后2680元",
        "纯棉T恤 99元 包邮",
        "无线蓝牙耳机 199元",
        "某品牌 双肩包 直播间到手价 1299",
        "原价 2999 活动价 899 的羽绒服",
        '{"name": "真皮手提包", "price": 1580}',
    ]
    for c in cases:
        r = detect(c)
        res = r["results"][0]
        print(f"{'WRITE' if r['written'] else 'DUP  '} | {res['category']} | "
              f"{res['price']} | cloth={res['is_cloth']} premium={res['premium']} | {res['reason']}")
    print("\nstats:", stats())
    print("total records:", len(list_records("all")))