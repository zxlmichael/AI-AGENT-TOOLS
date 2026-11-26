#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""时事新闻数据层：多源 RSS 抓取 → 统一条目 → 分类与关键词统计

数据源（全部免费公开，多源容错）：
- BBC 中文（简体）：https://feeds.bbci.co.uk/zhongwen/simp/rss.xml
- BBC World（英文国际）：https://feeds.bbci.co.uk/news/world/rss.xml
- 纽约时报中文网：https://cn.nytimes.com/rss/
- Google News（中文聚合，含国内/国际）：https://news.google.com/rss?hl=zh-CN...ceid=CN:zh-Hans

统计输出（供 ECharts 可视化与问答）：
- by_source：各来源条数
- by_region：国内/国际/亚洲等地区分布
- by_topic：主题分布（政治/经济/科技/军事/社会/国际/环境/体育）
- top_keywords：标题分词词频 TopN（jieba）
- per_hour：最近 24 小时条数分布
"""

from __future__ import annotations

import os
import re
import ssl
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta, timezone

import certifi
import jieba
import jieba.analyse

try:
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = None

SOURCES = [
    ("BBC中文", "https://feeds.bbci.co.uk/zhongwen/simp/rss.xml", "国际"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", "国际"),
    ("纽约时报中文", "https://cn.nytimes.com/rss/", "国际"),
    ("Google News中文", "https://news.google.com/rss?hl=zh-CN&gl=CN&ceid=CN:zh-Hans", "混合"),
]

_CACHE_TTL = 600            # 抓取结果缓存 10 分钟（RSS 更新频率低，别频繁打源）
_cache = {"lock": threading.Lock(), "t": 0.0, "items": [], "errors": []}

# 简单停用词（标题分词用，过长/过泛词汇过滤）
_STOP = set("""的 了 是 在 和 与 及 或 为 对 中 上 下 将 被 从 向 而 其 之 以 于 后 前 也 都 还 又
再 更 最 很 太 不 没 有 这 那 就 能 要 会 可 但 并 等 我们 你们 他们 自己 一个 一些 什么 如何 为何 哪些
今天 明天 昨天 本月 本周 今年 去年 次年 上半年 下半年 表示 认为 报道 记者 消息 新闻 中国 美国 英国 日本 俄罗斯
the a an of to in for on with and or from by at as is are was were be been has have had it its
this that these those will would can could may might not no yes so than then into over under
after before during about between against through among along within without around """.split())

# 自定义词典（jieba 默认切碎的专业词）
for _w in ("英伟达", "黄仁勋", "黄仁勳", "人工智能", "SpaceX", "星链", "特朗普", "拜登",
           "美联储", "纳斯达克", "标准普尔", "新能源汽车", "半导体", "台积电", "苹果公司",
           "OpenAI", "ChatGPT", "字节跳动", "阿里巴巴", "神舟", "世界杯", "联合国", "美国大选"):
    jieba.add_word(_w)

# 主题关键词规则（按命中得分排序，取最高分主题）
_TOPIC_RULES = [
    ("政治", ["大选", "选举", "总统", "总理", "国会", "议会", "政府", "外交", "制裁", "峰会",
              "竞选", "政变", "内阁", "执政", "政党", "特朗普", "拜登", "普京", "习近平",
              "国务卿", "白宫", "克里姆林宫"]),
    ("经济", ["经济", "通胀", "GDP", "股价", "股市", "汇率", "加息", "降息", "关税", "贸易",
              "出口", "进口", "财政", "失业", "消费", "央行", "美联储", "人民币", "美元",
              "billion", "million", "inflation", "economy", "market", "trade"]),
    ("科技", ["AI", "人工智能", "芯片", "半导体", "机器人", "航天", "火箭", "卫星", "SpaceX",
              "苹果", "谷歌", "微软", "英伟达", "华为", "特斯拉", "软件", "数据", "互联网",
              "科技", "tech", "chip", "AI"]),
    ("军事", ["战争", "冲突", "导弹", "坦克", "轰炸", "空袭", "停火", "军演", "核武", "武器",
              "战事", "无人机", "战争", "military", "missile", "war", "invasion"]),
    ("环境", ["气候", "台风", "地震", "洪水", "干旱", "污染", "碳排放", "环保", "极端天气",
              "暴雨", "高温", "山火", "climate", "flood", "storm"]),
    ("社会", ["起火", "大火", "事故", "坠机", "沉船", "犯罪", "罢工", "抗议", "游行", "罢工",
              "丢失", "盗窃", "中毒", "爆炸", "疫情", "流感", "病毒"]),
    ("体育", ["世界杯", "奥运会", "欧冠", "英超", "NBA", "足球", "篮球", "网球", "冠军",
              "联赛", "比赛", "football", "soccer", "NBA", "Olympic"]),
]

# 国内/国际判断关键词（标题级别；标题不含则按来源默认）
_CN_HINTS = ["中国", "北京", "上海", "广州", "深圳", "国务院", "全国", "国内", "新华", "人民网",
             "央视", "人大", "政协", "台海", "台湾", "香港", "澳门", "中欧", "中美", "外交部",
             "中国央行", "神舟", "国产", "央企", "中企", "司法部", "湖北", "四川", "浙江",
             "江苏", "山东", "河南", "河北", "湖南", "安徽", "福建", "广东", "云南", "贵州",
             "陕西", "甘肃", "辽宁", "吉林", "黑龙江"]
# 国内媒体 provider（Google News 聚合常见来源）
_CN_MEDIA = ["新华网", "人民网", "央视", "央视网", "环球网", "环球时报", "澎湃", "新浪", "网易",
             "腾讯", "百度", "凤凰", "中新网", "光明", "经济日报", "中国日报", "中青", "观察者",
             "界面", "财新", "第一财经", "每日经济", "南方都市", "上观", "封面", "红星",
             "极目", "齐鲁", "澎湃新闻", "湖北日报", "四川日报", "北京日报", "解放日报",
             "东方财富", "证券时报", "钛媒体", "36氪", "虎嗅", "IT之家", "快科技", "中关村",
             "搜狐", "爱奇艺", "优酷", "字节跳动", "知乎", "微博", "抖音", "驱动之家",
             "游民星空", "游侠", "3DM", "机锋", "天极", "太平洋", "猫眼", "豆瓣", "B站",
             "哔哩哔哩", "网易娱乐", "新浪娱乐", "腾讯娱乐", "欢聚", "陌陌", "快手"]


def _fetch(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read(600000)


def _html_strip(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _parse_rss(xml_bytes: bytes, source: str, default_region: str) -> list[dict]:
    text = xml_bytes.decode("utf-8", "ignore")
    text = re.sub(r"^<\?xml[^>]*\?>", "", text)
    root = ET.fromstring(text)
    items = []
    for it in root.findall(".//item"):
        title = _html_strip(it.findtext("title", default=""))
        link = it.findtext("link", default="").strip()
        desc = _html_strip(it.findtext("description", default=""))
        pub = it.findtext("pubDate", default="") or it.findtext("dc:date", default="")
        # provider（Google News 标题带 " - 来源" 后缀，拆掉）
        prov = it.findtext("source", default="") or ""
        m = re.search(r"\s*-\s*(.+)$", title)
        if m and (prov or title.count(" - ") >= 1):
            title = title[: m.start()].strip()
        if not title:
            continue
        dt = _parse_pub(pub)
        items.append({
            "title": title, "link": link, "desc": desc[:220],
            "source": source, "provider": prov,
            "time": dt.isoformat() if dt else None,
            "ts": dt.timestamp() if dt else 0,
            "region": _judge_region(title, default_region, prov),
            "topic": _judge_topic(title),
        })
    return items


def _parse_pub(pub: str) -> datetime | None:
    """解析 RSS pubDate：RFC822 或 ISO8601。失败返回 None。"""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(pub)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(pub.replace("Z", "+00:00"))
    except Exception:
        return None


def _judge_region(title: str, default: str, provider: str = "") -> str:
    if any(h in title for h in _CN_HINTS):
        return "国内"
    if provider and _CN_MEDIA and any(m in provider for m in _CN_MEDIA):
        return "国内"
    if default in ("国际", "混合"):
        return "国际"
    return default


def _judge_topic(title: str) -> str:
    """标题命中规则得分，取最高主题；无命中归"综合"。"""
    best, score = "综合", 0
    for topic, kws in _TOPIC_RULES:
        s = sum(1 for k in kws if k.lower() in title.lower())
        if s > score:
            best, score = topic, s
    return best


def fetch(items_limit: int = 0, timeout: int = 15) -> tuple[list[dict], list[str]]:
    """抓取全部源；单源失败只记错误不影响其他源。返回 (items, errors)。"""
    items, errors = [], []
    for source, url, default_region in SOURCES:
        try:
            got = _parse_rss(_fetch(url, timeout=timeout), source, default_region)
            items.extend(got)
            time.sleep(0.3)          # 源间礼貌间隔
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}")
    items.sort(key=lambda x: -x["ts"])
    return (items[:items_limit] if items_limit else items), errors


def cached_items(force: bool = False) -> tuple[list[dict], list[str]]:
    """带 10 分钟缓存的抓取入口（服务端频繁调用时用）。"""
    with _cache["lock"]:
        if not force and _cache["items"] and time.time() - _cache["t"] < _CACHE_TTL:
            return _cache["items"], _cache["errors"]
        items, errors = fetch()
        _cache.update({"t": time.time(), "items": items, "errors": errors})
        return items, errors


def top_keywords(items: list[dict], n: int = 12) -> list[dict]:
    """标题全量分词词频 TopN（jieba.cut，过滤停用词/单字/媒体名）。"""
    titles = " ".join(it["title"] for it in items)
    if not titles:
        return []
    words = [w.strip().lower() for w in jieba.cut(titles) if w.strip()]
    counter = Counter(
        w for w in words
        if len(w) > 1 and w not in _STOP and not w.isdigit()
        and not any(m in w for m in ("新华", "新浪", "网易", "腾讯", "百度", "客户端", "com", "cn"))
    )
    return [{"name": k, "count": v} for k, v in counter.most_common(n)]


def stats(items: list[dict]) -> dict:
    """统计汇总：来源/地区/主题分布、关键词 TOP、24h 时间线。"""
    now = time.time()
    by_source = Counter(it["source"] for it in items)
    by_region = Counter(it["region"] for it in items)
    by_topic = Counter(it["topic"] for it in items)
    per_hour = Counter()
    for it in items:
        if it["ts"] and now - it["ts"] < 24 * 3600:
            per_hour[datetime.fromtimestamp(it["ts"], tz=timezone.utc).strftime("%H:00")] += 1
    hours = [f"{h:02d}:00" for h in range(24)]
    return {
        "total": len(items),
        "fresh": sum(1 for it in items if it["ts"] and now - it["ts"] < 12 * 3600),
        "by_source": [{"name": k, "count": v} for k, v in by_source.most_common()],
        "by_region": [{"name": k, "count": v} for k, v in by_region.most_common()],
        "by_topic": [{"name": k, "count": v} for k, v in by_topic.most_common()],
        "top_keywords": top_keywords(items),
        "per_hour": [{"name": h, "count": per_hour.get(h, 0)} for h in hours],
    }


def recent(items: list[dict], k: int = 30) -> list[dict]:
    return items[:k]


if __name__ == "__main__":
    # 自测：python news_data.py
    import json
    items, errors = fetch()
    print(f"抓取 {len(items)} 条，源错误：{errors}")
    if items:
        s = stats(items)
        print(json.dumps({k: s[k] for k in ("total", "fresh", "by_source", "by_region",
                                            "by_topic")}, ensure_ascii=False, indent=1))
        print("关键词 TOP:", [f"{k['name']}:{k['count']}" for k in s["top_keywords"][:8]])
        print("\n最新 5 条：")
        for it in recent(items, 5):
            print(f"  [{it['region']}/{it['topic']}] {it['title'][:50]} @{it['source']}")