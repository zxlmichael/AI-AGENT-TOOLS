#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFL 赛事助手 · 数据层（纯标准库）

数据源：ESPN 公开接口（本网络实测可用，2026-11-13）：
  - 赛程比分 /apis/site/v2/sports/football/nfl/scoreboard
  - 新闻流   /apis/site/v2/sports/football/nfl/news
  - (standings 接口 404，未使用；球队排名从赛程比分推导)

新闻归档：nfl_news_archive/ 下按月份 JSONL（nfl_news_YYYYMM.jsonl），
供"每月新闻"在线发布与回溯。
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None

HERE = Path(__file__).resolve().parent
ARCHIVE_DIR = HERE / "nfl_news_archive"

SCOREBOARD_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
NEWS_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"

CST = timezone(timedelta(hours=8))  # 北京时间


def _ssl_ctx() -> ssl.SSLContext | None:
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _cn_time(iso: str) -> str:
    """ISO 时间转北京时间字符串；解析失败返回原串。"""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return iso or ""


def _team_info(comp) -> dict:
    t = comp["team"]
    return {"name": t.get("displayName", ""), "abbr": t.get("abbreviation", ""),
            "score": comp.get("score", "")}


# --------------------------------------------------------------------------- #
# 赛程 / 比分
# --------------------------------------------------------------------------- #

def scores(limit: int = 14) -> dict:
    """当日/近期赛程与比分。返回 {date_hint, games: [...]}。"""
    d = _fetch_json(SCOREBOARD_API)
    out = []
    for e in d.get("events", [])[:max(1, min(int(limit), 50))]:
        comps = e.get("competitions", [])
        if not comps:
            continue
        c = comps[0]
        st = e.get("status", {}).get("type", {})
        out.append({
            "id": e.get("id", ""),
            "name": e.get("name", ""),
            "date": _cn_time(e.get("date", "")),
            "state": st.get("state", ""),          # pre/in/post
            "detail": st.get("detail", ""),        # Final / 3rd Qtr / 今日 08:20
            "away": _team_info(c["competitors"][0]),
            "home": _team_info(c["competitors"][1]),
        })
    return {"date_hint": d.get("day", date_hint_today()),
            "games": out}


def date_hint_today() -> str:
    return datetime.now(CST).strftime("%Y%m%d")


def leaders() -> list[dict]:
    """简易球队排名：从近 14 场比分统计 胜-负（仅作参考，非官方排名）。"""
    s = scores(14)
    rec: dict = {}
    for g in s["games"]:
        if g["state"] != "post" or g["away"]["score"] == "" or g["home"]["score"] == "":
            continue
        a, h = g["away"], g["home"]
        try:
            a_s, h_s = int(a["score"]), int(h["score"])
        except (TypeError, ValueError):
            continue
        for team in (a, h):
            rec.setdefault(team["name"], {"w": 0, "l": 0, "abbr": team["abbr"]})
        if a_s > h_s:
            rec[a["name"]]["w"] += 1
            rec[h["name"]]["l"] += 1
        else:
            rec[h["name"]]["w"] += 1
            rec[a["name"]]["l"] += 1
    rows = [{"name": n, "abbr": v["abbr"], "wins": v["w"], "losses": v["l"],
             "pct": round(v["w"] / (v["w"] + v["l"]), 3) if (v["w"] + v["l"]) else 0}
            for n, v in rec.items()]
    rows.sort(key=lambda x: (-x["pct"], -x["wins"]))
    return rows


# --------------------------------------------------------------------------- #
# 新闻
# --------------------------------------------------------------------------- #

def news(limit: int = 10) -> list[dict]:
    """ESPN NFL 新闻流（最新在前）。"""
    d = _fetch_json(f"{NEWS_API}?limit={max(1, min(int(limit), 30))}")
    out = []
    for a in d.get("articles", []):
        link = ""
        if a.get("links", {}).get("web"):
            link = a["links"]["web"].get("href", "")
        img = ""
        if a.get("images"):
            img = a["images"][0].get("url", "")
        out.append({
            "id": a.get("id", ""),
            "headline": a.get("headline", ""),
            "summary": (a.get("description") or "")[:260],
            "published": _cn_time(a.get("published", "")),
            "byline": a.get("byline", ""),
            "link": link, "image": img,
            "category": (a.get("categories") or [{}])[0].get("description", "NFL"),
        })
    # 按发布时间倒序
    out.sort(key=lambda x: x["published"], reverse=True)
    return out


def _archive_path(ym: str) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR / f"nfl_news_{ym}.jsonl"


def archive_news(items: list[dict]) -> dict:
    """按年月归档新闻（同 id 去重），返回 {added, total}。"""
    if not items:
        return {"added": 0, "total": 0}
    added = 0
    for it in items:
        ym = it["published"][:7].replace("-", "") if it["published"] else "unknown"
        path = _archive_path(ym)
        exists = set()
        if path.exists():
            for line in path.read_text("utf-8", errors="replace").splitlines():
                try:
                    exists.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if it["id"] in exists:
            continue
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        added += 1
    total = sum(1 for p in ARCHIVE_DIR.glob("nfl_news_*.jsonl")
                for _ in p.read_text("utf-8", errors="replace").splitlines() if _.strip())
    return {"added": added, "total": total}


def monthly_news(ym: str = "") -> list[dict]:
    """读取某月归档新闻（ym=YYYYMM；空则读最新月份），按时间倒序。

    返回 {month, items: [...], available: [月份列表]}。
    """
    months = sorted(p.stem.replace("nfl_news_", "")
                    for p in ARCHIVE_DIR.glob("nfl_news_*.jsonl")) if ARCHIVE_DIR.exists() else []
    if not months:
        return {"month": ym, "items": [], "available": []}
    if not ym:
        ym = months[-1]
    path = ARCHIVE_DIR / f"nfl_news_{ym}.jsonl"
    items = []
    if path.exists():
        for line in path.read_text("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    items.sort(key=lambda x: x.get("published", ""), reverse=True)
    return {"month": ym, "items": items, "available": months}


# --------------------------------------------------------------------------- #
# 月报
# --------------------------------------------------------------------------- #

def build_monthly_report(ym: str = "") -> str:
    """生成某月 NFL 新闻月报（Markdown）；无归档时抓取当前新闻并归档。"""
    m = monthly_news(ym)
    if not m["items"]:
        arch = archive_news(news(10))
        m = monthly_news(arch and m["month"] or ym)
    items = m["items"]
    if not items:
        return "暂无月度新闻归档。"
    lines = [f"# NFL 赛事月报（{m['month'][:4]}-{m['month'][4:]}）",
             f"\n- 新闻条数：{len(items)}",
             f"- 最新时间：{items[0]['published']}",
             ""]
    for i, it in enumerate(items, 1):
        lines.append(f"## {i}. {it['headline']}")
        lines.append(f"- 时间：{it['published']} ｜ 栏目：{it.get('category', 'NFL')}")
        if it.get("byline"):
            lines.append(f"- 作者：{it['byline']}")
        if it.get("summary"):
            lines.append(f"- 摘要：{it['summary']}")
        if it.get("link"):
            lines.append(f"- 原文：{it['link']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    s = scores(4)
    print("SCORES:", s["date_hint"])
    for g in s["games"][:4]:
        print(f"  {g['away']['name']} {g['away']['score']} @ {g['home']['name']} "
              f"{g['home']['score']} [{g['detail']}]")
    print("\nLEADERS(近14场胜率):")
    for r in leaders()[:5]:
        print(f"  {r['name']} {r['wins']}-{r['losses']} ({r['pct']:.3f})")
    n = news(6)
    print(f"\nNEWS: {len(n)} 条")
    for x in n[:3]:
        print(f"  {x['published']} {x['headline'][:60]}")
    a = archive_news(n)
    print("archive:", a)
    m = monthly_news("")
    print(f"monthly: {m['month']} items={len(m['items'])} available={m['available']}")
    report = build_monthly_report(m["month"])
    print(f"report head: {report.splitlines()[0] if report else 'EMPTY'}")