#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛事数据层：F1 / NBA / 英超 / 高尔夫全球巡回赛。

数据源
  F1   : Jolpica F1 API（Ergast 兼容）  https://api.jolpi.ca/ergast/f1/current/...
  NBA  : ESPN 公开 API                 https://site.api.espn.com/apis/v2/sports/basketball/nba/standings
  英超 : ESPN 公开 API                 https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings
  高尔夫: ESPN 公开 API（PGA / DP World / LPGA）

所有请求带本地 JSON 缓存：默认 10 分钟新鲜期，网络失败时回退到旧缓存，
并在 meta 中标记 live / cache / stale，报告里会如实注明数据新鲜度。
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import certifi

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "sports_cache"
CACHE_TTL = 600          # 常规数据 10 分钟
GOLF_PAST_TTL = 7 * 86400  # 已结束的高尔夫赛事不再变化

_CTX = ssl.create_default_context(cafile=certifi.where())
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36")


# --------------------------------------------------------------------------- #
# HTTP + 缓存
# --------------------------------------------------------------------------- #

def _get(url: str, key: str, ttl: int = CACHE_TTL, force: bool = False):
    """读取接口，返回 (data, meta)。meta: source / fetched_at / error。"""
    path = CACHE_DIR / f"{key}.json"
    now = time.time()

    if not force and path.exists() and now - path.stat().st_mtime < ttl:
        try:
            return (json.loads(path.read_text("utf-8")),
                    {"source": "cache", "fetched_at": _iso(path.stat().st_mtime)})
        except Exception:
            pass

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=25, context=_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        return data, {"source": "live", "fetched_at": _iso(now)}
    except Exception as exc:  # 网络失败 → 旧缓存兜底
        if path.exists():
            try:
                return (json.loads(path.read_text("utf-8")),
                        {"source": "stale", "fetched_at": _iso(path.stat().st_mtime),
                         "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        raise


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _merge_meta(*metas) -> dict:
    """多个接口的 meta 合并成一份数据源说明。"""
    sources = [m.get("source", "?") for m in metas if m]
    if "stale" in sources:
        src = "stale"
    elif "live" in sources:
        src = "live"
    else:
        src = "cache"
    fetched = max((m.get("fetched_at", "") for m in metas if m), default="")
    errors = [m["error"] for m in metas if m and m.get("error")]
    out = {"source": src, "fetched_at": fetched, "n_requests": len(sources)}
    if errors:
        out["error"] = " | ".join(errors[:2])
    return out


# --------------------------------------------------------------------------- #
# F1
# --------------------------------------------------------------------------- #

def f1_snapshot(force: bool = False) -> dict:
    base = "https://api.jolpi.ca/ergast/f1/current"
    drv, m1 = _get(f"{base}/driverstandings.json", "f1_driver_standings", force=force)
    con, m2 = _get(f"{base}/constructorstandings.json", "f1_constructor_standings", force=force)
    last, m3 = _get(f"{base}/last/results.json", "f1_last_results", force=force)
    sched, m4 = _get(f"{base}.json", "f1_schedule", force=force)

    drv_list = drv["MRData"]["StandingsTable"]["StandingsLists"]
    constructors_meta = con["MRData"]["StandingsTable"]["StandingsLists"]
    if not drv_list:
        raise RuntimeError("F1 车手积分榜为空")

    dlist = drv_list[0]
    season, rnd = dlist.get("season", ""), _int(dlist.get("round"))

    drivers = []
    for d in dlist.get("DriverStandings", []):
        drv_info = d.get("Driver", {})
        teams = d.get("Constructors") or [{}]
        drivers.append({
            "pos": _int(d.get("position")),
            "name": f"{drv_info.get('givenName', '')} {drv_info.get('familyName', '')}".strip(),
            "code": drv_info.get("code", ""),
            "team": teams[0].get("name", ""),
            "points": _num(d.get("points")),
            "wins": _int(d.get("wins")),
            "nationality": drv_info.get("nationality", ""),
        })

    constructors = []
    if constructors_meta:
        for c in constructors_meta[0].get("ConstructorStandings", []):
            info = c.get("Constructor", {})
            constructors.append({
                "pos": _int(c.get("position")),
                "name": info.get("name", ""),
                "points": _num(c.get("points")),
                "wins": _int(c.get("wins")),
                "nationality": info.get("nationality", ""),
            })

    # 上一站
    last_race = {}
    races = last["MRData"]["RaceTable"].get("Races", [])
    if races:
        r = races[0]
        results = []
        for item in r.get("Results", []):
            fl = item.get("FastestLap") or {}
            t = item.get("Time") or {}
            results.append({
                "pos": _int(item.get("position")),
                "driver": f"{item['Driver'].get('givenName', '')} {item['Driver'].get('familyName', '')}".strip(),
                "code": item["Driver"].get("code", ""),
                "team": item.get("Constructor", {}).get("name", ""),
                "grid": _int(item.get("grid")),
                "laps": _int(item.get("laps")),
                "status": item.get("status", ""),
                "points": _num(item.get("points")),
                "time": t.get("time", ""),
                "fastest_lap": _int(fl.get("rank")) == 1,
                "fl_time": (fl.get("Time") or {}).get("time", ""),
            })
        finished = [x for x in results
                    if x["status"] in ("Finished", "Lapped")
                    or x["status"].startswith("+")
                    or "Lap" in x["status"]]
        dnf = [x for x in results if x not in finished]
        climb = None
        if results:
            gain = max(results, key=lambda x: x["grid"] - x["pos"] if x["pos"] and x["grid"] else -99)
            if gain["grid"] - gain["pos"] > 0:
                climb = {"driver": gain["driver"], "from": gain["grid"], "to": gain["pos"]}
        fl = next((x for x in results if x["fastest_lap"]), None)
        last_race = {
            "name": r.get("raceName", ""),
            "round": _int(r.get("round")),
            "date": r.get("date", ""),
            "circuit": r.get("Circuit", {}).get("circuitName", ""),
            "locality": r.get("Circuit", {}).get("Location", {}).get("locality", ""),
            "country": r.get("Circuit", {}).get("Location", {}).get("country", ""),
            "laps": _int(r.get("Results", [{}])[0].get("laps")),
            "results": results,
            "podium": results[:3],
            "fastest_lap": {"driver": fl["driver"], "team": fl["team"], "time": fl["fl_time"]} if fl else None,
            "biggest_climb": climb,
            "dnf": [{"driver": x["driver"], "team": x["team"], "status": x["status"]} for x in dnf],
            "pole_sitter": next((x["driver"] for x in results if x["grid"] == 1), ""),
        }

    # 赛历
    races_all = sched["MRData"]["RaceTable"].get("Races", [])
    races_total = len(races_all)
    next_race = None
    remaining_races = []
    for r in races_all:
        if _int(r.get("round")) > rnd:
            item = {
                "name": r.get("raceName", ""),
                "round": _int(r.get("round")),
                "date": r.get("date", ""),
                "circuit": r.get("Circuit", {}).get("circuitName", ""),
                "locality": r.get("Circuit", {}).get("Location", {}).get("locality", ""),
                "country": r.get("Circuit", {}).get("Location", {}).get("country", ""),
            }
            remaining_races.append(item)
            if next_race is None:
                next_race = item

    return {
        "sport": "f1",
        "title": "F1 一级方程式",
        "season": season,
        "round": rnd,
        "races_total": races_total,
        "races_done": rnd,
        "remaining": max(races_total - rnd, 0),
        "remaining_races": remaining_races,
        "drivers": drivers,
        "constructors": constructors,
        "last_race": last_race,
        "next_race": next_race,
        "meta": _merge_meta(m1, m2, m3, m4),
    }


# --------------------------------------------------------------------------- #
# NBA
# --------------------------------------------------------------------------- #

def _espn_stat(entry: dict, name: str, default=None, prefer_display: bool = False):
    for s in entry.get("stats", []):
        if s.get("name") == name:
            if prefer_display and s.get("displayValue") not in (None, ""):
                return s["displayValue"]
            v = s.get("value")
            if v is not None:
                return v
            dv = s.get("displayValue")
            try:
                return float(dv)
            except (TypeError, ValueError):
                return dv
    return default


_STATUS_CN = {
    "Final": "已结束", "Full Time": "已结束", "FT": "已结束",
    "Scheduled": "未开始", "Pre-Game": "未开始",
    "In Progress": "进行中", "Halftime": "中场", "End of Period": "节间",
}


def _cn_status(desc: str) -> str:
    return _STATUS_CN.get(desc, desc or "")


def _local_time(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return iso[:16]


def nba_snapshot(force: bool = False) -> dict:
    st, m1 = _get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
                  "nba_standings", force=force)
    sb, m2 = _get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                  "nba_scoreboard", force=force)

    conferences = []
    all_teams = []
    for child in st.get("children", []):
        teams = []
        for e in child.get("standings", {}).get("entries", []):
            team = e.get("team", {})
            row = {
                "id": team.get("id", ""),
                "name": team.get("displayName", ""),
                "short": team.get("abbreviation", ""),
                "logo": (team.get("logos") or [{}])[0].get("href", ""),
                "wins": _int(_espn_stat(e, "wins")),
                "losses": _int(_espn_stat(e, "losses")),
                "win_pct": _num(_espn_stat(e, "winPercent")),
                "gb": _num(_espn_stat(e, "gamesBehind")),
                "streak": str(_espn_stat(e, "streak", "", prefer_display=True) or ""),
                "pf": _num(_espn_stat(e, "avgPointsFor")),
                "pa": _num(_espn_stat(e, "avgPointsAgainst")),
                "diff": _num(_espn_stat(e, "pointDifferential")),
                "seed": _int(_espn_stat(e, "playoffSeed")),
                "last10": str(_espn_stat(e, "Last Ten Games", "") or ""),
            }
            teams.append(row)
        teams.sort(key=lambda r: (r["seed"] or 99, -r["win_pct"]))
        conferences.append({"name": child.get("name", ""), "teams": teams})
        all_teams.extend(teams)

    games = []
    for ev in sb.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        by_side = {t.get("homeAway"): t for t in comp.get("competitors", [])}
        if "home" not in by_side or "away" not in by_side:
            continue
        home, away = by_side["home"], by_side["away"]
        status = ((comp.get("status") or {}).get("type") or {}).get("description", "")
        games.append({
            "date": ev.get("date", ""),
            "time": _local_time(ev.get("date", "")),
            "status": _cn_status(status),
            "home": home["team"].get("displayName", ""),
            "away": away["team"].get("displayName", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
            "home_record": (home.get("records") or [{}])[0].get("summary", ""),
            "away_record": (away.get("records") or [{}])[0].get("summary", ""),
            "venue": (comp.get("venue") or {}).get("fullName", ""),
        })

    season = (sb.get("leagues") or [{}])[0].get("season", {}).get("displayName", "") or \
             (st.get("season") or {}).get("year", "")

    east = next((c["teams"] for c in conferences if "Eastern" in c["name"]), [])
    west = next((c["teams"] for c in conferences if "Western" in c["name"]), [])
    league = {
        "leader": max(all_teams, key=lambda r: r["win_pct"]) if all_teams else {},
        "best_offense": max(all_teams, key=lambda r: r["pf"]) if all_teams else {},
        "best_defense": min([t for t in all_teams if t["pa"] > 0], key=lambda r: r["pa"], default={}),
        "best_diff": max(all_teams, key=lambda r: r["diff"]) if all_teams else {},
        "worst": min(all_teams, key=lambda r: r["win_pct"]) if all_teams else {},
        "east_leader": east[0] if east else {},
        "west_leader": west[0] if west else {},
        "east_avg_pct": sum(t["win_pct"] for t in east) / len(east) if east else 0,
        "west_avg_pct": sum(t["win_pct"] for t in west) / len(west) if west else 0,
        "east_top8_pct": sum(t["win_pct"] for t in east[:8]) / len(east[:8]) if east else 0,
        "west_top8_pct": sum(t["win_pct"] for t in west[:8]) / len(west[:8]) if west else 0,
        "east_playin_gap": (east[7]["wins"] - east[7]["losses"]) - (east[9]["wins"] - east[9]["losses"])
                          if len(east) >= 10 else 0,
        "west_playin_gap": (west[7]["wins"] - west[7]["losses"]) - (west[9]["wins"] - west[9]["losses"])
                          if len(west) >= 10 else 0,
    }

    return {
        "sport": "nba",
        "title": "NBA 常规赛",
        "season": str(season),
        "conferences": conferences,
        "all_teams": all_teams,
        "league": league,
        "games": games,
        "meta": _merge_meta(m1, m2),
    }


# --------------------------------------------------------------------------- #
# 英超
# --------------------------------------------------------------------------- #

_NOTE_CN = {
    "Champions League": "欧冠区",
    "Europa League": "欧联区",
    "Conference League": "欧协联区",
    "Relegation": "降级区",
}


def _note_cn(note: dict) -> str:
    desc = (note or {}).get("description", "")
    for en, cn in _NOTE_CN.items():
        if en.lower() in desc.lower():
            return cn
    return desc


def epl_snapshot(force: bool = False) -> dict:
    st, m1 = _get("https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings",
                  "epl_standings", force=force)
    sb, m2 = _get("https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard",
                  "epl_scoreboard", force=force)

    child = (st.get("children") or [{}])[0]
    season = child.get("name", "")
    rows = []
    for e in child.get("standings", {}).get("entries", []):
        team = e.get("team", {})
        gp = _int(_espn_stat(e, "gamesPlayed"))
        pts = _int(_espn_stat(e, "points"))
        rows.append({
            "id": team.get("id", ""),
            "team": team.get("displayName", ""),
            "short": team.get("abbreviation", ""),
            "logo": (team.get("logos") or [{}])[0].get("href", ""),
            "pos": _int(_espn_stat(e, "rank", 99)),
            "gp": gp,
            "w": _int(_espn_stat(e, "wins")),
            "d": _int(_espn_stat(e, "ties")),
            "l": _int(_espn_stat(e, "losses")),
            "gf": _int(_espn_stat(e, "pointsFor")),
            "ga": _int(_espn_stat(e, "pointsAgainst")),
            "gd": _int(_espn_stat(e, "pointDifferential")),
            "pts": pts,
            "ppg": round(pts / gp, 2) if gp else 0.0,
            "rank_change": _int(_espn_stat(e, "rankChange")),
            "note": _note_cn(e.get("note") or {}),
        })
    rows.sort(key=lambda r: r["pos"])

    matches = []
    for ev in sb.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        by_side = {t.get("homeAway"): t for t in comp.get("competitors", [])}
        if "home" not in by_side or "away" not in by_side:
            continue
        home, away = by_side["home"], by_side["away"]
        status = ((comp.get("status") or {}).get("type") or {}).get("description", "")
        matches.append({
            "date": ev.get("date", ""),
            "time": _local_time(ev.get("date", "")),
            "status": _cn_status(status),
            "home": home["team"].get("displayName", ""),
            "away": away["team"].get("displayName", ""),
            "home_score": home.get("score", ""),
            "away_score": away.get("score", ""),
            "venue": (comp.get("venue") or {}).get("fullName", ""),
        })

    table = rows
    leader = table[0] if table else {}
    second = table[1] if len(table) > 1 else {}
    best_attack = max(table, key=lambda r: r["gf"], default={})
    best_defense = min([r for r in table if r["gp"] > 0], key=lambda r: r["ga"], default={})
    return {
        "sport": "epl",
        "title": "英超联赛",
        "season": season or "2026-27",
        "table": table,
        "leader": leader,
        "second": second,
        "best_attack": best_attack,
        "best_defense": best_defense,
        "matches": matches,
        "meta": _merge_meta(m1, m2),
    }


# --------------------------------------------------------------------------- #
# 高尔夫（PGA / DP World / LPGA）
# --------------------------------------------------------------------------- #

GOLF_TOURS = {
    "pga": "PGA 巡回赛",
    "eur": "DP World 巡回赛",
    "lpga": "LPGA 巡回赛",
}


def _golf_score_to_par(value) -> int | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ("", "-", "--"):
        return None
    if s in ("E", "EVEN", "0"):
        return 0
    try:
        return int(s.replace("+", ""))
    except ValueError:
        return None


def _parse_golf_event(ev: dict) -> dict:
    comp = (ev.get("competitions") or [{}])[0]
    status = ((ev.get("status") or {}).get("type") or {})
    final = bool(status.get("completed"))
    competitors = comp.get("competitors", [])
    board = []
    for c in competitors:
        score = _golf_score_to_par(c.get("score"))
        if score is None:
            continue
        rounds = []
        for ls in c.get("linescores", [])[:4]:          # 只取正赛 4 轮，排除加洞赛
            period = _int(ls.get("period"))
            value = ls.get("value")
            if 1 <= period <= 4 and isinstance(value, (int, float)) and 50 <= value <= 100:
                rounds.append(int(value))
        board.append({
            "rank": _int(c.get("order"), len(board) + 1),
            "name": (c.get("athlete") or {}).get("displayName", ""),
            "country": ((c.get("athlete") or {}).get("flag") or {}).get("alt", ""),
            "score": score,
            "rounds": rounds,
        })
    board.sort(key=lambda r: (r["score"], r["rank"]))
    for i, row in enumerate(board, 1):
        row["rank"] = i

    best_round = None
    for row in board:
        if row["rounds"]:
            low = min(row["rounds"])
            if best_round is None or low < best_round["strokes"]:
                best_round = {"player": row["name"], "strokes": low}

    return {
        "name": ev.get("name", ""),
        "date": ev.get("date", ""),
        "end_date": ev.get("endDate", ""),
        "status": _cn_status(status.get("description", "")),
        "final": final,
        "field_size": len(competitors),
        "leaderboard": board[:15],
        "winner": board[0] if board else None,
        "runner_up": board[1] if len(board) > 1 else None,
        "margin": (board[1]["score"] - board[0]["score"]) if len(board) > 1 else None,
        "best_round": best_round,
        "link": next((l.get("href", "") for l in ev.get("links", [])), ""),
    }


def _golf_tour(slug: str, force: bool = False) -> dict:
    cur, m1 = _get(f"https://site.api.espn.com/apis/site/v2/sports/golf/{slug}/scoreboard",
                   f"golf_{slug}_current", force=force)
    current_events = [_parse_golf_event(e) for e in cur.get("events", [])]

    last_completed = next((e for e in current_events if e["final"] and e["leaderboard"]), None)

    # 当前没有已结束赛事时，从最近的周日开始向前回溯 4 周找上一站
    if last_completed is None:
        today = datetime.now()
        last_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
        for week in range(4):
            day = (last_sunday - timedelta(days=7 * week)).strftime("%Y%m%d")
            try:
                past, _m = _get(
                    f"https://site.api.espn.com/apis/site/v2/sports/golf/{slug}/scoreboard?dates={day}",
                    f"golf_{slug}_{day}", ttl=GOLF_PAST_TTL, force=False)  # 已结束赛事不再重抓
            except Exception:
                continue
            for e in past.get("events", []):
                parsed = _parse_golf_event(e)
                if parsed["final"] and parsed["leaderboard"]:
                    last_completed = parsed
                    break
            if last_completed:
                break

    current = current_events[0] if current_events else None
    return {
        "slug": slug,
        "tour": GOLF_TOURS.get(slug, slug),
        "current": current,
        "last_completed": last_completed,
        "meta": m1,
    }


def golf_snapshot(force: bool = False) -> dict:
    tours = [_golf_tour(slug, force=force) for slug in GOLF_TOURS]
    metas = [t["meta"] for t in tours]
    return {
        "sport": "golf",
        "title": "高尔夫全球巡回赛",
        "season": datetime.now().year,
        "tours": tours,
        "meta": _merge_meta(*metas),
    }


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #

SNAPSHOT_FUNCS = {
    "f1": f1_snapshot,
    "nba": nba_snapshot,
    "epl": epl_snapshot,
    "golf": golf_snapshot,
}

SPORT_NAMES = {
    "f1": "F1 一级方程式",
    "nba": "NBA",
    "epl": "英超联赛",
    "golf": "高尔夫全球巡回赛",
}


def all_snapshots(force: bool = False) -> dict:
    out = {}
    for key, fn in SNAPSHOT_FUNCS.items():
        try:
            out[key] = fn(force=force)
        except Exception as exc:
            out[key] = {"sport": key, "title": SPORT_NAMES[key],
                        "error": f"{type(exc).__name__}: {exc}", "meta": {}}
    return out


if __name__ == "__main__":
    snaps = all_snapshots()
    for key, snap in snaps.items():
        if "error" in snap:
            print(f"[{key}] ERROR {snap['error']}")
            continue
        print(f"[{key}] OK  source={snap['meta'].get('source')} "
              f"fetched={snap['meta'].get('fetched_at')}")
    f1 = snaps["f1"]
    print("F1:", f1["season"], "round", f1["round"], "/", f1["races_total"],
          "| leader", f1["drivers"][0]["name"], f1["drivers"][0]["points"])
    nba = snaps["nba"]
    print("NBA:", nba["season"], "| leader", nba["league"]["leader"]["name"],
          nba["league"]["leader"]["wins"], "-", nba["league"]["leader"]["losses"])
    epl = snaps["epl"]
    print("EPL:", epl["season"], "| leader", epl["leader"]["team"], epl["leader"]["pts"])
    golf = snaps["golf"]
    for t in golf["tours"]:
        lc = t["last_completed"] or {}
        print("Golf", t["tour"], "| last:", lc.get("name"), "| winner:",
              (lc.get("winner") or {}).get("name"), (lc.get("winner") or {}).get("score"))
