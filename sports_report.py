#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛事分析报告生成器。

用法：
    python sports_report.py --sport all
    python sports_report.py --sport f1 --out reports/f1.md
    python sports_report.py --sport nba --llm          # 追加 AI 深度解读
    python sports_report.py --ask "现在英超前四是谁？"  # 数据问答

报告 = 规则统计（数字全部来自数据层） + 可选的大模型解读（qwen-flash）。
大模型只拿到事实 JSON，要求不编造；失败时自动退回纯统计报告。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import sports_data as sd

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "sports_reports"
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

SOURCE_CN = {
    "f1": "Jolpica F1 API（Ergast 兼容）",
    "nba": "ESPN NBA 数据接口",
    "epl": "ESPN 英超数据接口",
    "golf": "ESPN 高尔夫数据接口",
    "all": "Jolpica F1 API、ESPN 公开数据接口",
}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _num(v, nd=1) -> str:
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _signed(v, nd=1) -> str:
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v)):+d}"
        return f"{v:+.{nd}f}"
    return f"{v:+d}"


def _par(v) -> str:
    """高尔夫相对标准杆：0 → E，正数 → +N，负数 → -N。"""
    if v is None:
        return "-"
    if v == 0:
        return "E"
    return f"{int(v):+d}"


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def _meta_desc(meta: dict) -> str:
    src = (meta or {}).get("source", "?")
    label = {"live": "实时抓取", "cache": "本地缓存", "stale": "旧缓存（网络异常回退）"}.get(src, src)
    ts = (meta or {}).get("fetched_at", "")
    return f"{label}，{ts}" if ts else label


def _date_cn(iso: str) -> str:
    return (iso or "")[:10]


# --------------------------------------------------------------------------- #
# 事实提炼（报告与问答的事实基础）
# --------------------------------------------------------------------------- #

def f1_facts(s: dict) -> dict:
    ds, cs = s.get("drivers") or [], s.get("constructors") or []
    if not ds:
        raise ValueError("F1 车手积分榜为空")
    leader = ds[0]
    second = ds[1] if len(ds) > 1 else None
    gap = leader["points"] - second["points"] if second else 0
    remaining = s.get("remaining", 0)
    done = max(s.get("round", 1), 1)
    max_normal = remaining * 25
    max_sprint = remaining * 33
    if remaining == 0:
        driver_status = "赛季已结束"
    elif gap > max_sprint:
        driver_status = f"已提前锁定车手总冠军（剩余 {remaining} 站即使全部为冲刺周末，理论最高也只有 {max_sprint:.0f} 分可追）"
    elif gap > max_normal:
        driver_status = f"按常规分站积分已锁定，但若剩余均为冲刺周末仍有理论悬念（常规上限 {max_normal:.0f} 分，冲刺上限 {max_sprint:.0f} 分）"
    else:
        driver_status = f"冠军悬念仍在（剩余 {remaining} 站，追赶者理论最高可追 {max_sprint:.0f} 分）"

    c_leader = cs[0] if cs else {}
    c_second = cs[1] if len(cs) > 1 else {}
    cgap = c_leader.get("points", 0) - c_second.get("points", 0)
    c_max_normal, c_max_sprint = remaining * 43, remaining * 58
    if remaining == 0:
        c_status = "赛季已结束"
    elif cgap > c_max_sprint:
        c_status = "已提前锁定车队总冠军"
    elif cgap > c_max_normal:
        c_status = "按常规分站已锁定，冲刺周末理论上仍有悬念"
    else:
        c_status = f"车队冠军悬念仍在（剩余 {remaining} 站，常规理论最高 {c_max_normal:.0f} 分）"

    lr = s.get("last_race") or {}
    podium = lr.get("podium") or []
    winner = podium[0] if podium else {}
    team_map = {}
    for d in ds:
        team_map.setdefault(d["team"], []).append(d["name"])
    return {
        "season": s.get("season", ""),
        "round": s.get("round", 0),
        "races_total": s.get("races_total", 0),
        "remaining": remaining,
        "leader": leader,
        "second": second,
        "top_drivers": ds[:5],
        "gap": gap,
        "points_per_race": round(leader["points"] / done, 2),
        "win_rate": leader["wins"] / done,
        "win_rate_note": f"{leader['wins']} 胜 / 已完成 {done} 站 = {leader['wins'] / done * 100:.1f}%",
        "team_driver_map": team_map,
        "driver_status": driver_status,
        "max_normal": max_normal,
        "max_sprint": max_sprint,
        "constructors": cs[:4],
        "c_leader": c_leader,
        "c_second": c_second,
        "c_gap": cgap,
        "c_max_normal": c_max_normal,
        "c_max_sprint": c_max_sprint,
        "c_status": c_status,
        "notes": {
            "gap": "当前 P1 与 P2 的积分差",
            "max_normal": "剩余所有分站合计理论最高分（每站 25 分，未计冲刺赛）",
            "max_sprint": "剩余所有分站合计理论最高分（每站 33 分，含冲刺赛 8 分）",
            "c_max_normal": "车队剩余所有分站合计理论最高分（每站 43 分）",
            "c_max_sprint": "车队剩余所有分站合计理论最高分（每站 58 分，含冲刺赛）",
        },
        "last_race": lr,
        "race_winner": winner,
        "next_race": s.get("next_race") or {},
        "remaining_races": s.get("remaining_races") or [],
    }


def nba_facts(s: dict) -> dict:
    teams = s.get("all_teams") or []
    if not teams:
        raise ValueError("NBA 球队数据为空")
    east = next((c["teams"] for c in s.get("conferences", []) if "Eastern" in c["name"]), [])
    west = next((c["teams"] for c in s.get("conferences", []) if "Western" in c["name"]), [])
    league = s.get("league") or {}

    off_rank = {t["id"]: i + 1 for i, t in enumerate(sorted(teams, key=lambda x: x["pf"], reverse=True))}
    def_rank = {t["id"]: i + 1 for i, t in enumerate(sorted(teams, key=lambda x: x["pa"]))}
    net_rank = {t["id"]: i + 1 for i, t in enumerate(
        sorted(teams, key=lambda x: x["pf"] - x["pa"], reverse=True))}
    avg_pf = sum(t["pf"] for t in teams) / len(teams)
    avg_pa = sum(t["pa"] for t in teams) / len(teams)

    def _gap8_10(conf):
        if len(conf) < 10:
            return 0.0
        a, b = conf[7], conf[9]
        return ((a["wins"] - a["losses"]) - (b["wins"] - b["losses"])) / 2

    win_streaks = []
    for t in teams:
        st = (t.get("streak") or "").upper()
        if st.startswith("W"):
            try:
                win_streaks.append((int(st[1:]), t))
            except ValueError:
                pass
    win_streaks.sort(key=lambda x: -x[0])

    games = s.get("games") or []
    finals = [g for g in games if g["status"] == "已结束" and g.get("home_score") not in ("", None)]
    notable = []
    for g in finals:
        try:
            hs, as_ = int(g["home_score"]), int(g["away_score"])
        except (TypeError, ValueError):
            continue
        notable.append((abs(hs - as_), g, hs, as_))
    notable.sort(key=lambda x: -x[0])

    return {
        "season": s.get("season", ""),
        "league_leader": league.get("leader", {}),
        "east_leader": league.get("east_leader", {}),
        "west_leader": league.get("west_leader", {}),
        "best_offense": league.get("best_offense", {}),
        "best_defense": league.get("best_defense", {}),
        "best_diff": league.get("best_diff", {}),
        "worst": league.get("worst", {}),
        "east_top8_pct": league.get("east_top8_pct", 0),
        "west_top8_pct": league.get("west_top8_pct", 0),
        "east_gap_8_10": _gap8_10(east),
        "west_gap_8_10": _gap8_10(west),
        "leader_off_rank": off_rank.get(league.get("leader", {}).get("id"), 0),
        "leader_def_rank": def_rank.get(league.get("leader", {}).get("id"), 0),
        "def_leader_off_rank": off_rank.get(league.get("best_defense", {}).get("id"), 0),
        "def_leader_def_rank": def_rank.get(league.get("best_defense", {}).get("id"), 0),
        "leader_net_rank": net_rank.get(league.get("leader", {}).get("id"), 0),
        "avg_pf": avg_pf,
        "avg_pa": avg_pa,
        "win_streaks": [{"team": t["name"], "streak": f"W{n}", "record": f"{t['wins']}-{t['losses']}"}
                        for n, t in win_streaks[:3]],
        "east": east,
        "west": west,
        "games": games,
        "notable_games": [{"margin": mg,
                           "score": f"{g['away']} {g['away_score']} : {g['home_score']} {g['home']}"}
                          for mg, g, hs, as_ in notable[:3]],
    }


def epl_facts(s: dict) -> dict:
    table = s.get("table") or []
    if not table:
        raise ValueError("英超积分榜为空")
    leader = table[0]
    second = table[1] if len(table) > 1 else {}
    fourth = table[3] if len(table) > 3 else {}
    fifth = table[4] if len(table) > 4 else {}
    safe = table[16] if len(table) > 16 else {}
    drop = table[17] if len(table) > 17 else {}

    def _rank(key, reverse=True):
        order = sorted(table, key=lambda r: r[key], reverse=reverse)
        return {r["id"]: i + 1 for i, r in enumerate(order)}

    gf_rank, ga_rank, gd_rank = _rank("gf"), _rank("ga", reverse=False), _rank("gd")
    return {
        "leader_gf_rank": gf_rank.get(leader.get("id"), 0),
        "leader_ga_rank": ga_rank.get(leader.get("id"), 0),
        "leader_gd_rank": gd_rank.get(leader.get("id"), 0),
        "season": s.get("season", ""),
        "leader": leader,
        "second": second,
        "gap": leader.get("pts", 0) - second.get("pts", 0),
        "best_attack": s.get("best_attack", {}),
        "best_defense": s.get("best_defense", {}),
        "fourth": fourth,
        "fifth": fifth,
        "top4_gap": fourth.get("pts", 0) - fifth.get("pts", 0),
        "safe": safe,
        "drop": drop,
        "relegation_gap": safe.get("pts", 0) - drop.get("pts", 0),
        "pace38": round(leader.get("ppg", 0) * 38, 1),
        "table": table,
        "matches": s.get("matches") or [],
    }


def golf_facts(s: dict) -> dict:
    tours = []
    for t in s.get("tours", []):
        lc = t.get("last_completed") or {}
        winner = lc.get("winner") or {}
        cur = t.get("current") or {}
        tours.append({
            "tour": t.get("tour", ""),
            "last_event": lc.get("name", ""),
            "last_date": _date_cn(lc.get("date", "")),
            "winner": winner.get("name", ""),
            "winner_country": winner.get("country", ""),
            "winner_score": winner.get("score"),
            "winner_rounds": winner.get("rounds", []),
            "runner_up": (lc.get("runner_up") or {}).get("name", ""),
            "runner_up_score": (lc.get("runner_up") or {}).get("score"),
            "margin": lc.get("margin"),
            "best_round": lc.get("best_round") or {},
            "field_size": lc.get("field_size", 0),
            "top5": (lc.get("leaderboard") or [])[:5],
            "current_event": cur.get("name", ""),
            "current_date": _date_cn(cur.get("date", "")),
            "current_status": cur.get("status", ""),
            "current_field": cur.get("field_size", 0),
        })
    return {"season": s.get("season", ""), "tours": tours}


def build_facts(sport: str, snap: dict) -> dict:
    return {"f1": f1_facts, "nba": nba_facts, "epl": epl_facts, "golf": golf_facts}[sport](snap)


# --------------------------------------------------------------------------- #
# 规则统计报告
# --------------------------------------------------------------------------- #

def f1_body(f: dict, h: str = "##") -> str:
    lr, nr = f["last_race"], f["next_race"]
    leader, second = f["leader"], f["second"] or {}
    c_leader, c_second = f["c_leader"], f["c_second"] or {}
    winner = f["race_winner"]
    lines = [
        f"{h} 一、核心结论",
        f"**{leader['name']}（{leader['team']}）以 {_num(leader['points'])} 分领跑车手积分榜**，"
        f"领先第二名 {second.get('name', '-')} **{_num(f['gap'])} 分**。{f['driver_status']}。",
        "",
        f"车队方面，**{c_leader.get('name', '-')}** 以 {_num(c_leader.get('points', 0))} 分领跑，"
        f"领先 {c_second.get('name', '-')} {_num(f['c_gap'])} 分；{f['c_status']}。",
    ]
    if lr:
        lines += ["", f"上一站 **{lr.get('name', '')}**（{lr.get('date', '')}，{lr.get('circuit', '')}），"
                      f"{winner.get('driver', '-')} 从 P{winner.get('grid', '-')} 起步夺冠。"]
    if nr:
        lines += ["", f"下一站 **{nr.get('name', '')}** 将于 {nr.get('date', '')} 在 "
                      f"{nr.get('circuit', '')}（{nr.get('locality', '')}）进行。"]
    lines += ["", f"{h} 二、关键数据", ""]
    rows = [
        ["赛季进度", f"第 {f['round']}/{f['races_total']} 站，剩余 {f['remaining']} 站"],
        ["车手榜首", f"{leader['name']}（{leader['team']}）{_num(leader['points'])} 分，{leader['wins']} 胜"],
        ["场均得分", f"{_num(f['points_per_race'], 2)} 分/站，胜率 {_pct(f['win_rate'])}"],
        ["积分差距", f"P1 - P2：{_num(f['gap'])} 分"],
        ["车队榜首", f"{c_leader.get('name', '-')} {_num(c_leader.get('points', 0))} 分"],
        ["剩余理论分", f"车手 {f['max_normal']:.0f} 分（未计冲刺赛），车队 {f['c_max_normal']:.0f} 分"],
    ]
    lines.append(_table(["项目", "数据"], rows))
    if lr and lr.get("podium"):
        lines += ["", "**上一站领奖台**", ""]
        podium_rows = [[p["pos"], p["driver"], p["team"], f"P{p['grid']} 起步", p.get("time") or p.get("status", "")]
                       for p in lr["podium"]]
        lines.append(_table(["名次", "车手", "车队", "发车位", "成绩"], podium_rows))
        extras = []
        if lr.get("pole_sitter"):
            extras.append(f"杆位：{lr['pole_sitter']}")
        if lr.get("fastest_lap"):
            fl = lr["fastest_lap"]
            extras.append(f"最快圈：{fl['driver']}（{fl['team']}）{fl['time']}")
        if lr.get("biggest_climb"):
            cl = lr["biggest_climb"]
            extras.append(f"最大爬升：{cl['driver']} P{cl['from']} → P{cl['to']}")
        if lr.get("dnf"):
            extras.append("退赛：" + "、".join(x["driver"] for x in lr["dnf"]))
        if extras:
            lines += ["", "；".join(extras) + "。"]

    lines += ["", f"{h} 三、深度分析", ""]
    lines.append(
        f"- **领跑者效率**：{leader['name']} 已完成 {f['round']} 站拿到 {_num(leader['points'])} 分，"
        f"场均 {_num(f['points_per_race'], 2)} 分；{leader['wins']} 个分站冠军对应胜率 {_pct(f['win_rate'])}。")
    if len(f["top_drivers"]) >= 3:
        third = f["top_drivers"][2]
        lines.append(
            f"- **追赶集团**：P2 {second['name']} 落后 {_num(f['gap'])} 分，"
            f"P3 {third['name']}（{third['team']}）{_num(third['points'])} 分，"
            f"与榜首相差 {_num(leader['points'] - third['points'])} 分。")
    # 上一站复盘
    if lr and winner:
        bits = [f"- **上一站复盘**：{winner.get('driver')} 从 P{winner.get('grid')} 起步夺冠"]
        if lr.get("biggest_climb"):
            cl = lr["biggest_climb"]
            bits.append(f"{cl['driver']} 从 P{cl['from']} 追到 P{cl['to']}")
        if lr.get("fastest_lap"):
            bits.append(f"最快圈由 {lr['fastest_lap']['driver']} 做出")
        if lr.get("dnf"):
            bits.append(f"共有 {len(lr['dnf'])} 位车手退赛")
        lines.append("，".join(bits) + "。")
    # 车队格局
    if f["constructors"]:
        top = f["constructors"][:4]
        desc = "、".join(f"{c['name']} {_num(c['points'])} 分" for c in top)
        lines.append(f"- **车队格局**：{desc}；{c_leader.get('name', '')} 的领先优势为 {_num(f['c_gap'])} 分。")
    # 剩余赛程
    if f["remaining_races"]:
        races = "、".join(f"{r['name']}（{r['date']}）" for r in f["remaining_races"][:5])
        lines.append(f"- **剩余赛程**：{races}。")
    lines += ["", f"{h} 四、后续看点", ""]
    if nr:
        lines.append(f"- {nr.get('name', '下一站')} 将于 {nr.get('date', '')} 在 {nr.get('locality', '')} 打响，"
                     f"街道/高速赛道的特性可能放大车队间的差距。")
    if f["remaining"] and f["gap"] <= f["max_sprint"]:
        lines.append(f"- 车手冠军仍有数学悬念：{leader['name']} 需要在剩余 {f['remaining']} 站中"
                     f"守住 {_num(f['gap'])} 分优势。")
    elif f["remaining"]:
        lines.append(f"- 车手冠军已无数学悬念，剩余 {f['remaining']} 站的看点是分站冠军、车队亚军争夺与新人表现。")
    lines.append("- 关注冲刺赛周末的额外积分：若剩余赛历包含冲刺赛，理论分差上限会相应提高。")
    return "\n".join(lines)


def nba_body(f: dict, h: str = "##") -> str:
    ldr = f["league_leader"]
    lines = [
        f"{h} 一、核心结论",
        f"**{ldr.get('name', '-')} 以 {ldr.get('wins', 0)} 胜 {ldr.get('losses', 0)} 负（{_pct(ldr.get('win_pct', 0))}）领跑全联盟**，"
        f"场均净胜 {_signed(ldr.get('pf', 0) - ldr.get('pa', 0))} 分。",
        "",
        f"东部由 **{f['east_leader'].get('name', '-')}**（{f['east_leader'].get('wins', 0)}-{f['east_leader'].get('losses', 0)}）领跑，"
        f"西部由 **{f['west_leader'].get('name', '-')}**（{f['west_leader'].get('wins', 0)}-{f['west_leader'].get('losses', 0)}）领跑。",
        "",
        f"进攻端 **{f['best_offense'].get('name', '-')}** 场均 {_num(f['best_offense'].get('pf', 0))} 分联盟第一；"
        f"防守端 **{f['best_defense'].get('name', '-')}** 场均仅失 {_num(f['best_defense'].get('pa', 0))} 分。",
        "",
        f"{h} 二、关键数据",
        "",
        _table(["项目", "球队", "数据"], [
            ["联盟第一", ldr.get("name", "-"), f"{ldr.get('wins', 0)}-{ldr.get('losses', 0)}（{_pct(ldr.get('win_pct', 0))}）"],
            ["东部第一", f["east_leader"].get("name", "-"), f"{f['east_leader'].get('wins', 0)}-{f['east_leader'].get('losses', 0)}"],
            ["西部第一", f["west_leader"].get("name", "-"), f"{f['west_leader'].get('wins', 0)}-{f['west_leader'].get('losses', 0)}"],
            ["进攻第一", f["best_offense"].get("name", "-"), f"{_num(f['best_offense'].get('pf', 0))} 分/场"],
            ["防守第一", f["best_defense"].get("name", "-"), f"失 {_num(f['best_defense'].get('pa', 0))} 分/场"],
            ["场均净胜分第一", f["best_diff"].get("name", "-"),
             f"{_signed(f['best_diff'].get('pf', 0) - f['best_diff'].get('pa', 0))} 分"],
            ["联盟垫底", f["worst"].get("name", "-"), f"{f['worst'].get('wins', 0)}-{f['worst'].get('losses', 0)}"],
            ["东部前八平均胜率", "-", _pct(f["east_top8_pct"])],
            ["西部前八平均胜率", "-", _pct(f["west_top8_pct"])],
        ]),
        "",
        f"{h} 三、深度分析",
        "",
        f"- **联盟格局**：{ldr.get('name', '')} 的场均得分排名第 {f['leader_off_rank']}、场均失分排名第 {f['leader_def_rank']}，"
        f"属于{'攻防兼备' if f['leader_def_rank'] <= 10 else '攻强守弱'}的争冠形态。",
        f"- **东西部强度**：西部前八平均胜率 {_pct(f['west_top8_pct'])}，东部前八 {_pct(f['east_top8_pct'])}，"
        f"相差 {abs(f['west_top8_pct'] - f['east_top8_pct']) * 100:.1f} 个百分点。",
        f"- **附加赛线**：东部第 8 领先第 10 约 {f['east_gap_8_10']:.1f} 个胜场，西部约 {f['west_gap_8_10']:.1f} 个胜场。",
    ]
    if f["win_streaks"]:
        st = "、".join(f"{x['team']}（{x['streak']}，{x['record']}）" for x in f["win_streaks"])
        lines.append(f"- **当前连胜**：{st}。")
    if f["notable_games"]:
        games = "；".join(g["score"] for g in f["notable_games"])
        lines.append(f"- **近期最大分差比赛**：{games}。")
    lines += [
        f"- **攻防失衡的隐忧**：{f['best_defense'].get('name', '')} 场均失分联盟最少，但场均得分仅排第 {f['def_leader_off_rank']}；"
        f"季后赛面对高强度防守时，单靠防守的上限值得观察。",
        "",
        f"{h} 四、后续看点",
        "",
        "- 关注东西部前四的直接对话：这类比赛会同时影响排名与季后赛主场优势。",
        "- 关注第 8 至第 10 名的附加赛卡位战，胜场差仍在可追范围内。",
    ]
    if f["games"]:
        upcoming = [g for g in f["games"] if g["status"] != "已结束"]
        if upcoming:
            lines.append(f"- 最近一个比赛日有 {len(upcoming)} 场比赛待打，可重点观察连胜球队能否延续势头。")
        else:
            lines.append(f"- 最近一个比赛日的 {len(f['games'])} 场比赛已全部结束，下一轮赛程公布后可继续跟踪。")
    return "\n".join(lines)


def epl_body(f: dict, h: str = "##") -> str:
    leader, second = f["leader"], f["second"]
    lines = [
        f"{h} 一、核心结论",
        f"**{leader['team']} 以 {leader['pts']} 分领跑英超**，领先第二名 {second.get('team', '-')} "
        f"**{f['gap']} 分**；{leader['w']} 胜 {leader['d']} 平 {leader['l']} 负，场均 {leader['ppg']} 分。",
        "",
        f"进攻最强的是 **{f['best_attack'].get('team', '-')}**（{f['best_attack'].get('gf', 0)} 球），"
        f"防守最稳的是 **{f['best_defense'].get('team', '-')}**（仅失 {f['best_defense'].get('ga', 0)} 球）。",
        "",
        f"{h} 二、关键数据",
        "",
        _table(["排名", "球队", "场次", "胜-平-负", "进/失", "净胜球", "积分", "场均分"], [
            [r["pos"], r["team"], r["gp"], f"{r['w']}-{r['d']}-{r['l']}",
             f"{r['gf']}/{r['ga']}", _signed(r["gd"]), r["pts"], r["ppg"]]
            for r in f["table"][:6]
        ]),
        "",
        f"**降级区**：",
        "",
        _table(["排名", "球队", "积分", "净胜球"], [
            [r["pos"], r["team"], r["pts"], _signed(r["gd"])]
            for r in f["table"][-3:]
        ]),
        "",
        f"{h} 三、深度分析",
        "",
        f"- **争冠节奏**：{leader['team']} 场均 {leader['ppg']} 分，按此节奏 38 轮约 {f['pace38']} 分"
        f"（线性外推，仅作参照）。领先 {f['gap']} 分意味着还有约 {max(f['gap'], 0)} 分的容错空间。",
        f"- **争四集团**：第 4 名 {f['fourth'].get('team', '-')}（{f['fourth'].get('pts', 0)} 分）只领先第 5 名 "
        f"{f['fifth'].get('team', '-')}（{f['fifth'].get('pts', 0)} 分）{f['top4_gap']} 分，欧冠资格竞争胶着。",
        f"- **保级形势**：第 17 名 {f['safe'].get('team', '-')}（{f['safe'].get('pts', 0)} 分）领先第 18 名 "
        f"{f['drop'].get('team', '-')}（{f['drop'].get('pts', 0)} 分）仅 {f['relegation_gap']} 分，降级区边缘每轮都可能易主。",
        f"- **攻防对比**：{leader['team']} 进 {leader['gf']} 球（第 {f['leader_gf_rank']}）、"
        f"失 {leader['ga']} 球（第 {f['leader_ga_rank']}）、净胜球 {_signed(leader['gd'])}（第 {f['leader_gd_rank']}）；"
        f"{f['best_attack'].get('team', '')} 打进 {f['best_attack'].get('gf', 0)} 球（失 {f['best_attack'].get('ga', 0)} 球），"
        f"{f['best_defense'].get('team', '')} 只失 {f['best_defense'].get('ga', 0)} 球，两种建队思路形成直接对比。",
    ]
    if f["matches"]:
        recap = "；".join(f"{m['home']} {m['home_score']}-{m['away_score']} {m['away']}"
                          for m in f["matches"] if m["status"] == "已结束")
        if recap:
            lines.append(f"- **最近一轮赛果**：{recap}。")
    lines += [
        "",
        f"{h} 四、后续看点",
        "",
        f"- 关注 {leader['team']} 能否把场均 {leader['ppg']} 分的节奏保持到圣诞赛程，密集赛程通常是争冠的分水岭。",
        f"- 关注第 4 至第 6 名的欧冠资格卡位战，{f['top4_gap']} 分的差距一两轮就会反转。",
        f"- 关注降级区：第 18 名 {f['drop'].get('team', '')} 与安全线只差 {f['relegation_gap']} 分，保级对手之间的六分之战价值极高。",
    ]
    return "\n".join(lines)


def golf_body(f: dict, h: str = "##") -> str:
    tours = f["tours"]
    lines = [f"{h} 一、核心结论"]
    for t in tours:
        if t["winner"]:
            margin = f"，{t['margin']} 杆优势" if t["margin"] is not None else ""
            lines.append(
                f"- **{t['tour']}**：上一站 {t['last_event']}（{t['last_date']}），"
                f"{t['winner']} 以 {_par(t['winner_score'])} 杆夺冠{margin}。")
        if t["current_event"]:
            lines.append(f"  下一站 {t['current_event']} 将于 {t['current_date']} 开赛，"
                         f"参赛阵容 {t['current_field']} 人。")
    lines += ["", f"{h} 二、关键数据", ""]
    rows = []
    for t in tours:
        rows.append([
            t["tour"], t["last_event"],
            t["winner"] or "-",
            _par(t["winner_score"]),
            f"{t['margin']} 杆" if t["margin"] is not None else "-",
            f"{t['best_round'].get('player', '-')} {t['best_round'].get('strokes', '-')} 杆"
            if t["best_round"] else "-",
            t["field_size"],
        ])
    lines.append(_table(["巡回赛", "上一站", "冠军", "成绩", "胜差", "最低单轮", "阵容"], rows))
    lines += ["", f"{h} 三、深度分析", ""]
    margins = [(t["margin"], t["tour"], t["winner"]) for t in tours if t["margin"] is not None]
    if margins:
        mx = max(margins)
        mn = min(margins)
        lines.append(f"- **胜差对比**：{mx[1]}的 {mx[2]} 以 {mx[0]} 杆优势夺冠，是本周最大胜差；"
                     f"{mn[1]}的 {mn[2]} 仅赢 {mn[0]} 杆，竞争最胶着。")
    lows = [(t["best_round"].get("strokes", 999), t["best_round"].get("player", ""), t["tour"])
            for t in tours if t["best_round"]]
    if lows:
        low = min(lows)
        lines.append(f"- **单轮火力**：本周最低单轮是 {low[1]} 在{low[2]}打出的 {low[0]} 杆。")
    winners = [t for t in tours if t["winner"]]
    if winners:
        rounds_txt = "；".join(
            f"{t['winner']} {'-'.join(str(x) for x in t['winner_rounds'])}"
            for t in winners if t["winner_rounds"])
        lines.append(f"- **冠军四轮走势**：{rounds_txt}。")
    lines.append("- **口径提醒**：不同球场标准杆、长度与天气差异很大，跨巡回赛的绝对杆数不能直接横向比较，"
                 "胜差与单轮杆数更适合作为赛事内部的参照。")
    lines += ["", f"{h} 四、后续看点", ""]
    for t in tours:
        if t["current_event"]:
            finale = "（赛季收官战）" if "tour championship" in t["current_event"].lower() else ""
            lines.append(f"- {t['tour']}：{t['current_event']}{finale}，{t['current_date']} 开赛，"
                         f"{t['current_field']} 人的阵容值得关注领先榜变化。")
    return "\n".join(lines)


BODY_BUILDERS = {"f1": f1_body, "nba": nba_body, "epl": epl_body, "golf": golf_body}


# --------------------------------------------------------------------------- #
# LLM（通义千问）
# --------------------------------------------------------------------------- #

def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_key() -> str:
    key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text("utf-8").splitlines():
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def llm_available() -> bool:
    return bool(_load_key())


def _llm_chat(system: str, user: str, max_tokens: int = 900, temperature: float = 0.5):
    """返回 (text, error)。"""
    key = _load_key()
    if not key:
        return None, "未配置 DASHSCOPE_API_KEY"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=LLM_BASE, timeout=90.0)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = _strip_fence((resp.choices[0].message.content or "").strip())
        return (text or None), ("" if text else "空响应")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"




def _slim_team(team: dict) -> dict:
    if not isinstance(team, dict):
        return team
    return {k: v for k, v in team.items()
            if k not in ("id", "logo", "uid", "links", "link", "abbreviation", "isActive")}


def compact_facts(facts: dict) -> dict:
    """压缩事实体积，保证四项赛事都能进入 LLM 上下文。"""
    out = {}
    if "f1" in facts:
        f = dict(facts["f1"])
        f["leader"] = _slim_team(f.get("leader", {}))
        f["second"] = _slim_team(f.get("second", {}))
        f["top_drivers"] = [_slim_team(d) for d in f.get("top_drivers", [])[:6]]
        f["constructors"] = [_slim_team(c) for c in f.get("constructors", [])[:5]]
        f["remaining_races"] = f.get("remaining_races", [])[:5]
        lr = dict(f.get("last_race") or {})
        lr.pop("results", None)          # 完整 22 车成绩对解读无必要，保留领奖台与关键信息
        f["last_race"] = lr
        out["f1"] = f
    if "nba" in facts:
        f = dict(facts["nba"])
        f.pop("east", None)
        f.pop("west", None)
        for key in ("league_leader", "east_leader", "west_leader", "best_offense",
                    "best_defense", "best_diff", "worst"):
            f[key] = _slim_team(f.get(key, {}))
        f["games"] = f.get("games", [])[:8]
        out["nba"] = f
    if "epl" in facts:
        f = dict(facts["epl"])
        table = [_slim_team(r) for r in f.get("table", [])]
        f["table"] = table[:6] + table[-4:]
        f["leader"] = _slim_team(f.get("leader", {}))
        f["best_attack"] = _slim_team(f.get("best_attack", {}))
        f["best_defense"] = _slim_team(f.get("best_defense", {}))
        for key in ("second", "fourth", "fifth", "safe", "drop"):
            f.pop(key, None)
        out["epl"] = f
    if "golf" in facts:
        out["golf"] = facts["golf"]
    return out


_FACTCHECK_SYSTEM = ("你是一名严格的事实核查编辑。你只能依据给定的事实 JSON 修订稿件，"
                    "宁可删减也绝不添加；不得引入 JSON 之外的任何信息。")


def _clean_llm_text(text: str) -> str:
    """去掉模型偶发输出的字段名引用与反引号。"""
    import re
    text = re.sub(r"[（(]\s*`[^`]+`[^）)]*[）)]", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    return text.replace("`", "").strip()


def _qa_clean(text: str) -> str:
    """问答场景的额外清理：去掉 a=1、pa、def_leader_def_rank 这类原始字段引用。"""
    import re
    text = _clean_llm_text(text)
    text = re.sub(r"[（(][^）)]*[A-Za-z_][A-Za-z0-9_]*\s*=[^）)]*[）)]", "", text)
    text = re.sub(r"[（(]\s*(?:pa|pf|diff|gb|[a-z_][a-z0-9_]{2,40})\s*[）)]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _factcheck(draft: str, facts: dict) -> str:
    """用第二次 LLM 调用核对草稿中的数字与断言，失败时退回原稿。"""
    if not draft:
        return draft
    payload = json.dumps(compact_facts(facts), ensure_ascii=False, default=str)
    user = f"""事实 JSON：
{payload}

待审稿件：
{draft}

请逐句修订：
1. 删除或改正任何与事实 JSON 不符的数字、排名、球队/车手归属、胜负关系；
   车手所属车队必须以 team_driver_map 为准；删除正文中的 JSON 字段名与反引号；
2. 删除事实 JSON 中没有的信息，包括比赛地点、赛道类型、是否为冲刺周末、赛事历史、“首次举办”、天气、伤病、转会等背景；
3. 删除“历史级”“堪称”等无法由数据支撑的评价；
4. 复核所有加减法（如分差 = 两队积分相减、胜场差 = 胜场数相减），错误就改正；
5. 不要新增任何事实，保持原有 Markdown 结构与大致篇幅；
6. 只输出修订后的稿件正文，不要输出解释或前后缀。"""
    text, _err = _llm_chat(_FACTCHECK_SYSTEM, user, max_tokens=1500, temperature=0.1)
    return _clean_llm_text(text or draft)


_LLM_SYSTEM = ("你是一名资深体育数据分析师，擅长把统计数据转写成有洞察的中文分析。"
               "只能依据给定事实，绝不编造球员、比分、纪录、车队归属或赛程；"
               "事实未覆盖的内容要明确说“数据未涵盖”。不要使用无法由数据支撑的评价性说法"
               "（如“历史级”“堪称完美”）。不要引入 JSON 之外的背景知识，包括赛事历史、赛道属性、"
               "球队/车手传统或“新设分站”之类的说法；只做数字之间的比较与基于数字的推断。"
               "不要判断某站是否为冲刺赛周末、赛道类型或比赛地点，除非 JSON 明确给出。"
               "引用车手所属车队时必须以 team_driver_map 为准；不要输出 JSON 字段名或反引号。"
               "JSON 中的 notes 字段解释了关键统计口径，引用前先阅读。")


def llm_analysis(sport: str, facts: dict):
    facts_json = json.dumps(compact_facts({sport: facts}), ensure_ascii=False, default=str)
    if len(facts_json) > 16000:
        facts_json = facts_json[:16000] + "…（截断）"
    user = f"""赛事：{sd.SPORT_NAMES.get(sport, sport)}
以下是程序抓取并计算得到的事实 JSON：
{facts_json}

请写一段 400-600 字的「深度解读」，使用 Markdown，覆盖：
1) 当前格局与关键差距；2) 数据背后的竞技含义（战术、效率、赛程或体能）；3) 接下来最值得关注的变量。
要求：引用具体数字，且每个数字都必须能在上述 JSON 中找到；不要罗列全部数据；不要输出 JSON；
不要补充 JSON 中不存在的事实（包括车手/球队归属、历史纪录、比赛地点、赛道类型、是否为冲刺周末、伤病、天气）；
引用车手所属车队时以 team_driver_map 为准；不要输出 JSON 字段名或反引号；不确定的内容直接省略。"""
    draft, err = _llm_chat(_LLM_SYSTEM, user, temperature=0.3)
    if not draft:
        return None, err
    return _factcheck(_clean_llm_text(draft), {sport: facts}), ""


def answer_question(question: str, snaps: dict, use_llm: bool = True,
                    focus: str | None = None):
    """自然语言问答。返回 (answer, llm_used, error)。

    focus：当前页面赛事 key（f1/nba/epl/golf），用于没有明确赛事关键词时的兜底。
    """
    facts = {}
    for key, snap in snaps.items():
        if "error" in snap:
            continue
        try:
            facts[key] = build_facts(key, snap)
        except Exception:
            continue
    if use_llm and llm_available():
        payload = json.dumps(compact_facts(facts), ensure_ascii=False, default=str)
        if len(payload) > 18000:
            payload = payload[:18000] + "…（截断）"
        focus_line = (f"用户当前在「{sd.SPORT_NAMES.get(focus, focus)}」页面，"
                      f"优先使用该赛事数据回答。\n" if focus else "")
        user = f"""以下是四项赛事的最新事实 JSON：
{payload}

{focus_line}用户问题：{question}

请用中文直接回答，120 字以内：先给结论，最多再补两点关键数字。
口径提醒：NBA 的 pf/pa 是场均得分/失分，diff 是赛季总净胜分，不要当成场均；
分差用积分相减，胜场差用胜场数相减，不要自行计算跨分区的胜场差；
事实中没有答案就说“当前数据未涵盖”，不要猜测。
不要输出 JSON 字段名、英文键名或 gb=0.0 这类原始值。"""
        text, err = _llm_chat(_LLM_SYSTEM, user, max_tokens=400, temperature=0.3)
        if text:
            return _qa_clean(text), True, ""
        return _fallback_answer(question, facts, focus), False, err
    return _fallback_answer(question, facts, focus), False, ""


def _fallback_answer(question: str, facts: dict, focus: str | None = None) -> str:
    q = question.lower()

    def has(*keys):
        return any(k.lower() in q for k in keys)

    sport = None
    if "f1" in facts and has("f1", "车手", "方程式", "维斯塔潘", "诺里斯", "安东内利",
                             "法拉利", "梅赛德斯", "迈凯伦", "红牛", "冲刺赛"):
        sport = "f1"
    elif "nba" in facts and has("nba", "篮球", "湖人", "凯尔特人", "勇士", "东部",
                                "西部", "球队", "场均"):
        sport = "nba"
    elif "epl" in facts and has("英超", "足球", "阿森纳", "曼联", "利物浦", "切尔西",
                                "争四", "保级", "降级"):
        sport = "epl"
    elif "golf" in facts and has("高尔夫", "pga", "lpga", "巡回赛"):
        sport = "golf"
    if sport is None and focus in facts:
        sport = focus

    if sport == "f1":
        f = facts["f1"]
        ldr, sec = f["leader"], f["second"] or {}
        head = (f"F1：{ldr['name']}（{ldr['team']}）以 {_num(ldr['points'])} 分领跑，"
                f"领先 {sec.get('name', '-')} {_num(f['gap'])} 分。")
        base = head + f"{f['driver_status']}。"
        if has("前五", "前 5", "积分榜", "排名"):
            rows = "、".join(f"{i + 1}. {d['name']} {_num(d['points'])} 分"
                            for i, d in enumerate(f["top_drivers"][:5]))
            return base + f" 车手前五：{rows}。"
        if has("剩余", "几站", "下一站", "赛程", "接下来"):
            nr = f["next_race"] or {}
            txt = head + f"剩余 {f['remaining']} 站"
            if nr:
                txt += (f"，下一站 {nr.get('name', '')}（{nr.get('date', '')}，"
                        f"{nr.get('locality', '')}）")
            return txt + "。"
        if has("上一站", "上一场", "夺冠", "领奖台"):
            lr = f["last_race"] or {}
            podium = "、".join(f"P{p.get('pos')} {p.get('driver')}"
                               for p in (lr.get("podium") or [])[:3])
            txt = (f"F1 上一站 {lr.get('name', '')}（{lr.get('date', '')}）："
                   f"冠军 {f['race_winner'].get('driver', '-')}")
            return txt + (f"；领奖台 {podium}。" if podium else "。")
        return base

    if sport == "nba":
        f = facts["nba"]
        ldr = f["league_leader"]
        if has("防守", "失分"):
            d = f["best_defense"]
            return (f"NBA 防守最好的是 {d.get('name')}，场均失 {_num(d.get('pa', 0))} 分，"
                    f"联盟第 1。")
        if has("进攻", "得分"):
            o = f["best_offense"]
            return (f"NBA 进攻最好的是 {o.get('name')}，场均 {_num(o.get('pf', 0))} 分，"
                    f"联盟第 1。")
        if has("净胜", "分差", "净效率"):
            d = f["best_diff"]
            return (f"NBA 场均净胜分最高的是 {d.get('name')}，"
                    f"{_signed(round(d.get('pf', 0) - d.get('pa', 0), 1))} 分/场。")
        if has("前八", "东部", "西部"):
            east = "、".join(f"{i + 1}. {t['name']} {t['wins']}-{t['losses']}"
                             for i, t in enumerate(f["east"][:8]))
            west = "、".join(f"{i + 1}. {t['name']} {t['wins']}-{t['losses']}"
                             for i, t in enumerate(f["west"][:8]))
            return f"东部前八：{east}。西部前八：{west}。"
        return (f"NBA：{ldr.get('name')} 以 {ldr.get('wins')}-{ldr.get('losses')} 领跑全联盟；"
                f"东部第一 {f['east_leader'].get('name')}，西部第一 {f['west_leader'].get('name')}。"
                f"进攻第一 {f['best_offense'].get('name')}（{_num(f['best_offense'].get('pf', 0))} 分/场），"
                f"防守第一 {f['best_defense'].get('name')}（失 {_num(f['best_defense'].get('pa', 0))} 分/场）。")

    if sport == "epl":
        f = facts["epl"]
        ldr, sec = f["leader"], f["second"] or {}
        if has("争四", "第四", "前四", "欧冠") and has("保级", "降级"):
            return (f"英超：{ldr['team']} 以 {ldr['pts']} 分领跑，领先 {sec.get('team')} "
                    f"{f['gap']} 分；争四方面第 4 {f['fourth'].get('team')} "
                    f"{f['fourth'].get('pts')} 分，只领先第 5 {f['fifth'].get('team')} "
                    f"{f['top4_gap']} 分；保级方面第 17 {f['safe'].get('team')} 领先第 18 "
                    f"{f['drop'].get('team')} {f['relegation_gap']} 分。")
        if has("降级", "保级"):
            return (f"英超保级：第 17 名 {f['safe'].get('team')} {f['safe'].get('pts')} 分，"
                    f"领先第 18 名 {f['drop'].get('team')} {f['relegation_gap']} 分。")
        if has("争四", "第四", "前四", "欧冠"):
            return (f"英超争四：第 4 {f['fourth'].get('team')} {f['fourth'].get('pts')} 分，"
                    f"只领先第 5 {f['fifth'].get('team')} {f['top4_gap']} 分。")
        return (f"英超：{ldr['team']} 以 {ldr['pts']} 分领跑，领先 {sec.get('team')} "
                f"{f['gap']} 分；第 4 {f['fourth'].get('team')} 领先第 5 "
                f"{f['fifth'].get('team')} {f['top4_gap']} 分；第 17 {f['safe'].get('team')} "
                f"领先降级区 {f['relegation_gap']} 分。")

    if sport == "golf":
        f = facts["golf"]
        tours = [t for t in f["tours"] if t["winner"]]
        if has("lpga"):
            tours = [t for t in tours if t["tour"].startswith("LPGA")]
        elif has("pga"):
            tours = [t for t in tours if t["tour"].startswith("PGA")]
        elif has("dp world", "欧巡"):
            tours = [t for t in tours if "DP World" in t["tour"]]
        if not tours:
            return "当前数据暂未覆盖该巡回赛。"
        return "高尔夫：" + "；".join(
            f"{t['tour']} 上一站 {t['last_event']} 冠军 {t['winner']}"
            f"（{_par(t['winner_score'])}，{t['margin']} 杆优势）"
            for t in tours) + "。"

    parts = []
    if "f1" in facts:
        f = facts["f1"]
        parts.append(f"F1 {f['leader']['name']} 以 {_num(f['leader']['points'])} 分领跑，"
                     f"领先 {_num(f['gap'])} 分，剩余 {f['remaining']} 站")
    if "nba" in facts:
        f = facts["nba"]
        ldr = f["league_leader"]
        d = f["best_diff"]
        parts.append(f"NBA {ldr.get('name')} {ldr.get('wins')}-{ldr.get('losses')} 领跑，"
                     f"{d.get('name')} 场均净胜 "
                     f"{_signed(round(d.get('pf', 0) - d.get('pa', 0), 1))} 分")
    if "epl" in facts:
        f = facts["epl"]
        parts.append(f"英超 {f['leader']['team']} {f['leader']['pts']} 分领跑，"
                     f"第 4 到第 5 只差 {f['top4_gap']} 分")
    if "golf" in facts:
        winners = [t for t in facts["golf"]["tours"] if t["winner"]]
        parts.append("高尔夫 " + "、".join(
            f"{t['tour']} {t['winner']}（{_par(t['winner_score'])}）" for t in winners))
    if parts:
        return "今日看点：" + "；".join(parts) + "。"
    return ("当前问题未匹配到具体赛事。可以问：F1 车手积分榜、NBA 东西部排名、"
            "英超争冠与保级形势、高尔夫三大巡回赛冠军等。")


# --------------------------------------------------------------------------- #
# 组装报告
# --------------------------------------------------------------------------- #

def _sport_markdown(sport: str, snap: dict, facts: dict, gen_at: str) -> str:
    title = sd.SPORT_NAMES.get(sport, sport)
    md = [f"# {title} 分析报告",
          "",
          f"> 生成时间：{gen_at} ｜ 数据状态：{_meta_desc(snap.get('meta', {}))}",
          ""]
    md.append(BODY_BUILDERS[sport](facts))
    return "\n".join(md)


def _footer(sport: str, snap: dict) -> str:
    meta = snap.get("meta", {})
    note = ""
    if meta.get("error"):
        note = f"\n\n> 数据源异常说明：{meta['error']}（已使用本地缓存兜底）"
    return (f"\n\n---\n\n*数据来源：{SOURCE_CN.get(sport, SOURCE_CN['all'])}；"
            f"数据状态：{_meta_desc(meta)}。*"
            f"{note}\n\n*本报告由「赛事分析报告助手」自动生成，统计口径以数据源为准；"
            f"AI 解读仅供辅助，不构成任何预测或投注建议。*")


def _overview_summary(facts_map: dict) -> str:
    lines = ["## 本周速览", ""]
    if "f1" in facts_map:
        f = facts_map["f1"]
        lines.append(f"- **F1**：{f['leader']['name']} 以 {_num(f['leader']['points'])} 分领跑，"
                     f"领先 {f['second'].get('name', '-') if f['second'] else '-'} {_num(f['gap'])} 分；"
                     f"剩余 {f['remaining']} 站。")
    if "nba" in facts_map:
        f = facts_map["nba"]
        ldr = f["league_leader"]
        lines.append(f"- **NBA**：{ldr.get('name')} 以 {ldr.get('wins')}-{ldr.get('losses')} 领跑全联盟；"
                     f"东部第一 {f['east_leader'].get('name')}，西部第一 {f['west_leader'].get('name')}。")
    if "epl" in facts_map:
        f = facts_map["epl"]
        lines.append(f"- **英超**：{f['leader']['team']} 以 {f['leader']['pts']} 分领跑，"
                     f"领先 {f['second'].get('team', '-')} {f['gap']} 分；"
                     f"第 4 与第 5 名仅差 {f['top4_gap']} 分。")
    if "golf" in facts_map:
        f = facts_map["golf"]
        winners = "；".join(f"{t['tour']}冠军 {t['winner']}（{_par(t['winner_score'])}）"
                           for t in f["tours"] if t["winner"])
        lines.append(f"- **高尔夫**：上一站 {winners}。")
    return "\n".join(lines)


def generate_report(sport: str = "all", snaps: dict | None = None,
                    use_llm: bool = False, refresh: bool = False) -> dict:
    """生成报告，返回 {markdown, llm_used, llm_error, generated_at, meta}。"""
    if snaps is None:
        snaps = sd.all_snapshots(force=refresh)
    gen_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    facts_map, errors = {}, []
    for key, snap in snaps.items():
        if "error" in snap:
            errors.append(f"{sd.SPORT_NAMES.get(key, key)}：{snap['error']}")
            continue
        try:
            facts_map[key] = build_facts(key, snap)
        except Exception as exc:
            errors.append(f"{sd.SPORT_NAMES.get(key, key)}：{type(exc).__name__}: {exc}")

    if not facts_map:
        return {"markdown": "# 报告生成失败\n\n所有数据源均不可用：\n\n" + "\n".join(f"- {e}" for e in errors),
                "llm_used": False, "llm_error": "", "generated_at": gen_at, "meta": {}}

    llm_used, llm_error = False, ""

    if sport == "all":
        if use_llm:
            llm_error = "综合周报仅做规则统计；单赛事报告支持 AI 深度解读"
        parts = [f"# 全球顶级赛事分析周报", "",
                 f"> 生成时间：{gen_at} ｜ 覆盖：F1、NBA、英超、高尔夫全球巡回赛", "",
                 _overview_summary(facts_map), ""]
        for key in ("f1", "nba", "epl", "golf"):
            if key not in facts_map:
                continue
            parts += ["---", "", f"# {sd.SPORT_NAMES[key]}", "",
                      BODY_BUILDERS[key](facts_map[key], h="##"), "",
                      _footer(key, snaps[key])]
        md = "\n".join(parts)
    else:
        if sport not in facts_map:
            return {"markdown": f"# 报告生成失败\n\n{sd.SPORT_NAMES.get(sport, sport)}：数据不可用\n\n"
                                + "\n".join(f"- {e}" for e in errors),
                    "llm_used": False, "llm_error": "", "generated_at": gen_at, "meta": {}}
        md = _sport_markdown(sport, snaps[sport], facts_map[sport], gen_at)
        if use_llm:
            text, err = llm_analysis(sport, facts_map[sport])
            if text:
                md += f"\n\n## AI 深度解读\n\n{text}"
                llm_used = True
            else:
                llm_error = err or "未知错误"
                md += f"\n\n> AI 解读未生成：{llm_error}"
        md += _footer(sport, snaps[sport])

    if errors:
        md += "\n\n> 部分赛事数据缺失：" + "；".join(errors)
    return {"markdown": md, "llm_used": llm_used, "llm_error": llm_error,
            "generated_at": gen_at, "meta": {k: v.get("meta", {}) for k, v in snaps.items()}}


def save_report(sport: str, markdown: str) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = REPORT_DIR / f"{sport}_report_{stamp}.md"
    path.write_text(markdown, "utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="赛事分析报告助手（F1 / NBA / 英超 / 高尔夫）")
    ap.add_argument("--sport", default="all", choices=["all", "f1", "nba", "epl", "golf"])
    ap.add_argument("--out", default="", help="输出 Markdown 路径（默认存到 sports_reports/）")
    ap.add_argument("--llm", action="store_true", help="追加 AI 深度解读")
    ap.add_argument("--refresh", action="store_true", help="强制刷新数据")
    ap.add_argument("--ask", default="", help="用自然语言提问，不生成报告")
    args = ap.parse_args()

    if args.ask:
        snaps = sd.all_snapshots(force=args.refresh)
        answer, used, err = answer_question(args.ask, snaps, use_llm=True)
        print(answer)
        if err and not used:
            print(f"\n（AI 问答不可用，使用本地规则回答：{err}）", file=sys.stderr)
        return

    result = generate_report(args.sport, use_llm=args.llm, refresh=args.refresh)
    md = result["markdown"]
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, "utf-8")
    else:
        path = save_report(args.sport, md)
    print(f"报告已生成：{path}")
    print(f"AI 解读：{'已生成' if result['llm_used'] else ('未启用' if not args.llm else '失败 - ' + result['llm_error'])}")
    print("\n" + md[:800] + ("\n…" if len(md) > 800 else ""))


if __name__ == "__main__":
    main()
