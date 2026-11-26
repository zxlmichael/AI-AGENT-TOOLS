#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地多 Agent 协调中心(agents_hub)：一键调用、编排本机全部助手服务。

架构：Supervisor(协调者)模式 —— 本机 10 个助手服务(地图 8892 / 赛事 8900 /
Comate 8901 / 股票 8902 / 直播识别 8903 / 天气 8904 / NFL 8905 / 汽车出口 8906 /
文档库 8907 / 记忆机器人 8908)本身就是独立的「子 Agent」(有自己的 LLM 循环、工具与
知识库)。本脚本把每个子 Agent 封装成 Agently 工具注册给一个「主协调 Agent」
(本地 Ollama qwen3:8b 驱动)，由主 Agent 判断问题归属哪个领域、调用哪个或哪几个
子 Agent，再汇总成一份完整回答 —— 这就是多 Agent 协作中的
「Orchestrator-Workers / Router 模式」。
每个子 Agent = 数据层 + 助手层两个脚本，与 hub 放在同一目录，见 README 的对照表。

注：公开版刻意不含「电商首页抽取」类工具（涉及平台反爬对抗），扩展你自己的
子助手只需三步：定义工具函数 → 加入 ALL_TOOLS/_TOOL_MAP/_RULES → AGENT_ROWS 加一行。

用法：
    .venv/bin/python agents_hub.py "北京未来一周天气怎么样"
    .venv/bin/python agents_hub.py --chat                 # 交互模式
    .venv/bin/python agents_hub.py --list                 # 列出可用的子助手
    .venv/bin/python agents_hub.py --fast "北京天气"        # 规则路由（不依赖 LLM，秒回）
    .venv/bin/python agents_hub.py --serve --port 8910     # HTTP 服务（/ask /health / 页面）
    .venv/bin/python agents_hub.py --acp                   # 作为 ACP Agent 运行（stdio）

环境变量：
    LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 可切换到任意 OpenAI 兼容厂商
    (默认本地 Ollama：http://localhost:11434/v1，qwen3:8b，api_key=ollama)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from agently import Agently

# ---------------------------------------------------------------- 后端设置
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# 逐字段设置（整块 dict 会覆盖插件默认 path_mapping，不可用）
Agently.set_settings("plugins.ModelRequester.OpenAICompatible.base_url", LLM_BASE)
Agently.set_settings("plugins.ModelRequester.OpenAICompatible.model", LLM_MODEL)
Agently.set_settings("plugins.ModelRequester.OpenAICompatible.model_type", "chat")
Agently.set_settings("plugins.ModelRequester.OpenAICompatible.api_key", LLM_API_KEY)
# 8b 偶发 JSON 校验失败：最多重试 3 次，超限抛错走规则路由降级，避免单次问答拖数分钟
Agently.set_settings("plugins.ModelRequester.OpenAICompatible.request_retry",
                     {"max_attempts": 3, "after_output": True})

HOST = "127.0.0.1"
TIMEOUT = 30


# ---------------------------------------------------------------- HTTP 封装
def _http_post(port: int, path: str, payload: dict, timeout: int = TIMEOUT) -> str:
    """POST JSON 到子助手服务，返回字符串（Agently 工具必须返回字符串）。"""
    url = f"http://{HOST}:{port}{path}"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return f"[调用失败] {url} -> {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)
    if data.get("ok") is False:
        return f"[服务返回错误] {data.get('error') or data}"
    # 问答型接口：直接返回回答原文，不附加来源信息以免污染模型看到的文本
    if data.get("answer"):
        return str(data["answer"]).strip()
    # 结构化接口（非 answer 型），直接序列化
    return json.dumps(data, ensure_ascii=False)


def _http_get(port: int, path: str, params: dict) -> str:
    """GET 子助手服务，返回字符串。"""
    url = f"http://{HOST}:{port}{path}?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return f"[调用失败] {url} -> {type(exc).__name__}: {exc}"
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------- 子助手工具
# 每个工具 = 一台本地子 Agent 服务；函数名 + docstring 即模型看到的工具描述，
# 因此 docstring 里写明「什么领域的问题该用它」。


_WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@Agently.tool.action_func
def get_forecast(city: str = "北京", date: str | None = None, days: int = 7) -> str:
    """查询中国某城市某天的天气预报；不指定日期时返回未来几天逐日列表。

    Args:
        city: 城市名（北京/上海/广州等 51 个主要城市），默认北京
        date: 指定日期（如 "2026-11-20" 或 "2026年11月20日"），只有用户明确到某一天才填
        days: 预报天数 1-10，默认 7（仅 date 为空时生效）
    """
    try:
        data = json.loads(_http_get(8904, "/api/forecast", {"city": city, "days": days}))
    except Exception as exc:
        return f"[调用失败] {exc}"
    if not data.get("ok"):
        return f"[服务返回错误] {data.get('error')}"
    d = data["data"]
    rows = d["days"]

    def _fmt(label: str, day: dict) -> str:
        wd = _WEEK[datetime.strptime(day["date"], "%Y-%m-%d").weekday()]
        return (
            f"{label}{day['date']}（{wd}）：{day['desc']}，"
            f"最高 {day['tmax']}°C / 最低 {day['tmin']}°C，降水概率 {day['precip_prob']}%"
        )

    if date:
        # 指定日期的单日查询：由工具代码做日期匹配，模型只负责转述参数，避免多行比对出错
        date = str(date).strip().replace("年", "-").replace("月", "-").replace("日", "")
        if re.fullmatch(r"\d{1,2}-\d{1,2}", date):  # "11-20" 无年份，补当前年
            date = f"{datetime.now().year}-{date}"
        for day in rows:
            if day["date"] == date:
                return _fmt("", day)
        return f"[未找到] {city} 在 {date} 的预报（提供范围：{rows[0]['date']} ~ {rows[-1]['date']}）"

    lines = [f"{d.get('city', city)}未来 {len(rows)} 天预报（数据更新于 {d.get('updated', '')}）："]
    for day in rows:
        # 换成绝对星期，避免"今天/明天"这类相对指代让模型转述时弄混日期
        lines.append(_fmt("", day))
    return "\n".join(lines)


@Agently.tool.action_func
def geocode(address: str, city: str = "") -> str:
    """把中文地址/地点名解析为坐标和地图链接（如"天安门"、"深圳市南山区科技园"）。

    Args:
        address: 要解析的地点或地址文字
        city: 限定城市名（可选，用于消歧）
    """
    return _http_get(8892, "/api/geocode", {"q": address, "city": city})


@Agently.tool.action_func
def ask_comate(question: str) -> str:
    """解答「Comate 产品」使用问题（Agent 模式 / Context / Skills / MCP / 模型计费等）。

    只用于与 Comate（百度 AI 编程智能体）产品相关的问题。
    """
    return _http_post(8901, "/ask", {"question": question})


@Agently.tool.action_func
def ask_stock(question: str) -> str:
    """查询 A股/港股/美股科技股行情、涨跌、公司动态、指数走势等与股票相关的问题。"""
    return _http_post(8902, "/ask", {"question": question})


@Agently.tool.action_func
def ask_sports(question: str) -> str:
    """查询综合体育赛事信息（足球/篮球/网球等赛程、比分、新闻、周报汇总）"""
    return _http_post(8900, "/ask", {"question": question})


@Agently.tool.action_func
def ask_nfl(question: str) -> str:
    """查询 NFL 美式橄榄球专属信息（球队赛程、比分、排名、新闻、月度报告）。"""
    return _http_post(8905, "/ask", {"question": question})


@Agently.tool.action_func
def detect_live(text: str) -> str:
    """识别一段直播讲解文字中出现的衣物/商品（识别单价 ≥1000 元的高价值品类）。

    只用于直播带货文字的商品识别场景。
    """
    return _http_post(8903, "/detect", {"text": text})


@Agently.tool.action_func
def ask_auto_export(question: str) -> str:
    """查询全球汽车出口数据（各国家/车企出口量、趋势、月度数据、行业报告）。"""
    return _http_post(8906, "/api/qa", {"question": question})


@Agently.tool.action_func
def ask_doc_kb(question: str) -> str:
    """查询本地文档库统计信息、在本地知识库中检索文档内容（统计/检索/问答）。"""
    return _http_post(8907, "/api/qa", {"question": question, "mode": "rag"})


@Agently.tool.action_func
def ask_memory_bot(question: str, session_id: str = "hub") -> str:
    """与本地记忆机器人对话——它带持久记忆（会记住用户说过的事实并在后续对话中引用）。

    Args:
        question: 想问的问题
        session_id: 会话 ID，同一 ID 共享记忆，默认 "hub"
    """
    return _http_post(8908, "/api/qa", {"question": question, "session_id": session_id})


@Agently.tool.action_func
def ask_starlink(question: str) -> str:
    """查询 SpaceX 星链卫星实时信息（卫星总数/平均高度/倾角/轨道周期/分布图）。"""
    return _http_post(8914, "/api/qa", {"question": question})


@Agently.tool.action_func
def ask_news(question: str) -> str:
    """查询国内外时事新闻（总结概述/国内要闻/国际要闻/科技财经热点，带可视化看板）。"""
    return _http_post(8915, "/api/qa", {"question": question})


@Agently.tool.action_func
def ask_procurement(question: str) -> str:
    """查询政府招采网近 30 天地图需求招标/采购意向项目（地区/类型/预算/趋势，含问答与导出）。

    深挖类问题（趋势/分析）会自动升级 OpenManus，可能耗时 1-4 分钟。
    """
    return _http_post(8916, "/api/qa", {"question": question}, timeout=300)


ALL_TOOLS = [
    get_forecast, geocode, ask_comate, ask_stock, ask_sports,
    ask_nfl, detect_live, ask_auto_export, ask_doc_kb, ask_memory_bot,
    ask_starlink, ask_news, ask_procurement,
]

AGENT_SYSTEM = """你是「本地多助手协调中心」。你可以调用以下工具，每个工具对应一台本地 AI 助手
服务（子 Agent），各自拥有独立知识库与能力：
- get_forecast：城市天气预报
- geocode：地点坐标解析与地图
- ask_comate：Comate 产品使用问题
- ask_stock：股票行情与公司动态（A股/港股/美股）
- ask_sports：综合体育赛事（赛程/比分/新闻）
- ask_nfl：NFL 美式橄榄球专属信息
- detect_live：直播讲解文字中的高价值衣物识别
- ask_auto_export：全球汽车出口数据
- ask_doc_kb：本地文档库统计与知识检索
- ask_memory_bot：带持久记忆的本地机器人
- ask_starlink：SpaceX 星链卫星实时位置与轨道统计
- ask_news：国内外时事新闻总结与要点（国内/国际/科技/财经）
- ask_procurement：政府招采网地图需求监测（招标/采购意向项目、趋势问答）

协调规则：
1. 先判断用户问题属于哪个领域，只调用最贴合的工具，不要调查无关领域；
2. 问题跨多个领域时，依次调用对应的每个工具，把结果汇总为一份完整回答，分节标注；
3. 一切事实以工具返回内容为准，不得编造工具未提供的数据；工具报错时如实转述错误；
4. 转述时间信息时严格沿用工具返回的「日期 + 星期」，不要自行改写指代
   （例如工具标"今天=11-19，明天=11-20"，用户问"明天天气"就必须报 11-20 的数据）；
5. 与所有领域都无关的问题（闲聊等），直接说明你能协调哪些助手，不要调用工具；
6. 涉及「招标/采购/招采/投标」的政府项目（如"地图需求项目"）一律用 ask_procurement；
   只有问地点坐标/路线（如"天安门在哪"）才用 geocode，即使问题里含"地图"二字；
7. 中文回答。"""


# ---------------------------------------------------------------- 规则路由（降级路径）
# 不依赖 LLM：按关键词把问题路由给子助手，LLM 不可用或想秒回时用（--fast）。
_CITIES = ["北京", "上海", "广州", "深圳", "天津", "重庆", "成都", "杭州", "武汉", "西安",
           "南京", "苏州", "郑州", "长沙", "青岛", "大连", "厦门", "昆明", "哈尔滨", "乌鲁木齐"]

_RULES = [
    ("get_forecast", ["天气", "气温", "下雨", "降水", "预报", "几度"]),
    ("geocode", ["在哪", "怎么走", "地址", "坐标", "路线", "地图"]),
    ("ask_stock", ["股票", "股价", "涨", "跌", "港股", "美股", "a股", "大盘", "指数"]),
    ("ask_comate", ["comate", "说明书"]),
    ("ask_sports", ["比赛", "赛程", "赛事", "比分", "体育"]),
    ("ask_nfl", ["nfl", "橄榄球"]),
    ("detect_live", ["直播", "衣物", "衣服", "带货"]),
    ("ask_auto_export", ["汽车出口", "出口量", "车企"]),
    ("ask_doc_kb", ["文档库", "知识库", "本地文档"]),
    ("ask_starlink", ["星链", "卫星", "spacex", "starlink", "轨道"]),
    ("ask_news", ["新闻", "时事", "要闻", "头条", "大事"]),
    ("ask_procurement", ["招采", "招标", "采购", "投标", "中标"]),
]
_TOOL_MAP = {
    "get_forecast": get_forecast,
    "geocode": geocode,
    "ask_comate": ask_comate,
    "ask_stock": ask_stock,
    "ask_sports": ask_sports,
    "ask_nfl": ask_nfl,
    "detect_live": detect_live,
    "ask_auto_export": ask_auto_export,
    "ask_doc_kb": ask_doc_kb,
    "ask_memory_bot": ask_memory_bot,
    "ask_starlink": ask_starlink,
    "ask_news": ask_news,
    "ask_procurement": ask_procurement,
}


def route_by_rules(question: str) -> str:
    """关键词规则路由：命中哪个领域就直调哪个子助手（无 LLM 参与，秒回）。"""
    hits = [name for name, kws in _RULES if any(kw in question for kw in kws)]
    # 招采类问题里出现"地图"仍属招采领域，不要让 geocode 抢答
    if "ask_procurement" in hits and "geocode" in hits and \
            any(k in question for k in dict(_RULES)["ask_procurement"]):
        hits.remove("geocode")
    if not hits:
        return ("规则路由未匹配到任何助手领域。可用领域：天气 / 地图 / 股票 / Comate / "
                "体育赛事 / NFL / 直播衣物识别 / 汽车出口 / 文档库 / 记忆机器人 / "
                "星链 / 时事新闻 / 招采地图监测。")
    parts = []
    for name in hits:
        fn = _TOOL_MAP[name]
        if name == "get_forecast":
            city = next((c for c in _CITIES if c in question), "北京")
            parts.append(f"[天气] {city}：\n{fn(city)}")
        elif name == "geocode":
            addr = question.replace("在哪", "").replace("怎么走", "").replace("地址", "").strip() or question
            parts.append(f"[地图] {addr}：\n{fn(addr)}")
        else:
            parts.append(f"[{name}] {fn(question)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- HTTP 服务
# 异步问答任务(网页端:提交 → 轮询 → 取结果;ask_hub 约 1~3 分钟)
_ASKS = {}
_ASK_LOCK = threading.Lock()


def _ask_worker(task_id: str, question: str) -> None:
    ans = ask_hub(question)
    with _ASK_LOCK:
        if task_id in _ASKS:
            _ASKS[task_id].update(status="done", answer=ans,
                                  elapsed=round(time.time() - _ASKS[task_id]["started"], 2))


def start_ask(question: str) -> str:
    task_id = uuid.uuid4().hex[:10]
    with _ASK_LOCK:
        _ASKS[task_id] = {"status": "running", "started": time.time()}
    threading.Thread(target=_ask_worker, args=(task_id, question), daemon=True).start()
    return task_id


def get_ask(task_id: str) -> dict | None:
    with _ASK_LOCK:
        t = _ASKS.get(task_id)
        return dict(t) if t else None


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地多 Agent 协调中心</title>
<style>
:root{--blue:#2563eb;--bg:#f6f7f9;--card:#fff;--tx:#222;--mut:#888}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,'PingFang SC',sans-serif;background:var(--bg);color:var(--tx)}
.wrap{max-width:1020px;margin:0 auto;padding:24px 16px 60px}
header{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border-radius:14px;padding:22px 24px;margin-bottom:18px}
header h1{margin:0 0 6px;font-size:22px}header p{margin:0;opacity:.9;font-size:13px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}
button{padding:10px 26px;font-size:15px;border:none;border-radius:8px;background:var(--blue);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
.badge{font-size:12px;padding:3px 10px;border-radius:999px;color:#fff;background:#2563eb}
.badge.g{background:#16a34a}.badge.r{background:#dc2626}.badge.o{background:#d97706}
.card{background:var(--card);border-radius:12px;padding:18px 20px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card h2{margin:0 0 12px;font-size:16px;border-left:4px solid var(--blue);padding-left:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.agent{border:1px solid #eef0f3;border-radius:10px;padding:12px 14px;font-size:13px;cursor:pointer;transition:border-color .15s}
.agent:hover{border-color:#93b4f5;background:#f7faff}
.agent .t{font-weight:600;font-size:14px;display:flex;justify-content:space-between;align-items:center}
.agent .p{color:var(--mut);font-size:11px;margin:3px 0}
.agent .d{color:#555;line-height:1.5}
.dot{display:inline-block;width:8px;height:8px;border-radius:999px;margin-right:4px}
.dot.on{background:#16a34a}.dot.off{background:#cbd5e1}
textarea{width:100%;height:72px;font-size:15px;padding:10px 12px;box-sizing:border-box;border:1px solid #d8dce2;border-radius:8px;resize:vertical}
#prog{display:none;background:var(--card);border:1px solid #eee;border-radius:10px;padding:14px 16px;margin:12px 0}
#prog .bar2{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-top:8px}
#prog .fill{height:100%;width:0;background:var(--blue);border-radius:4px;transition:width .5s}
#out{margin-top:14px;white-space:pre-wrap;line-height:1.7;background:#f7f7f8;padding:16px;border-radius:10px;min-height:40px;font-size:14px;display:none}
.hint{color:#888;font-size:12px;margin-top:6px}
</style>
</head>
<body><div class="wrap">
<header><h1>本地多 Agent 协调中心</h1>
<p>一个入口协调 10 台本地助手：天气 / 地图 / 股票 / Comate / 体育 / NFL / 直播识别 / 汽车出口 / 文档库 / 记忆机器人 · 主 Agent 用本地 Ollama 路由</p></header>
<div class="bar">
  <span id="stats" class="badge">助手状态加载中…</span>
  <span class="hint">点击卡片可直接追问对应领域；跨领域问题交给「一键协调」自动分派汇总。</span>
</div>
<div class="card"><h2>子助手</h2><div id="agents" class="grid"></div></div>
<div class="card"><h2>提问</h2>
<textarea id="q" placeholder="例如：北京明天天气怎么样？顺便看看腾讯股价"></textarea><br>
<button id="go" onclick="ask()">一键协调</button>
<span class="hint">主 Agent 两次本地推理约 1~3 分钟；点击某张助手卡片可只问该领域。</span>
</div>
<div id="prog"><div id="ptxt">已提交…</div><div class="bar2"><div class="fill" id="fill"></div></div></div>
<div id="out"></div>
<script>
let polling=null, agents=[];
const EXAMPLES={get_forecast:'北京明天天气怎么样？',geocode:'天安门在哪？',
 ask_stock:'腾讯股价今天怎么样？',ask_sports:'最近有什么足球比赛？',
 ask_comate:'Comate 的 Agent 模式怎么用？',ask_nfl:'NFL 本周赛程？',
 detect_live:'这件衣服是直播里说的那件吗',ask_auto_export:'中国汽车出口量趋势？',
 ask_doc_kb:'文档库里有多少份文档？',ask_memory_bot:'你还记得我上次说过什么吗？',
 ask_starlink:'现在有多少颗星链卫星？',
  ask_news:'总结一下最近的国内外大事',
  ask_procurement:'最近政府招采有哪些地图需求项目？'};
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}
function loadAgents(){
  fetch('/agents').then(r=>r.json()).then(d=>{
    agents=d.agents||[];
    const on=agents.filter(a=>a.online).length;
    document.getElementById('stats').textContent=
      '子助手在线 '+on+'/'+agents.length+' · 模型 '+d.llm;
    document.getElementById('stats').className='badge '+(on?'g':'o');
    const h=document.getElementById('agents'); h.innerHTML='';
    agents.forEach(a=>{
      const c=document.createElement('div'); c.className='agent';
      c.innerHTML='<div class="t">'+esc(a.name)+'<span><span class="dot '+(a.online?'on':'off')+'"></span>'+(a.online?'在线':'离线')+'</span></div>'+
        '<div class="p">:'+a.port+' · '+esc(a.tool)+'</div><div class="d">'+esc(a.desc)+'</div>';
      c.onclick=()=>{document.getElementById('q').value=EXAMPLES[a.tool]||'';document.getElementById('q').focus()};
      h.appendChild(c);
    });
  }).catch(()=>{document.getElementById('stats').textContent='助手状态获取失败';});
}
function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q){alert('先输入问题');return}
  document.getElementById('go').disabled=true;
  document.getElementById('prog').style.display='block';
  const p=document.getElementById('ptxt'), f=document.getElementById('fill');
  p.textContent='已提交「'+q.slice(0,30)+'…」协调中…'; f.style.width='5%';
  fetch('/ask/async',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})
  .then(r=>r.json()).then(d=>{
    if(!d.task_id){p.textContent='提交失败';go.disabled=false;return}
    polling=setInterval(()=>{
      fetch('/task/'+d.task_id).then(r=>r.json()).then(t=>{
        f.style.width=Math.min(95,(t.pct||5)+'%');
        p.textContent='协调中(主 Agent 推理 + 子助手调用)…';
        if(t.status==='done'){
          clearInterval(polling);f.style.width='100%';p.textContent='完成(用时 '+t.elapsed+'s)';
          const o=document.getElementById('out');o.style.display='block';o.textContent=t.answer;
          document.getElementById('go').disabled=false;document.getElementById('go').textContent='再问一次';
        }
      }).catch(()=>{});
    },3000);
  }).catch(e=>{p.textContent='失败:'+e;document.getElementById('go').disabled=false});
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}});
loadAgents();
</script>
</body>
</html>"""


def serve(port: int = 8910) -> None:
    """HTTP 服务：POST /ask {"question"}(同步) · POST /ask/async · GET /task/<id> ·
    GET /agents(子助手在线探测) · GET /health、/ 页面。"""
    from flask import Flask, jsonify, request

    app = Flask("agents-hub")

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "service": "agents-hub", "llm": LLM_MODEL,
                        "sub_agents": len(ALL_TOOLS),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.get("/agents")
    def agents():
        return jsonify({"ok": True, "llm": LLM_MODEL, "agents": probe_agents()})

    @app.route("/ask", methods=["GET", "POST", "OPTIONS"])
    def ask():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            q = str(body.get("question") or body.get("q") or "").strip()
        else:
            q = str(request.args.get("q") or request.args.get("question") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        started = time.time()
        ans = ask_hub(q)
        return jsonify({"ok": True, "question": q, "answer": ans,
                        "elapsed_seconds": round(time.time() - started, 2)})

    @app.post("/ask/async")
    def ask_async():
        body = request.get_json(silent=True) or {}
        q = str(body.get("question") or "").strip()
        if not q:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        return jsonify({"task_id": start_ask(q)})

    @app.get("/task/<task_id>")
    def task(task_id: str):
        t = get_ask(task_id)
        if not t:
            return jsonify({"status": "notfound"}), 404
        out = {"status": t["status"], "started": t.get("started"),
               "elapsed": round(time.time() - t["started"], 1) if t["status"] == "running"
               else t.get("elapsed"),
               "pct": 90 if t["status"] == "running" else 100}
        if t.get("answer") is not None:
            out["answer"] = t["answer"]
        return jsonify(out)

    @app.get("/")
    def index():
        return PAGE

    print(f"多 Agent 协调中心已启动：http://127.0.0.1:{port}  (子助手 x{len(ALL_TOOLS)})")
    app.run(host="127.0.0.1", port=port)


# ---------------------------------------------------------------- ACP Agent
async def _acp_serve() -> None:
    """作为 ACP（Agent Client Protocol）Agent 运行（stdio），供 Zed 等 ACP 客户端拉起。"""
    import acp

    class HubAcpAgent:
        def __init__(self) -> None:
            self._conn = None
            self._sessions: dict[str, dict] = {}

        def on_connect(self, conn) -> None:
            self._conn = conn

        async def initialize(self, protocol_version: int, client_capabilities=None,
                             client_info=None, **kwargs):
            from acp.schema import InitializeResponse, Implementation
            return InitializeResponse(
                protocol_version=protocol_version,
                agent_info=Implementation(name="agents-hub-acp", title="本地多 Agent 协调中心",
                                          version="0.2.0"),
            )

        async def new_session(self, cwd: str, additional_directories=None,
                              mcp_servers=None, **kwargs):
            from acp.schema import NewSessionResponse
            sid = uuid.uuid4().hex
            self._sessions[sid] = {"cwd": cwd, "history": []}
            return NewSessionResponse(session_id=sid)

        async def prompt(self, session_id: str, prompt: list, **kwargs):
            from acp.schema import AgentMessageChunk, PromptResponse, TextContentBlock
            sess = self._sessions.setdefault(session_id, {"cwd": None, "history": []})
            text = "\n".join(
                getattr(b, "text", "") or "" for b in (prompt or [])
                if getattr(b, "type", None) in (None, "text")
            ).strip() or "（空消息）"
            # ask_hub 是阻塞式（Agently 同步执行），放线程池避免卡住事件循环
            try:
                reply = await asyncio.to_thread(ask_hub, text)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
            sess["history"].append(text)
            message_id = uuid.uuid4().hex
            if self._conn is not None:
                for i in range(0, len(reply), 100):
                    await self._conn.session_update(
                        session_id=session_id,
                        update=AgentMessageChunk(
                            content=TextContentBlock(type="text", text=reply[i:i + 100]),
                            message_id=message_id,
                        ),
                    )
                    await asyncio.sleep(0.01)
            return PromptResponse(stop_reason="end_turn")

        async def list_sessions(self, cwd: str | None = None, cursor: str | None = None, **kwargs):
            from acp.schema import ListSessionsResponse, SessionInfo
            return ListSessionsResponse(sessions=[
                SessionInfo(session_id=sid, cwd=d.get("cwd") or str(HERE))
                for sid, d in self._sessions.items()
            ])

        async def close_session(self, session_id: str, **kwargs):
            from acp.schema import CloseSessionResponse
            self._sessions.pop(session_id, None)
            return CloseSessionResponse()

        async def cancel(self, session_id: str, **kwargs) -> None:
            pass

    await acp.run_agent(HubAcpAgent())


# ---------------------------------------------------------------- 主入口
_WEEKDAY_CN = {"周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6, "周天": 6}
_SINGLE = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def _week_no(word: str) -> int:
    """从"周五/上周日/下周二"等词里取星期序号（周一=0）。"""
    if word[:2] in _WEEKDAY_CN:
        return _WEEKDAY_CN[word[:2]]
    return _SINGLE[word[-1:]]


def _normalize_time(text: str) -> str:
    """把问题里的相对时间词归一化为绝对日期（host 侧确定性消解）。

    小模型对"今天/明天/周X"这类相对指代转述不可靠，这里先在 host 侧换成
    绝对日期，让模型只需按日期字面匹配工具返回的数据行。
    周基准：周一为一周起点；"本周X"取本自然周该天；"下周X"取下一自然周该天；
    "上周X/上上周X"取前 1/2 个自然周；裸"周X"取未来最近（含今天）的该天。
    """
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())

    def _abs(m: re.Match) -> str:
        word = m.group(0)
        offset = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}.get(word)
        if offset is not None:
            return (today + timedelta(days=offset)).strftime("%Y年%m月%d日")
        if word.startswith("上上"):
            return (monday + timedelta(days=_week_no(word) - 14)).strftime("%Y年%m月%d日")
        if word.startswith("上"):
            return (monday + timedelta(days=_week_no(word) - 7)).strftime("%Y年%m月%d日")
        if word.startswith("下下"):
            return (monday + timedelta(days=14 + _week_no(word))).strftime("%Y年%m月%d日")
        if word.startswith("本"):
            return (monday + timedelta(days=_week_no(word))).strftime("%Y年%m月%d日")
        if word.startswith("下"):
            return (monday + timedelta(days=7 + _week_no(word))).strftime("%Y年%m月%d日")
        target = monday + timedelta(days=_week_no(word))
        if target < today:
            target += timedelta(days=7)
        return target.strftime("%Y年%m月%d日")

    # lookbehind 排除"一周/几周"等量词用法（"一周天气"里的"周天"不应命中）
    pattern = (
        r"大后天|后天|明天|今天|(?:上上|下下|上|本|下)周[一二三四五六日天]"
        r"|(?<![一二三四五六七八九十两几每这那上])周[一二三四五六日天]"
    )
    return re.sub(pattern, _abs, text)


def ask_hub(question: str) -> str:
    """一键协调：主 Agent 路由到合适的子助手并汇总回答；LLM 失败时降级到规则路由。"""
    agent = Agently.create_agent("agents-hub")
    agent.use_tools(ALL_TOOLS)
    try:
        return (
            agent.create_execution()
            .use_tools(ALL_TOOLS)
            .system(AGENT_SYSTEM)
            .input(_normalize_time(question))
            .get_text()
        )
    except Exception as exc:
        # 降级路径：不依赖 LLM 的关键词路由，保证服务可用
        return f"[主 Agent 调用失败：{type(exc).__name__}: {exc}]\n已降级到规则路由：\n{route_by_rules(question)}"


# ---------------------------------------------------------------- 子助手清单
# (端口, 名称, 工具名, 一句话简介) —— --list / 网页卡片 / 在线探测共用
AGENT_ROWS = [
    (8892, "地图助手", "geocode", "地点坐标解析与地图链接"),
    (8900, "赛事周报", "ask_sports", "综合体育赛事赛程/比分/新闻"),
    (8901, "Comate 产品助手", "ask_comate", "Comate 用法/Agent/Skills/MCP"),
    (8902, "股票助手", "ask_stock", "A股/港股/美股行情与公司动态"),
    (8903, "直播衣物识别", "detect_live", "直播讲解中的高价值衣物识别"),
    (8904, "天气助手", "get_forecast", "51 城市未来 10 天预报"),
    (8905, "NFL 助手", "ask_nfl", "NFL 赛程/比分/排名/新闻"),
    (8906, "汽车出口监测", "ask_auto_export", "全球汽车出口数据与趋势"),
    (8907, "文档知识库", "ask_doc_kb", "本地文档统计与知识检索"),
    (8908, "记忆机器人", "ask_memory_bot", "带持久记忆的对话机器人"),
    (8914, "星链卫星", "ask_starlink", "SpaceX 星链实时位置与轨道统计"),
    (8915, "时事新闻", "ask_news", "国内外新闻总结与要点(可视化)"),
    (8916, "招采地图监测", "ask_procurement", "政府招采地图需求项目与问答(可视化)"),
]


def _list() -> None:
    print("可用子助手（端口 | 服务 | 工具名）：")
    rows = [(port, name, tool) for port, name, tool, _ in AGENT_ROWS]
    for port, name, tool in rows:
        print(f"  {port:5d}  {name:<10} -> {tool}")


def probe_agents() -> list:
    """并发探测各子助手 /health,返回 [{port,name,tool,desc,online}]."""
    def _probe(port: int) -> bool:
        try:
            req = urllib.request.Request(f"http://{HOST}:{port}/health",
                                         headers={"User-Agent": "agents-hub"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    results = []
    with ThreadPoolExecutor(max_workers=len(AGENT_ROWS)) as ex:
        states = list(ex.map(lambda r: (_probe(r[0]), r), AGENT_ROWS))
    for online, (port, name, tool, desc) in states:
        results.append({"port": port, "name": name, "tool": tool,
                        "desc": desc, "online": online})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="本地多 Agent 协调中心（一键调用所有子助手）")
    parser.add_argument("question", nargs="?", help="问题；一次一问，跨领域问题自动分派多个子助手")
    parser.add_argument("--chat", action="store_true", help="交互模式")
    parser.add_argument("--list", action="store_true", help="列出可用子助手")
    parser.add_argument("--fast", action="store_true",
                        help="规则路由模式（不依赖 LLM，秒回；LLM 不可用时自动降级到它）")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--port", type=int, default=8910, help="HTTP 服务端口，默认 8910")
    parser.add_argument("--acp", action="store_true",
                        help="作为 ACP Agent 运行（stdio），供 Zed 等 ACP 客户端拉起")
    args = parser.parse_args()

    if args.serve:
        serve(args.port)
    elif args.acp:
        asyncio.run(_acp_serve())
    elif args.list:
        _list()
    elif args.chat:
        print("输入问题（quit 退出）：")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print(route_by_rules(q) if args.fast else ask_hub(q), "\n")
    elif args.question:
        print(route_by_rules(args.question) if args.fast else ask_hub(args.question))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()