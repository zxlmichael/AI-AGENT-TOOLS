#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛事分析 Agent（可随时调用）

能力：
  - 工具调用：读取四项赛事最新事实、生成并归档报告、检索历史报告、RAG 知识库检索/更新、查看数据状态
  - RAG 增强：每次回答自动注入知识库检索片段（--no-rag 关闭）
  - 命令行：单次提问 / 交互式对话
  - HTTP 接口：POST /ask、GET /ask?q=、GET /health

用法：
    .venv/bin/python sports_agent.py "F1 冠军还有悬念吗？"
    .venv/bin/python sports_agent.py --chat
    .venv/bin/python sports_agent.py --serve --port 8900
    .venv/bin/python sports_agent.py "帮我生成 NBA 报告" --trace

HTTP 调用示例：
    curl -s -X POST http://127.0.0.1:8900/ask \
         -H "Content-Type: application/json" \
         -d '{"question":"英超争四形势如何？"}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import sports_data as sd
import sports_rag
import sports_report as sr

HERE = Path(__file__).resolve().parent
# 历史报告归档目录：可用 SPORTS_REPORT_DIR 环境变量覆盖
DAILY_REPORT_DIR = Path(os.environ.get("SPORTS_REPORT_DIR", str(Path.home() / "SportsWeekly" / "app" / "sports_reports")))

AGENT_SYSTEM = """你是「赛事分析助手」Agent，服务于 F1、NBA、英超、高尔夫四项赛事。
你可以调用工具获取最新统计数据、生成报告、检索历史报告。回答必须遵守：
1. 先调用工具拿到数据再回答，不要凭记忆或常识作答；
2. 中文回答，先给结论，再给 1-2 条关键数字依据，控制在 150 字以内；
3. 所有数字必须来自工具结果；数据未覆盖就回答“当前数据未涵盖”；
4. 不编造赛程、伤病、转会、历史纪录、赛道属性；
5. NBA 的 pf/pa 是场均得分/失分，diff 是赛季总净胜分，不要当成场均；
6. 回答里不要出现 pa、pf、diff、gb、def_leader_def_rank 等英文字段名，全部用中文表述；
7. 问题与四项赛事无关时，直接说明服务范围，不要调用工具；
8. 问题下方【知识库参考片段】来自历史归档报告的语义检索，可能与最新数据有出入；
   若与工具返回的最新数据冲突，以工具数据为准，引用时注明报告日期。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_sport_facts",
            "description": "获取赛事最新事实数据：积分榜、排名、赛程、上一站冠军、数据口径等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sport": {"type": "string", "enum": ["all", "f1", "nba", "epl", "golf"],
                              "description": "赛事；拿不准时用 all"},
                },
                "required": ["sport"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "生成某赛事的完整 Markdown 分析报告并归档到 sports_reports/。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sport": {"type": "string", "enum": ["all", "f1", "nba", "epl", "golf"]},
                    "use_ai": {"type": "boolean",
                               "description": "是否追加 AI 深度解读（仅单赛事有效，约 30-40 秒）"},
                },
                "required": ["sport"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": "在历史归档报告中按关键词检索（如球队名、车手名、赛事名）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "limit": {"type": "integer", "description": "最多返回几份报告，默认 5"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在 RAG 知识库（历史归档报告的语义索引）中检索相关段落，"
                           "适合查询往期数据、历史对决、复盘类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题或关键词"},
                    "k": {"type": "integer", "description": "返回片段数，默认 4"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_knowledge",
            "description": "增量更新 RAG 知识库索引（新增或变化的报告重新向量化）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "force": {"type": "boolean",
                              "description": "true 时全量重建，默认 false 只增量更新"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_status",
            "description": "查看四项赛事数据的抓取状态、数据来源和更新时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local_answer",
            "description": "不调用大模型，直接用本地规则回答一个数据问题（适合简单排名/积分问题）。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def _snaps() -> dict:
    return sd.all_snapshots()


def _facts(snaps: dict, sport: str) -> dict:
    keys = list(snaps) if sport == "all" else [sport]
    out = {}
    for key in keys:
        snap = snaps.get(key)
        if not snap or "error" in snap:
            continue
        try:
            out[key] = sr.build_facts(key, snap)
        except Exception:
            continue
    return out


def _json(data, limit: int = 14000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


def tool_get_sport_facts(sport: str = "all") -> str:
    snaps = _snaps()
    facts = _facts(snaps, sport)
    if not facts:
        return _json({"error": f"{sport} 数据不可用"})
    meta = {k: snaps[k].get("meta", {}) for k in facts}
    return _json({"data_status": meta, "facts": sr.compact_facts(facts)})


def tool_generate_report(sport: str = "all", use_ai: bool = False) -> str:
    result = sr.generate_report(sport, use_llm=use_ai)
    path = sr.save_report(sport, result["markdown"])
    return _json({
        "saved_to": str(path),
        "sport": sport,
        "llm_used": result["llm_used"],
        "llm_error": result.get("llm_error", ""),
        "markdown_head": result["markdown"][:1200],
    })


def _report_dirs() -> list[Path]:
    return [d for d in (sr.REPORT_DIR, DAILY_REPORT_DIR) if d.exists()]


def tool_search_reports(keyword: str, limit: int = 5) -> str:
    keyword = (keyword or "").strip()
    if not keyword:
        return _json({"error": "keyword 不能为空"})
    hits = []
    files = sorted({p for d in _report_dirs() for p in d.glob("*.md")},
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            lines = path.read_text("utf-8").splitlines()
        except Exception:
            continue
        matched = [ln.strip() for ln in lines if keyword.lower() in ln.lower()]
        if matched:
            hits.append({"file": path.name, "matches": matched[:4]})
        if len(hits) >= max(1, int(limit)):
            break
    return _json({"keyword": keyword, "count": len(hits), "reports": hits}, limit=6000)


def tool_search_knowledge(query: str, k: int = 4) -> str:
    hits = sports_rag.retrieve(query, k=max(1, min(int(k or 4), 10)))
    return _json({"query": query, "count": len(hits), "hits": hits}, limit=9000)


def tool_update_knowledge(force: bool = False) -> str:
    res = sports_rag.update(force=bool(force), quiet=True)
    return _json({"updated": res["updated"], "n_files": res["files"],
                  "n_chunks": res["chunks"], "added": res["added"]}, limit=2000)


def tool_data_status() -> str:
    snaps = _snaps()
    rows = {}
    for key, snap in snaps.items():
        meta = snap.get("meta", {})
        rows[key] = {
            "name": sd.SPORT_NAMES.get(key, key),
            "source": meta.get("source", "?"),
            "fetched_at": meta.get("fetched_at", ""),
            "error": snap.get("error", ""),
        }
    return _json(rows, limit=3000)


def tool_local_answer(question: str) -> str:
    answer, _, err = sr.answer_question(question, _snaps(), use_llm=False)
    return _json({"answer": answer, "error": err}, limit=3000)


TOOL_IMPL = {
    "get_sport_facts": tool_get_sport_facts,
    "generate_report": tool_generate_report,
    "search_reports": tool_search_reports,
    "search_knowledge": tool_search_knowledge,
    "update_knowledge": tool_update_knowledge,
    "data_status": tool_data_status,
    "local_answer": tool_local_answer,
}


# --------------------------------------------------------------------------- #
# Agent 循环
# --------------------------------------------------------------------------- #

def _chat(messages: list, tools: bool = True):
    from openai import OpenAI
    client = OpenAI(api_key=sr._load_key(), base_url=sr.LLM_BASE, timeout=90.0)
    kwargs = dict(model=sr.LLM_MODEL, messages=messages,
                  temperature=0.2, max_tokens=800)
    if tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs).choices[0].message


def _assistant_msg(msg) -> dict:
    data = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        data["tool_calls"] = [{
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in msg.tool_calls]
    return data


def run_agent(question: str, history: list | None = None, use_llm: bool = True,
              max_steps: int = 5, use_rag: bool = True):
    """执行一次 Agent 问答，返回 (answer, tools_used, error, rag_hits)。"""
    question = (question or "").strip()
    if not question:
        return "请输入问题。", [], "", []
    if not use_llm or not sr.llm_available():
        answer, _, err = sr.answer_question(question, _snaps(), use_llm=False)
        return answer, [], err, []

    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    if history:
        messages.append({"role": "system",
                         "content": "以下是最近的对话，供理解指代：\n" + "\n".join(history[-6:])})

    # RAG：把知识库相关片段注入问题，作为答题背景
    rag_hits = []
    user_content = question
    if use_rag:
        try:
            rag_hits = sports_rag.retrieve(question, k=3)
        except Exception:
            rag_hits = []
    if rag_hits:
        refs = "\n".join(
            f"- 《{h['title']}》（{h['sport']}，{h['date'] or '日期未知'}）相似度 {h['score']}：\n  {h['text'][:300]}"
            for h in rag_hits)
        user_content = (question
                        + "\n\n【知识库参考片段】以下段落来自历史归档报告的语义检索，"
                          "可能与最新数据有出入；若与工具返回的最新数据冲突，以工具数据为准：\n"
                        + refs)
    messages.append({"role": "user", "content": user_content})

    used = []
    try:
        for _ in range(max_steps):
            msg = _chat(messages)
            calls = getattr(msg, "tool_calls", None) or []
            messages.append(_assistant_msg(msg))
            if not calls:
                text = (msg.content or "").strip()
                if text:
                    return sr._qa_clean(text), used, "", rag_hits
                break
            for tc in calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = TOOL_IMPL.get(name)
                if not fn:
                    result = _json({"error": f"未知工具 {name}"})
                else:
                    try:
                        result = fn(**args)
                        used.append({"tool": name, "args": args, "ok": True})
                    except Exception as exc:
                        result = _json({"error": f"{type(exc).__name__}: {exc}"})
                        used.append({"tool": name, "args": args, "ok": False})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # 工具步数用完，要求直接作答（不再提供工具）
        messages.append({"role": "user",
                         "content": "请基于已获取的工具结果直接给出最终回答，不要再调用工具。"})
        msg = _chat(messages, tools=False)
        text = (msg.content or "").strip()
        if text:
            return sr._qa_clean(text), used, "", rag_hits
        return "Agent 没有生成有效回答，请换个问法。", used, "空响应", rag_hits
    except Exception as exc:
        # 模型/网络异常时退回本地规则，保证随时可用
        answer, _, _ = sr.answer_question(question, _snaps(), use_llm=False)
        return answer, used, f"{type(exc).__name__}: {exc}", rag_hits


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, jsonify, request

    app = Flask("sports_agent")

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.get("/health")
    def health():
        rag = sports_rag.stats()
        return jsonify({"ok": True, "service": "sports-agent",
                        "llm": "ready" if sr.llm_available() else "fallback",
                        "rag_chunks": rag.get("n_chunks", 0),
                        "rag_updated": rag.get("updated", ""),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    @app.route("/ask", methods=["GET", "POST", "OPTIONS"])
    def ask():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            question = body.get("question") or body.get("q") or ""
        else:
            question = request.args.get("q") or request.args.get("question") or ""
        question = str(question).strip()
        if not question:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        started = time.time()
        use_rag = str(request.args.get("rag", "1")) != "0" if request.method == "GET" \
            else bool(body.get("rag", True))
        answer, used, err, rag_hits = run_agent(question, use_rag=use_rag)
        return jsonify({
            "ok": bool(answer),
            "question": question,
            "answer": answer,
            "tools": used,
            "rag_sources": [f"{h['title']} {h['date'] or ''}".strip() for h in rag_hits],
            "rag_hits": len(rag_hits),
            "error": err,
            "elapsed_seconds": round(time.time() - started, 2),
        })

    return app


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def _print_result(answer: str, used: list, err: str, trace: bool = False,
                  rag_hits: list | None = None):
    if trace and (used or rag_hits):
        print("[工具调用]")
        for item in used:
            flag = "ok" if item["ok"] else "fail"
            print(f"  - {item['tool']}({json.dumps(item['args'], ensure_ascii=False)}) {flag}")
        if rag_hits:
            print(f"  - [RAG] 命中 {len(rag_hits)} 条知识库片段")
        print()
    print(answer)
    if err and not answer:
        print(f"\n（Agent 不可用：{err}）", file=sys.stderr)


def _chat_loop(trace: bool, use_llm: bool = True, use_rag: bool = True):
    print("赛事分析 Agent 已就绪，输入问题开始对话；输入 exit / quit 退出。\n")
    history = []
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        started = time.time()
        answer, used, err, rag = run_agent(q, history=history, use_llm=use_llm,
                                           use_rag=use_rag)
        _print_result(answer, used, err, trace, rag)
        print(f"\n({time.time() - started:.1f}s)\n")
        history.append(f"用户：{q}")
        history.append(f"助手：{answer}")


def main():
    ap = argparse.ArgumentParser(description="赛事分析 Agent")
    ap.add_argument("question", nargs="?", default="", help="要问的问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--trace", action="store_true", help="显示工具调用过程")
    ap.add_argument("--no-llm", action="store_true", help="只用本地规则，不调用大模型")
    ap.add_argument("--no-rag", action="store_true", help="不注入知识库参考片段")
    args = ap.parse_args()

    if args.serve:
        # 启动时增量更新一次知识库索引（失败不影响服务）
        try:
            sports_rag.update(quiet=True)
        except Exception:
            pass
        app = create_app()
        print(f"赛事分析 Agent HTTP 服务：http://{args.host}:{args.port}")
        print(f"  POST /ask  {{\"question\": \"...\"}}   GET /ask?q=...   GET /health")
        app.run(host=args.host, port=args.port, threaded=True)
        return

    if args.chat or not args.question:
        _chat_loop(args.trace, use_llm=not args.no_llm, use_rag=not args.no_rag)
        return

    answer, used, err, rag = run_agent(args.question, use_llm=not args.no_llm,
                                       use_rag=not args.no_rag)
    _print_result(answer, used, err, args.trace, rag)


if __name__ == "__main__":
    main()
