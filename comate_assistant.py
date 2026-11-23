#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comate 产品助手：解答 Comate（百度 AI 编程智能体）产品使用问题

基于 comate_rag.py 的 RAG 知识库（产品说明书：Agent 模式 / Context 系统 / Skills /
MCP / Zulu CLI / 模型计费等），回答时自动注入知识库片段并引用文档来源；
知识库可随时增量更新（--update / 工具 / 服务启动时自动同步）。

用法：
    .venv/bin/python comate_assistant.py "Comate 的 Agent 模式有哪几种？"
    .venv/bin/python comate_assistant.py --chat
    .venv/bin/python comate_assistant.py --serve --port 8901
    .venv/bin/python comate_assistant.py --update          # 重建知识库索引

HTTP：
    curl -X POST http://127.0.0.1:8901/ask -H "Content-Type: application/json" \
         -d '{"question":"如何安装 skill？"}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import comate_rag

HERE = Path(__file__).resolve().parent
# 可用 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 环境变量切换到任意 OpenAI 兼容厂商
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")

AGENT_SYSTEM = """你是「Comate 产品助手」，解答关于 Comate（百度 AI 编程智能体）产品使用的问题。
你可以调用工具从知识库检索产品说明书（Agent 模式 / Context 系统 / Skills / Subagents /
Rules / Commands / MCP / @ 关联上下文 / Zulu CLI / Jinx Code / 模型与计费等）。回答必须遵守：
1. 先检索知识库再回答，以知识库内容为准，不要凭记忆或常识作答；
2. 中文回答，先给结论，再给要点，控制在 200 字以内；
3. 引用知识库时注明来源文档名，如（来源：product-overview.md）；
4. 不编造文档中不存在的能力、参数、价格或版本信息；未覆盖就回答“当前说明书未涵盖”；
5. 问题与 Comate 无关（其他 AI 工具、IDE 通用设置、与产品无关的编码问题等）时，
   直接说明服务范围是 Comate 产品使用，不要调用工具。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在 Comate 产品说明书 RAG 知识库中按语义检索相关段落。",
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
            "description": "增量更新知识库索引（说明书文件变化后调用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "force": {"type": "boolean",
                              "description": "true 时全量重建，默认 false 只增量更新"},
                },
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

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


def _clean(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("md\n").strip()
    return text


def _chat(messages: list, tools: bool = True):
    from openai import OpenAI
    client = OpenAI(api_key=_load_key(), base_url=LLM_BASE, timeout=90.0)
    kwargs = dict(model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=900)
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


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #

def _json(data, limit: int = 8000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


def tool_search_knowledge(query: str, k: int = 4) -> str:
    hits = comate_rag.retrieve(query, k=max(1, min(int(k or 4), 10)))
    return _json({"query": query, "count": len(hits), "hits": hits})


def tool_update_knowledge(force: bool = False) -> str:
    res = comate_rag.update(force=bool(force), quiet=True)
    return _json({"updated": res["updated"], "n_files": res["files"],
                  "n_chunks": res["chunks"], "added": res["added"]}, limit=2000)


TOOL_IMPL = {
    "search_knowledge": tool_search_knowledge,
    "update_knowledge": tool_update_knowledge,
}


# --------------------------------------------------------------------------- #
# Agent 循环
# --------------------------------------------------------------------------- #

def run_agent(question: str, history: list | None = None, use_llm: bool = True,
              max_steps: int = 4, use_rag: bool = True):
    """执行一次问答，返回 (answer, tools_used, error, rag_hits)。"""
    question = (question or "").strip()
    if not question:
        return "请输入问题。", [], "", []
    if not use_llm or not llm_available():
        return "未配置 DASHSCOPE_API_KEY，无法调用大模型。", [], "no key", []

    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    if history:
        messages.append({"role": "system",
                         "content": "以下是最近的对话，供理解指代：\n" + "\n".join(history[-6:])})

    # RAG 注入：把说明书相关片段作为答题背景
    rag_hits = []
    user_content = question
    if use_rag:
        try:
            rag_hits = comate_rag.retrieve(question, k=3)
        except Exception:
            rag_hits = []
    if rag_hits:
        refs = "\n".join(
            f"- 《{h['title']}》相似度 {h['score']}：\n  {h['text'][:340]}"
            for h in rag_hits)
        user_content = (question
                        + "\n\n【知识库参考片段】以下段落来自 Comate 产品说明书检索，"
                          "回答时请引用来源文档名：\n" + refs)
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
                    return _clean(text), used, "", rag_hits
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

        messages.append({"role": "user",
                         "content": "请基于已获取的知识库内容直接给出最终回答，不要再调用工具。"})
        msg = _chat(messages, tools=False)
        text = (msg.content or "").strip()
        if text:
            return _clean(text), used, "", rag_hits
        return "助手没有生成有效回答，请换个问法。", used, "空响应", rag_hits
    except Exception as exc:
        return f"调用大模型失败：{exc}", used, f"{type(exc).__name__}: {exc}", rag_hits


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #

def create_app():
    from flask import Flask, jsonify, request

    app = Flask("comate_assistant")

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.get("/health")
    def health():
        rag = comate_rag.stats()
        return jsonify({"ok": True, "service": "comate-assistant",
                        "llm": "ready" if llm_available() else "missing-key",
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
            use_rag = bool(body.get("rag", True))
        else:
            question = request.args.get("q") or request.args.get("question") or ""
            use_rag = str(request.args.get("rag", "1")) != "0"
        question = str(question).strip()
        if not question:
            return jsonify({"ok": False, "error": "缺少 question 参数"}), 400
        started = time.time()
        answer, used, err, rag_hits = run_agent(question, use_rag=use_rag)
        return jsonify({
            "ok": bool(answer),
            "question": question,
            "answer": answer,
            "tools": used,
            "rag_sources": [h["title"] for h in rag_hits],
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
        print(f"\n（助手不可用：{err}）", file=sys.stderr)


def _chat_loop(trace: bool, use_llm: bool = True, use_rag: bool = True):
    print("Comate 产品助手已就绪，输入问题开始对话；输入 exit / quit 退出。\n")
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
    ap = argparse.ArgumentParser(description="Comate 产品助手（RAG 知识库答疑）")
    ap.add_argument("question", nargs="?", default="", help="要问的问题；留空则进入交互模式")
    ap.add_argument("--chat", action="store_true", help="交互式对话")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--trace", action="store_true", help="显示工具调用过程")
    ap.add_argument("--no-llm", action="store_true", help="不调用大模型")
    ap.add_argument("--no-rag", action="store_true", help="不注入知识库参考片段")
    ap.add_argument("--update", action="store_true", help="重建知识库索引后退出")
    ap.add_argument("--rebuild", action="store_true", help="全量重建知识库索引后退出")
    args = ap.parse_args()

    if args.update or args.rebuild:
        res = comate_rag.update(force=args.rebuild)
        print(f"知识库已更新：{res['updated']}，文档 {res['files']} 份，"
              f"切片 {res['chunks']} 个，新增 {res['added']} 个")
        return

    if args.serve:
        try:
            res = comate_rag.update(quiet=True)  # 启动时增量同步一次
            print(f"知识库已同步：{res['chunks']} 个切片")
        except Exception:
            pass
        app = create_app()
        print(f"Comate 产品助手 HTTP 服务：http://{args.host}:{args.port}")
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