#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地智能机器人（Local Memory Bot）
==================================
基于 Agently 4.x + Ollama 本地大模型，自带 **Agent 记忆能力**：

- 会话记忆：同一 session_id 下多轮对话上下文连贯；
- 长期记忆：每轮对话后自动提取「值得记住的事实」（用户偏好/项目背景/关键信息），
  存入本地 RecordStore；下次对话时自动召回注入上下文（AgentlyMemory 插件）。

用法：
    # CLI 单次问答
    .venv/bin/python memory_bot.py "你叫什么名字"
    # CLI 交互模式（同一会话连续对话）
    .venv/bin/python memory_bot.py --chat
    # Web 服务 (端口 8908)
    .venv/bin/python memory_bot.py --serve

模型配置（默认全部本地，不需要任何云账户）：
    LLM_BASE_URL  默认 http://localhost:11434/v1   (Ollama OpenAI 兼容接口)
    LLM_MODEL     默认 qwen3:8b
    LLM_API_KEY   默认 ollama                      (可切换任意 OpenAI 兼容厂商)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

from agently import Agently

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 模型配置
# 默认用 Ollama 本地模型；三个环境变量可切换到任意 OpenAI 兼容厂商
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MEMORY_ROOT = Path(os.environ.get("BOT_MEMORY_DIR", HERE / "bot_memory_data")).expanduser()

PORT = 8908
HOST = "127.0.0.1"

_AGENT = None
_AGENT_LOCK = threading.Lock()
_SESSIONS: dict[str, object] = {}

# ---------------------------------------------------------------- 内置工具
@Agently.tool.action_func
def list_files(directory: str | None = None) -> str:
    """列出本地目录中的文件和子目录清单（含名称、类型、大小）。

    Args:
        directory: 目标目录路径；省略时列出当前项目目录（AI大模型）。
    """
    import contextlib

    base = Path(directory).expanduser() if directory else HERE
    if not base.is_dir():
        return f"目录不存在: {base}"
    try:
        entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return "无权限访问该目录"
    lines, total = [], 0
    for p in entries:
        with contextlib.suppress(OSError):
            if p.is_dir():
                lines.append(f"[目录] {p.name}")
                total += 1
            else:
                size = p.stat().st_size
                kb = size / 1024
                size_s = f"{kb:.1f}KB" if kb < 1024 else f"{kb/1024:.2f}MB"
                lines.append(f"[文件] {p.name} ({size_s})")
                total += 1
    return f"路径: {base}\n共 {total} 项:\n" + "\n".join(lines)

# ---------------------------------------------------------------- Agent 构建
def get_agent():
    """构建（或复用）配置好的 Agently Agent：Ollama 本地模型 + 记忆插件。"""
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    from agently import Agently

    # 逐字段配置（避免整块 dict 覆盖掉插件默认的 path_mapping 等）
    Agently.set_settings(
        "plugins.ModelRequester.OpenAICompatible.base_url", LLM_BASE_URL
    )
    Agently.set_settings(
        "plugins.ModelRequester.OpenAICompatible.model", LLM_MODEL
    )
    Agently.set_settings(
        "plugins.ModelRequester.OpenAICompatible.model_type", "chat"
    )
    Agently.set_settings(
        "plugins.ModelRequester.OpenAICompatible.api_key", LLM_API_KEY
    )
    # 记忆持久化位置（sqlite + 本地文件）
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    Agently.set_settings("record_store.default_root", str(MEMORY_ROOT))
    # 开启记忆向量索引（召回时语义检索，需嵌入模型；关闭则按摘要/标签检索）
    Agently.set_settings("session.memory.AgentlyMemory.vector_index.enabled", False)

    agent = Agently.create_agent("local-memory-bot")
    agent.use_tools([list_files])  # 挂载本地文件工具
    _AGENT = agent
    return agent


def _activate(session_id: str):
    """激活指定会话（复用已有会话上下文），并挂载记忆插件。"""
    agent = get_agent()
    agent.activate_session(session_id=session_id)
    if session_id not in _SESSIONS:
        agent.activated_session.use_memory()
        _SESSIONS[session_id] = agent.activated_session
    return agent


def ask(question: str, session_id: str | None = None) -> dict:
    """向机器人提问，自动携带记忆与历史，并可调用本地工具。

    返回: {"answer": str, "session_id": str, "memories": int, "error": str|None}
    """
    if not str(question or "").strip():
        return {"answer": "请输入问题。", "session_id": "", "memories": 0, "error": "empty"}
    sid = str(session_id or "").strip() or f"bot-{int(time.time())}"
    with _AGENT_LOCK:  # Agently agent 会话为单激活态，串行化保证安全
        try:
            agent = _activate(sid)
            ex = (
                agent.create_execution()
                .use_tools([list_files])
                .system(
                    "你是一个本地智能机器人，中文回答。回答简洁、友好、直接。"
                    "当用户要求查看本地文件/目录时，调用 list_files 工具获取真实清单，"
                    "不要编造。记住对话中用户提到的重要事实（助手会自动完成记忆）。"
                )
                .input(str(question))
            )
            answer = ex.get_text()
            return {"answer": answer, "session_id": sid, "memories": None, "error": None}
        except Exception as e:  # 展示原始错误便于排查
            return {
                "answer": "",
                "session_id": sid,
                "memories": None,
                "error": f"{type(e).__name__}: {e}",
            }


def list_memories(session_id: str | None = None) -> list[dict]:
    """列出已沉淀的长期记忆（global 记忆 + 可选某会话记忆）。"""
    import asyncio

    async def _fetch():
        agent = get_agent()
        try:
            refs = await agent.record_store.search(filters={"collection": "memory"})
        except Exception:
            refs = []
        out = []
        for ref in refs[:100]:
            try:
                data = await agent.record_store.get(ref)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except Exception:
                        pass
                if isinstance(data, dict):
                    if session_id and data.get("provenance", {}).get("session_id") not in (None, session_id):
                        continue
                    out.append(
                        {
                            "summary": data.get("summary") or "",
                            "body": data.get("body"),
                            "scope": data.get("memory_scope"),
                            "tags": data.get("tags"),
                        }
                    )
            except Exception:
                continue
        return out

    try:
        return asyncio.run(_fetch())
    except Exception as e:
        return [{"summary": f"读取记忆失败: {type(e).__name__}: {e}", "body": None, "scope": None, "tags": None}]


# ---------------------------------------------------------------- CLI
def cli_single(question: str):
    r = ask(question)
    if r.get("error"):
        print(f"[错误] {r['error']}")
        return 1
    print(r["answer"])
    return 0


def cli_chat():
    print(f"本地智能机器人（{LLM_MODEL} @ {LLM_BASE_URL}）\n输入 exit 退出；输入 /mem 查看已沉淀记忆\n")
    sid = f"cli-{int(time.time())}"
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q == "/mem":
            for m in list_memories():
                print(f"  - [{m['scope']}] {m['summary']}")
            continue
        if not q:
            continue
        r = ask(q, sid)
        if r.get("error"):
            print(f"[错误] {r['error']}")
        else:
            print(f"机器人> {r['answer']}\n")


# ---------------------------------------------------------------- Web
app = Flask(__name__)


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def page():
    return PAGE_HTML


@app.route("/health")
def health():
    return jsonify({"ok": True, "model": LLM_MODEL, "base": LLM_BASE_URL,
                    "memory_dir": str(MEMORY_ROOT)})


@app.route("/api/qa", methods=["POST"])
def api_qa():
    data = request.get_json(force=True, silent=True) or {}
    r = ask(data.get("question", ""), data.get("session_id"))
    return jsonify(r)


@app.route("/api/memories", methods=["GET"])
def api_memories():
    sid = request.args.get("session_id") or None
    return jsonify({"memories": list_memories(sid)})


PAGE_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>本地智能机器人</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--fg:#e2e8f0;--sub:#94a3b8;--ac:#38bdf8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC",sans-serif;background:var(--bg);color:var(--fg);min-height:100vh}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;margin-bottom:4px}.sub{color:var(--sub);font-size:12px;margin-bottom:20px}
.card{background:var(--card);border-radius:12px;padding:16px;margin-bottom:16px}
.chat{height:420px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;margin-bottom:12px}
.msg{max-width:78%;padding:10px 14px;border-radius:10px;line-height:1.6;font-size:14px;white-space:pre-wrap}
.user{align-self:flex-end;background:#0ea5e9;color:#082f49}
.bot{align-self:flex-start;background:#334155}
.bot .mem{font-size:10px;color:var(--sub);display:block;margin-top:6px}
.inrow{display:flex;gap:8px}
.inrow textarea{flex:1;resize:none;background:#0f172a;border:1px solid #334155;color:var(--fg);
  border-radius:8px;padding:10px;font-size:14px;font-family:inherit;min-height:44px}
.inrow button{background:var(--ac);color:#082f49;border:0;border-radius:8px;padding:0 20px;font-weight:600;cursor:pointer}
.inrow button:disabled{opacity:.5;cursor:wait}
.memlist{max-height:260px;overflow-y:auto}
.mi{font-size:13px;padding:8px 10px;border-bottom:1px solid #334155;color:#cbd5e1}
.mi b{color:var(--ac);font-weight:600}
.tag{font-size:10px;color:var(--sub);margin-left:6px}
.empty{color:var(--sub);font-size:13px;padding:8px}
.load{color:var(--sub);font-size:12px;text-align:center;padding:8px}
</style>
</head>
<body>
<div class="wrap">
<h1>本地智能机器人</h1>
<div class="sub">Agently 4.x + Ollama 本地模型 · 自带 Agent 记忆（自动沉淀与召回）· 模型: __MODEL__</div>
<div class="card">
  <div class="chat" id="chat">
    <div class="msg bot">你好，我是本地智能机器人 🎉 我现在完全在本地运行（不依赖任何云账户），并且拥有记忆：
<ul style="margin:6px 0 0 18px"><li>同一会话内连续对话</li><li>自动记住关键事实，下次对话直接引用</li></ul>
叫我记住什么，或直接提问吧。</div>
  </div>
  <div class="inrow">
    <textarea id="q" placeholder="输入问题…（Enter 发送）"></textarea>
    <button id="send" onclick="ask()">发送</button>
  </div>
</div>
<div class="card">
  <div style="font-size:14px;font-weight:600;margin-bottom:8px">🧠 已沉淀记忆 <span id="mcount" class="tag"></span></div>
  <div class="memlist" id="memlist"><div class="load">加载中…</div></div>
</div>
</div>
<script>
let sid = localStorage.getItem("bot_sid");
if (!sid) { sid = "web-" + Math.random().toString(36).slice(2, 10); localStorage.setItem("bot_sid", sid); }
const chat = document.getElementById("chat"), q = document.getElementById("q");
function add(cls, text, memNote) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  if (memNote) { const s = document.createElement("span"); s.className = "mem"; s.textContent = memNote; d.appendChild(s); }
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}
async function ask() {
  const text = q.value.trim(); if (!text) return;
  add("user", text); q.value = "";
  const btn = document.getElementById("send"); btn.disabled = true;
  add("bot", "思考中…");
  try {
    const r = await fetch("/api/qa", { method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: text, session_id: sid}) });
    const j = await r.json();
    chat.lastElementChild.remove();
    if (j.error) { add("bot", "[错误] " + j.error); }
    else { add("bot", j.answer, "session: " + j.session_id); }
  } catch (e) { chat.lastElementChild.remove(); add("bot", "[错误] " + e); }
  btn.disabled = false; loadMem();
}
async function loadMem() {
  try {
    const r = await fetch("/api/memories"); const j = await r.json();
    const box = document.getElementById("memlist");
    box.innerHTML = "";
    document.getElementById("mcount").textContent = j.memories.length + " 条";
    if (!j.memories.length) { box.innerHTML = '<div class="empty">暂无记忆 — 聊几句后这里会出现自动沉淀的事实</div>'; return; }
    for (const m of j.memories.slice(0, 60)) {
      const d = document.createElement("div"); d.className = "mi";
      d.innerHTML = "<b>" + escapeHtml(m.scope || "?") + "</b> " + escapeHtml(m.summary || "");
      box.appendChild(d);
    }
  } catch (e) {}
}
function escapeHtml(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
q.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
loadMem();
</script>
</body>
</html>""".replace("__MODEL__", LLM_MODEL)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="本地智能机器人 (Agently + Ollama)")
    ap.add_argument("question", nargs="?", help="单次问答的问题")
    ap.add_argument("--chat", action="store_true", help="交互模式")
    ap.add_argument("--serve", action="store_true", help="启动 Web 服务")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    if args.serve:
        print(f"本地智能机器人: http://{args.host}:{args.port}  (model={LLM_MODEL})")
        app.run(host=args.host, port=args.port, threaded=True)
    elif args.chat:
        cli_chat()
    elif args.question:
        sys.exit(cli_single(args.question))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()