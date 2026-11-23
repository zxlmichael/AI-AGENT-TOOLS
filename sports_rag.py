#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛事分析 RAG 知识库：把历史归档报告切片、向量化，供 Agent 检索增强。

设计要点：
  - 纯标准库实现（urllib 调百炼 text-embedding-v3），不依赖第三方包，
    可在 Desktop 项目环境与 ~/SportsWeekly 定时任务环境同时运行；
  - 增量更新：只对新增/变化/删除的报告重新处理，未变化的复用旧向量；
    注意：只有本环境能扫描到的文件变化才会生效（未扫描到的来源保留原块，
    清理请用 --rebuild；每日任务环境只归档目录，Desktop 环境两个目录都可见）；
  - 索引存放在 ~/SportsWeekly/app/sports_reports/rag_index/（与每日归档同目录，
    定时任务与 Agent 共用同一份；该目录位于 Home 下，launchd 可读写）。

用法：
    python sports_rag.py --update         # 增量更新（每日任务调用）
    python sports_rag.py --rebuild        # 全量重建索引
    python sports_rag.py --query "F1 下一站"   # 检索测试
    python sports_rag.py --stats          # 查看索引状态
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAG_DIR = Path.home() / "SportsWeekly" / "app" / "sports_reports" / "rag_index"
META_PATH = RAG_DIR / "meta.json"
VEC_PATH = RAG_DIR / "vectors.json"
LOCK_PATH = RAG_DIR / ".lock"

EMBED_MODEL = "text-embedding-v3"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_BATCH = 10          # 一次请求最多向量化的文本数（接口上限）
CHUNK_MAX = 700           # 单个切片的最大字符数
TOP_K = 4

_CACHE = {"mtime": -1.0, "index": None}


# --------------------------------------------------------------------------- #
# 扫描与切片
# --------------------------------------------------------------------------- #

def sources() -> list[Path]:
    """索引来源：定时任务归档目录 +（存在时）项目内 Agent 生成的报告目录。"""
    dirs = [Path.home() / "SportsWeekly" / "app" / "sports_reports",
            HERE / "sports_reports"]
    out, seen = [], set()
    for d in dirs:
        try:
            rp = str(d.resolve())
        except OSError:
            continue
        if d.is_dir() and rp not in seen:
            seen.add(rp)
            out.append(d)
    return out


def _scan(report_dirs: list[Path]) -> dict:
    """返回 {真实路径: (mtime, size)} 的报告文件清单。"""
    found = {}
    for d in report_dirs:
        for p in sorted(d.glob("*.md")):
            try:
                st = p.stat()
            except OSError:
                continue
            found[str(p.resolve())] = (st.st_mtime, st.st_size)
    return found


def _split_chunks(text: str, max_len: int = CHUNK_MAX) -> list[str]:
    """按二级/三级标题切块；超过 max_len 的块按标点二次切分。"""
    chunks, buf = [], []
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.+)$", line.strip())
        if m:
            if buf:
                chunks.append("\n".join(buf).strip())
            buf = [line]
            continue
        if line.strip():
            buf.append(line)
    if buf:
        chunks.append("\n".join(buf).strip())

    out = []
    for body in chunks:
        if len(body) < 15:
            continue
        while len(body) > max_len:
            cut = max(body.rfind(s, 0, max_len) for s in ("。", "；", "！", "？", "\n", "，"))
            if cut < max_len // 2:
                cut = max_len
            out.append(body[:cut])
            body = body[cut:].lstrip("。，；、 \n")
        if body.strip():
            out.append(body)
    return out


def _parse(path: Path) -> tuple[list[str], str, str, str]:
    """返回 (chunks, 标题, 赛事, 日期)。"""
    text = path.read_text("utf-8", errors="replace")
    title = ""
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        title = m.group(1).strip()
    m = re.search(r"(f1|nba|epl|golf|all)_report_(\d{8})", path.name)
    sport = m.group(1) if m else "all"
    date = f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}" if m else ""
    return _split_chunks(text), title or path.stem, sport, date


# --------------------------------------------------------------------------- #
# 向量化（百炼 embedding，urllib 直接调用）
# --------------------------------------------------------------------------- #

def _load_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text("utf-8").splitlines():
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _ssl_ctx() -> ssl.SSLContext:
    # 最小化 venv（~/SportsWeekly）里只有 certifi 可用，用它构造 CA 上下文
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _embed(texts: list[str]) -> list[list[float]]:
    key = _load_key()
    if not key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        payload = json.dumps({"model": EMBED_MODEL, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            EMBED_URL, data=payload,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "data" not in data or len(data["data"]) != len(batch):
            raise RuntimeError(f"embedding 返回异常: {str(data)[:200]}")
        out.extend(item["embedding"] for item in data["data"])
        if i + EMBED_BATCH < len(texts):
            time.sleep(0.2)  # 限速，避免触发百炼并发限制
    return out


# --------------------------------------------------------------------------- #
# 索引读写与更新
# --------------------------------------------------------------------------- #

def _load_index():
    if not META_PATH.exists() or not VEC_PATH.exists():
        return None
    try:
        meta = json.loads(META_PATH.read_text("utf-8"))
        vecs = json.loads(VEC_PATH.read_text("utf-8"))
    except Exception:
        return None
    docs = meta.get("docs") or []
    if not docs or len(docs) != len(vecs):
        return None
    return meta, vecs


def _cached_index():
    try:
        mtime = META_PATH.stat().st_mtime
    except OSError:
        return None
    if _CACHE["index"] is None or _CACHE["mtime"] != mtime:
        _CACHE["index"] = _load_index()
        _CACHE["mtime"] = mtime
    return _CACHE["index"]


def update(force: bool = False, quiet: bool = False) -> dict:
    """增量更新索引；返回 {"updated","files","chunks","added"}。"""
    files = _scan(sources())
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)  # 防止每日任务与 Agent 同时写
        except Exception:
            pass

        index = _load_index()
        if index and not force:
            old_docs, old_vecs = index[0]["docs"], index[1]
        else:
            old_docs, old_vecs = [], []
        by_file = {}
        for d in old_docs:
            by_file.setdefault(d["src"], []).append(d)

        changed = [p for p, (m, s) in files.items()
                   if force or p not in by_file
                   or by_file[p][0]["mtime"] != m or by_file[p][0]["size"] != s]

        # 先向量化新增/变化文件；失败的文件保留旧块，避免索引回退
        fresh_docs, fresh_vecs, failed = [], [], set()
        for path in sorted(changed):
            try:
                chunks, title, sport, date = _parse(Path(path))
                if not chunks:
                    continue
                vec = _embed(chunks)
            except Exception as exc:
                if not quiet:
                    print(f"[rag] 跳过 {Path(path).name}: {exc}", file=sys.stderr)
                failed.add(path)
                continue
            mtime, size = files[path]
            for text, v in zip(chunks, vec):
                fresh_docs.append({"text": text, "title": title, "sport": sport,
                                   "date": date, "src": path,
                                   "mtime": mtime, "size": size})
                fresh_vecs.append(v)

        # 移除：仅移除「本环境扫描到且已成功变更」的旧块。
        # 不在当前扫描范围的文件（本环境看不到的另一个来源目录）不删除，
        # 否则每日任务（只能看到归档目录）会把 Agent 生成的报告块误删；
        # 清理已删除文件请用 --rebuild 全量重建（在能看到全部来源的 Desktop 环境执行）。
        remove = set(changed) - failed
        keep_pairs = [(d, v) for d, v in zip(old_docs, old_vecs) if d["src"] not in remove]
        docs = [d for d, _ in keep_pairs] + fresh_docs
        vecs = [v for _, v in keep_pairs] + fresh_vecs

        meta = {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": EMBED_MODEL,
            "n_chunks": len(docs),
            "docs": docs,
        }
        tmp_m = META_PATH.with_name("meta.json.tmp")
        tmp_v = VEC_PATH.with_name("vectors.json.tmp")
        tmp_m.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
        tmp_v.write_text(json.dumps(vecs, ensure_ascii=False), "utf-8")
        os.replace(tmp_m, META_PATH)
        os.replace(tmp_v, VEC_PATH)
        return {"updated": meta["updated"], "files": len(files),
                "chunks": len(docs), "added": len(fresh_docs)}


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #

def _cosine(qv: list[float], vecs: list[list[float]]) -> list[float]:
    try:
        import numpy as np
        q = np.asarray(qv, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        m = np.asarray(vecs, dtype=np.float32)
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        return (m @ q).tolist()
    except Exception:
        qn = math.sqrt(sum(x * x for x in qv)) + 1e-9
        out = []
        for v in vecs:
            dot = sum(a * b for a, b in zip(qv, v))
            vn = math.sqrt(sum(a * a for a in v)) + 1e-9
            out.append(dot / (qn * vn))
        return out


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """向量检索，返回 [{text,title,sport,date,score,src}]；索引不可用时返回 []。"""
    query = (query or "").strip()
    if not query:
        return []
    index = _cached_index()
    if not index:
        return []
    docs, vecs = index[0]["docs"], index[1]
    try:
        qv = _embed([query])[0]
    except Exception:
        return []
    scores = _cosine(qv, vecs)
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(1, k)]
    return [{"text": docs[i]["text"], "title": docs[i]["title"],
             "sport": docs[i]["sport"], "date": docs[i].get("date", ""),
             "score": round(scores[i], 4), "src": docs[i]["src"]}
            for i in best if scores[i] > 0]


def stats() -> dict:
    index = _cached_index()
    if not index:
        return {"ok": False, "n_chunks": 0, "n_files": 0}
    docs = index[0]["docs"]
    return {"ok": True, "updated": index[0].get("updated", ""),
            "n_chunks": len(docs), "n_files": len({d["src"] for d in docs}),
            "model": index[0].get("model", "")}


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="赛事分析 RAG 知识库")
    ap.add_argument("--update", action="store_true", help="增量更新索引（默认动作）")
    ap.add_argument("--rebuild", action="store_true", help="全量重建索引")
    ap.add_argument("--query", metavar="Q", help="检索测试")
    ap.add_argument("--stats", action="store_true", help="查看索引状态")
    ap.add_argument("--quiet", action="store_true", help="不输出错误详情")
    args = ap.parse_args()

    if args.stats:
        print(json.dumps(stats(), ensure_ascii=False, indent=2))
        return

    if args.query:
        hits = retrieve(args.query, k=TOP_K)
        if not hits:
            print("（无结果，或索引未建立/向量化失败）")
            return
        for h in hits:
            print(f"[{h['score']:.3f}] 《{h['title']}》({h['sport']} {h['date']})")
            print("  " + h["text"][:160].replace("\n", " "))
            print()
        return

    res = update(force=args.rebuild, quiet=args.quiet)
    print(f"索引已更新：{res['updated']}，文件 {res['files']} 份，"
          f"切片 {res['chunks']} 个，新增 {res['added']} 个")


if __name__ == "__main__":
    main()