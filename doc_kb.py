# -*- coding: utf-8 -*-
"""
doc_kb.py — 本地文档统计 + 本地知识库 (纯标准库)

功能:
  1. scan_stats()   扫描目录树, 统计文档数量/字符数/中文字符/行数/扩展名/目录/TOP文件
  2. RAG 知识库     update() 切片向量化(百炼 text-embedding-v3) -> retrieve() 相似度检索
                    设计同 comate_rag.py (meta.json + vectors.json + fcntl.lock + 原子写)
  3. 支持 .docx 文字提取(zipfile + XML, 无需第三方库)

用法:
    python doc_kb.py --stats                 # 扫描统计
    python doc_kb.py --build                 # 构建/更新知识库
    python doc_kb.py --query "RAG 是什么"    # 检索测试
"""
from __future__ import annotations

import json
import math
import os
import re
import ssl
import sys
import threading
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
# 扫描根目录(默认): 工作区 + 下载 + 文稿; src 一律存 home 相对路径
ROOTS = [HERE, HOME / "Downloads", HOME / "Documents"]
ROOTS = [r for r in ROOTS if r.is_dir()] or [HERE]
RAG_DIR = HERE / "local_docs_kb"
META_PATH = RAG_DIR / "meta.json"
VEC_PATH = RAG_DIR / "vectors.json"
LOCK_PATH = RAG_DIR / ".lock"
CACHE_PATH = RAG_DIR / "scan_cache.json"
STATUS_PATH = RAG_DIR / "build_status.json"

# 文本类扩展名(参与统计与检索); 其余视为二进制跳过
TEXT_EXTS = {
    ".md", ".txt", ".py", ".json", ".csv", ".html", ".htm", ".yml", ".yaml",
    ".xml", ".ini", ".log", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h",
    ".cpp", ".hpp", ".sql", ".sh", ".rst", ".toml", ".cfg", ".ipynb", ".docx",
}
# 知识库(向量化)只收这些格式
KB_EXTS = {".md", ".txt", ".docx"}
# 排除目录/文件(含 .env 等敏感文件; venv/node_modules 等依赖与构建产物)
EXCLUDE_DIRS = {".git", ".venv", "venv", "site-packages", "__pycache__",
                "node_modules", ".pytest_cache", ".agently", ".kiro",
                "target", "stock_assets", "comate_docs_kb", "local_docs_kb",
                "auto_export_data", "nfl_news_archive", "typeshed", "frontend"}
EXCLUDE_FILES = {".DS_Store", ".env", ".popo.json"}
MAX_READ = 3 * 1024 * 1024     # 超过 3MB 的文本只统计大小不读内容

EMBED_MODEL = "text-embedding-v3"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_BATCH = 10
CHUNK_MAX = 700
TOP_K = 4

# 本地向量模型(离线嵌入, 不依赖百炼)。KB_EMBED=auto 时: 优先百炼, 失败自动切换本地。
LOCAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
LOCAL_EMBED_DIR = RAG_DIR / "bge-small-zh-v1.5"

_DS_FAILED = False       # 百炼多次失败后置位, 直接走本地
_LAST_PROVIDER = "auto"  # 最近一次成功使用的 provider, 写入索引 meta
_local_model = None
_local_lock = threading.Lock()

_CACHE = {"mtime": -1.0, "index": None, "stats": None}


# ---------------------------------------------------------------- 基础
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
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 扫描
def _iter_files(root: Path):
    """遍历单根目录下的文本文档, 产出 (Path, size); 跳过排除目录/文件。"""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_FILES:
                continue
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if p.is_symlink() or ext not in TEXT_EXTS:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            yield p, size


def _iter_all():
    """遍历所有根目录(工作区+下载+文稿)。"""
    for r in ROOTS:
        if r.is_dir():
            yield from _iter_files(r)


def _src_of(p: Path) -> str:
    """文件在索引中的 src 标识: home 相对路径(如 Desktop/AI大模型/x.md)。"""
    try:
        return str(p.relative_to(HOME))
    except ValueError:
        return str(p)


def read_text(p: Path) -> str:
    """读取文本内容; .docx 走 zipfile+XML 提取。"""
    if p.suffix.lower() == ".docx":
        return _docx_text(p)
    return p.read_text("utf-8", errors="replace")


def _docx_text(p: Path) -> str:
    """解析 word/document.xml 中的 <w:t> 文本(纯标准库)。"""
    try:
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml")
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        root = ET.fromstring(xml)
        return "\n".join(t.text or "" for t in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
    except Exception:
        return ""


def scan_stats(force: bool = False) -> dict:
    """扫描统计(所有根目录): 文档数量/字符/中文字符/行数 + 扩展名/目录分布 + TOP 文件。
    结果缓存到 local_docs_kb/scan_cache.json。"""
    if not force and _CACHE["stats"] is not None:
        return _CACHE["stats"]
    files = []
    n_bytes = n_texts = n_chars = n_cn = n_lines = n_words = 0
    for p, size in _iter_all():
        n_bytes += size
        n_texts += 1
        rec = {"path": _src_of(p), "ext": p.suffix.lower() or "(无)", "size": size}
        if size <= MAX_READ:
            try:
                text = read_text(p)
                n_chars += len(text)
                n_cn += len(re.findall(r"[\u4e00-\u9fff]", text))
                n_lines += text.count("\n") + 1
                n_words += len(re.findall(r"[A-Za-z0-9_]+", text))
                rec.update({"chars": len(text), "cn": len(re.findall(r"[\u4e00-\u9fff]", text))})
            except Exception:
                rec.update({"chars": 0, "cn": 0})
        else:
            rec.update({"chars": 0, "cn": 0})
        files.append(rec)

    by_ext, by_dir = {}, {}
    for r in files:
        by_ext[r["ext"]] = by_ext.get(r["ext"], 0) + 1
        d = os.path.dirname(r["path"]) or "."
        by_dir[d] = by_dir.get(d, 0) + 1
    top = sorted(files, key=lambda r: -r["size"])[:15]
    out = {
        "generated": _now(),
        "root": str(ROOTS[0]),
        "roots": [str(r) for r in ROOTS],
        "n_files": len(files),
        "n_bytes": n_bytes,
        "n_chars": n_chars,
        "n_cn_chars": n_cn,
        "n_lines": n_lines,
        "n_words": n_words,
        "by_ext": sorted(by_ext.items(), key=lambda x: -x[1]),
        "by_dir": sorted(by_dir.items(), key=lambda x: -x[1])[:12],
        "top_files": top,
    }
    try:
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_name("scan_cache.json.tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass
    _CACHE["stats"] = out
    return out


# ---------------------------------------------------------------- RAG 索引
def _split_chunks(text: str, max_len: int = CHUNK_MAX) -> list[str]:
    """按二级/三级标题切块; 超过 max_len 的块按标点二次切分。"""
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


def _parse(p: Path) -> tuple[list[str], str]:
    text = read_text(p)
    title = p.stem
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        title = m.group(1).strip()
    return _split_chunks(text), title or p.stem


def _embed_local(texts: list[str]) -> list[list[float]]:
    """本地向量化: bge-small-zh-v1.5 (sentence-transformers, hf-mirror 下载)。
    注意: macOS 上多线程并发 encode 会 SIGSEGV, 故固定 CPU + 单调用方串行。"""
    global _local_model
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    with _local_lock:
        if _local_model is None:
            import os as _os
            _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像
            try:
                from sentence_transformers import SentenceTransformer
                LOCAL_EMBED_DIR.mkdir(parents=True, exist_ok=True)
                # device="cpu": 避免 MPS/GPU 线程竞争导致的 segfault
                _local_model = SentenceTransformer(LOCAL_EMBED_MODEL, cache_folder=str(LOCAL_EMBED_DIR), device="cpu")
            except Exception as e:
                raise RuntimeError(f"本地向量模型加载失败: {e}")
    import numpy as np
    emb = _local_model.encode(texts, normalize_embeddings=True, batch_size=16,
                              show_progress_bar=False, convert_to_numpy=True, device="cpu")
    return [np.asarray(v, dtype=np.float32).tolist() for v in emb]


def _embed(texts: list[str], provider: str = "auto") -> list[list[float]]:
    """向量化入口。provider:
      auto      -> 百炼优先, 失败(欠费/网络)自动切本地;
      dashscope -> 仅百炼;  local -> 仅本地。
    可用 KB_EMBED=local 环境变量强制本地。返回 [ [float...], ... ]; 全部失败抛异常。"""
    global _DS_FAILED, _LAST_PROVIDER
    force_local = os.environ.get("KB_EMBED", "auto") == "local"
    use_ds = (not force_local) and (provider == "dashscope" or (provider == "auto" and not _DS_FAILED))
    if use_ds:
        try:
            out = _embed_dashscope(texts)
            _LAST_PROVIDER = "dashscope"
            return out
        except Exception:
            _DS_FAILED = True  # 记住失败, 后续直接本地
    out = _embed_local(texts)
    _LAST_PROVIDER = "local"
    return out


def _embed_dashscope(texts: list[str]) -> list[list[float]]:
    key = _load_key()
    if not key:
        raise RuntimeError("未配置 API key")
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
            time.sleep(0.2)
    return out


def _migrate_src(src: str) -> str:
    """旧索引 src 为工作区相对路径 -> 迁移为 home 相对路径。"""
    if not src or src.startswith(("Desktop/", "Downloads/", "Documents/")) or Path(src).is_absolute():
        return src
    try:
        return str((HERE / src).resolve().relative_to(HOME))
    except ValueError:
        return src


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
    migrated = False
    for d in docs:
        new_src = _migrate_src(d.get("src", ""))
        if new_src != d.get("src"):
            d["src"] = new_src
            migrated = True
    if migrated:  # 迁移后立即回写, 避免每次加载重复迁移
        meta["docs"] = docs
        try:
            tmp = META_PATH.with_name("meta.json.tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
            os.replace(tmp, META_PATH)
        except OSError:
            pass
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


def _save_status(state: str, done: int = 0, total: int = 0, msg: str = ""):
    """持久化构建进度(网页轮询用); 每 N 个文件写盘一次。"""
    try:
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(
            {"state": state, "done": done, "total": total, "msg": msg,
             "updated": _now()}, ensure_ascii=False), "utf-8")
    except OSError:
        pass


def build_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text("utf-8"))
    except Exception:
        return {"state": "idle", "done": 0, "total": 0, "msg": ""}


def update(force: bool = False, quiet: bool = False, progress: bool = True) -> dict:
    """构建/更新知识库索引(仅 .md/.txt/.docx); 返回统计。
    progress=True 时每处理一个文件更新一次构建状态(供 /api/kb/status 轮询)。"""
    files = {}
    for p, size in _iter_all():
        if p.suffix.lower() in KB_EXTS and size <= MAX_READ:
            files[str(p.resolve())] = (p.stat().st_mtime, size)
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    if progress:
        _save_status("scanning", 0, len(files), "扫描文件清单...")
    with LOCK_PATH.open("w") as lock:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
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
        if progress:
            _save_status("building", 0, len(changed), f"待处理 {len(changed)} 个文件")

        def _index_one(job):
            """单文件: 解析+向量化, 返回 [(doc, vec), ...] 或 None(失败/无内容)。"""
            path, mtime, size = job
            try:
                chunks, title = _parse(Path(path))
                if not chunks:
                    return []
                vec = _embed(chunks)
                src = _src_of(Path(path))
                return [({"text": t, "title": title, "src": src,
                          "mtime": mtime, "size": size}, v)
                        for t, v in zip(chunks, vec)]
            except Exception as exc:
                if not quiet:
                    print(f"[kb] 跳过 {Path(path).name}: {exc}", file=sys.stderr)
                return None

        jobs = [(p, files[p][0], files[p][1]) for p in sorted(changed)]
        fresh_docs, fresh_vecs, failed = [], [], set()
        done_n = 0
        _done_lock = threading.Lock() if progress else None

        def _progress_ok(job_idx, path):
            nonlocal done_n
            with _done_lock:
                done_n = job_idx
                if progress:
                    _save_status("building", done_n, len(jobs), os.path.basename(path))

        # 本地向量化时并发 encode 会 SIGSEGV -> 只用单线程; 百炼 API 场景才并行
        pool_workers = 3 if (not _DS_FAILED and os.environ.get("KB_EMBED") != "local") else 1
        if len(jobs) >= 2 and pool_workers > 1:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=pool_workers) as ex:
                futs = {ex.submit(_index_one, jb): jb for jb in jobs}
                for idx, (fut, jb) in enumerate(sorted(futs.items(), key=lambda kv: jobs.index(kv[1])), 1):
                    res = fut.result()
                    if res is None:
                        failed.add(jb[0])
                    elif res:
                        for d, v in res:
                            fresh_docs.append(d)
                            fresh_vecs.append(v)
                    _progress_ok(idx, jb[0])
        else:
            for idx, jb in enumerate(jobs, 1):
                res = _index_one(jb)
                if res is None:
                    failed.add(jb[0])
                elif res:
                    for d, v in res:
                        fresh_docs.append(d)
                        fresh_vecs.append(v)
                _progress_ok(idx, jb[0])

        remove = set(changed) - failed
        # src 为 home 相对路径, changed 为绝对路径 -> 统一比较
        changed_abs = {os.path.abspath(p) for p in changed}
        removed_abs = {os.path.abspath(p) for p in remove}
        keep_pairs = []
        for d, v in zip(old_docs, old_vecs):
            ap = os.path.abspath(os.path.join(HOME, d["src"]))
            if ap in changed_abs and ap in removed_abs:
                continue  # 该文件已变更为新块
            # 已删除的文件: 无法确认真实来源时保留(与 comate_rag 一致, 重建才清理)
            keep_pairs.append((d, v))
        docs = [d for d, _ in keep_pairs] + fresh_docs
        vecs = [v for _, v in keep_pairs] + fresh_vecs

        meta = {
            "updated": _now(),
            "model": EMBED_MODEL if _LAST_PROVIDER != "local" else LOCAL_EMBED_MODEL,
            "provider": _LAST_PROVIDER,   # 索引使用的向量供应商(检索必须一致)
            "n_chunks": len(docs),
            "n_kb_files": len({d["src"] for d in docs}),
            "docs": docs,
        }
        tmp_m = META_PATH.with_name("meta.json.tmp")
        tmp_v = VEC_PATH.with_name("vectors.json.tmp")
        tmp_m.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
        tmp_v.write_text(json.dumps(vecs, ensure_ascii=False), "utf-8")
        os.replace(tmp_m, META_PATH)
        os.replace(tmp_v, VEC_PATH)
        _CACHE["index"] = None
        skipped = len(failed)
        if progress:
            _save_status("done", len(changed), len(changed),
                         f"共 {len(docs)} 个切片, 跳过 {skipped} 个文件")
        return {"updated": meta["updated"], "files": len(files),
                "chunks": len(docs), "added": len(fresh_docs), "failed": skipped}


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


def _kw_terms(query: str) -> list[str]:
    """中文按双字切分 + 拉丁词, 作为关键词检索词(轻量分词)。"""
    terms = re.findall(r"[A-Za-z0-9_]{2,}", query)
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", query)
    for i in range(len(cjk) - 1):
        terms.append(cjk[i:i + 2])
    # 过滤纯停用字双字
    stop = {"什么", "如何", "怎么", "进行", "一个", "这个", "那个", "可以", "一下", "我们", "你们", "他们", "没有", "不是", "就是", "还是", "或者"}
    return [t for t in terms if len(t) >= 2 and t not in stop][:40]


def _kw_search(query: str, k: int = TOP_K) -> list[dict]:
    """关键词降级检索(无向量/向量失败时用): 对内存中的切片按词频打分。"""
    index = _cached_index()
    if not index:
        return []
    docs = index[0]["docs"]
    terms = _kw_terms(query)
    if not terms:
        return []
    scored = []
    for i, d in enumerate(docs):
        text = d["text"]
        score = 0
        for t in terms:
            c = text.count(t)
            if c:
                score += c * (1 + (len(t) > 2))  # 长词加权
        if score:
            scored.append((score * 10, i))  # 放大便于与向量分区分
    scored.sort(key=lambda x: -x[0])
    out = []
    for s, i in scored[:max(1, k)]:
        d = docs[i]
        out.append({"text": d["text"], "title": d["title"],
                    "score": round(s / 10, 2), "src": d["src"], "method": "kw"})
    return out


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """语义检索(向量) -> 失败自动降级为关键词检索; 返回 [{text,title,score,src,method?}]。
    向量供应商取索引 meta.provider(构建时确定), 保证向量维度一致。"""
    query = (query or "").strip()
    if not query:
        return []
    index = _cached_index()
    if not index:
        return []
    docs, vecs = index[0]["docs"], index[1]
    prov = index[0].get("provider", "auto") or "auto"
    try:
        qv = _embed([query], provider=prov)[0]
    except Exception:
        return _kw_search(query, k=k)
    scores = _cosine(qv, vecs)
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(1, k)]
    hits = [{"text": docs[i]["text"], "title": docs[i]["title"],
             "score": round(scores[i], 4), "src": docs[i]["src"], "method": "vector"}
            for i in best if scores[i] > 0]
    if not hits:
        return _kw_search(query, k=k)
    return hits


def stats() -> dict:
    index = _cached_index()
    if not index:
        return {"ok": False, "n_chunks": 0, "n_files": 0}
    docs = index[0]["docs"]
    return {"ok": True, "updated": index[0].get("updated", ""),
            "n_chunks": len(docs), "n_files": len({d["src"] for d in docs}),
            "model": index[0].get("model", "")}


# ---------------------------------------------------------------- 命令行
def main():
    import argparse
    ap = argparse.ArgumentParser(description="本地文档统计 + 本地知识库")
    ap.add_argument("--stats", action="store_true", help="扫描统计")
    ap.add_argument("--build", action="store_true", help="构建/更新知识库索引")
    ap.add_argument("--query", metavar="Q", help="检索测试")
    ap.add_argument("--force", action="store_true", help="全量重建")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.stats:
        s = scan_stats()
        print(f"扫描根: {s['root']}")
        print(f"文档数: {s['n_files']} | 总大小: {s['n_bytes']/1e6:.1f}MB | "
              f"总字符: {s['n_chars']:,} | 中文字符: {s['n_cn_chars']:,} | 总行数: {s['n_lines']:,}")
        print("扩展名 TOP5:", ", ".join(f"{e}({c})" for e, c in s["by_ext"][:5]))
        return
    if args.build:
        r = update(force=args.force, quiet=args.quiet)
        print(f"知识库已更新: {r['updated']}, 待索引文件 {r['files']} 份, "
              f"切片 {r['chunks']} 个, 新增 {r['added']} 个", file=sys.stderr if not args.quiet else sys.stdout)
        return
    if args.query:
        hits = retrieve(args.query)
        if not hits:
            print("（无结果, 或索引未建立/未配置 API key）")
            return
        for h in hits:
            print(f"[{h['score']:.3f}] 《{h['title']}》 ({h['src']})")
            print("  " + h["text"][:150].replace("\n", " "))
            print()
        return
    ap.print_help()


if __name__ == "__main__":
    main()