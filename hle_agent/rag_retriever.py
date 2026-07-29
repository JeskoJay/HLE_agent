"""
RAG 检索器：领域知识预检索（替代原 Forge 的规则注入）。

设计要点：
- 可插拔检索接口：RetrieverBase.retrieve(query, k) -> List[(text, score)]
- 当前提供基于 TF-IDF 的词频检索（SimpleRAGRetriever），零额外依赖，
  适用于用户后续放入 knowledge_base/ 的 .txt/.md 资料。
- 预留向量检索接口（VectorRAGRetriever）：当提供 embedding 函数时启用余弦相似度。
- 资料为空时返回 []，Agent 退化为纯模型推理（不影响管线）。

知识库使用方式：
  将领域资料（.txt 或 .md）放入 knowledge_base/ 目录，
  运行时会自动分块并建立索引。支持子目录。
"""
import os
import re
import math
from collections import Counter

import config

CHUNK_SIZE = 800          # 每块字符数
CHUNK_OVERLAP = 100
SUPPORTED_EXT = {".txt", ".md", ".json"}


def _strip_frontmatter(text: str) -> str:
    """去掉 YAML 前言（--- ... ---），避免其中的 scope 关键词污染检索。"""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


def _split_into_sections(text: str) -> list:
    """按 markdown 标题（# ~ ######）切分，使每个知识卡片/章节成为独立片段，
    提升 top-k 检索的语义相关性（避免把一个 K 卡片从中间截断导致命中错误卡片）。"""
    lines = text.splitlines()
    sections, cur = [], []
    for ln in lines:
        if re.match(r"^#{1,6}\s", ln):
            if cur:
                sections.append("\n".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append("\n".join(cur))
    return [s.strip() for s in sections if s.strip()]


def _fixed_chunk(text: str, size: int, overlap: int) -> list:
    text = text.strip()
    if not text:
        return []
    out = []
    start = 0
    while start < len(text):
        end = start + size
        piece = text[start:end]
        if piece.strip():
            out.append(piece.strip())
        if end >= len(text):
            break
        start = end - overlap
    return out


def _split_chunks(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    text = _strip_frontmatter(text.strip())
    if not text:
        return []
    chunks = []
    for sec in _split_into_sections(text):
        if len(sec) <= size:
            chunks.append(sec)
            continue
        # 超长章节：固定窗口切片，并把标题前置以保留上下文与检索关键词
        heading = sec.splitlines()[0] if sec.startswith("#") else ""
        body = sec[len(heading):].strip() if heading else sec
        for sub in _fixed_chunk(body, size, overlap):
            chunks.append(f"{heading}\n\n{sub}".strip() if heading else sub)
    return [c for c in chunks if c.strip()]


def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class RetrieverBase:
    def retrieve(self, query: str, k: int = 4) -> list:
        raise NotImplementedError


class SimpleRAGRetriever(RetrieverBase):
    """TF-IDF 词频检索，零依赖。"""

    def __init__(self, kb_dir: str = None):
        self.kb_dir = kb_dir or config.KB_DIR
        self.doc_chunks = []        # list of {"text":..., "source":...}
        self.df = Counter()         # 词 -> 出现文档数
        self.idf = {}
        self._built = False

    def build(self):
        if self._built:
            return
        if not os.path.isdir(self.kb_dir):
            self._built = True
            return
        for root, _, files in os.walk(self.kb_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXT:
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                for chunk in _split_chunks(text):
                    toks = _tokenize(chunk)
                    self.doc_chunks.append({
                        "text": chunk,
                        "source": os.path.relpath(path, self.kb_dir),
                        "tf": Counter(toks),
                    })
                    for w in set(toks):
                        self.df[w] += 1
        n_docs = max(len(self.doc_chunks), 1)
        self.idf = {w: math.log((n_docs + 1) / (c + 1)) + 1 for w, c in self.df.items()}
        self._built = True

    def retrieve(self, query: str, k: int = 4) -> list:
        self.build()
        if not self.doc_chunks:
            return []
        q_toks = _tokenize(query)
        q_tf = Counter(q_toks)
        scores = []
        for doc in self.doc_chunks:
            score = 0.0
            for w, c in q_tf.items():
                if w in doc["tf"]:
                    score += (c) * self.idf.get(w, 0) * doc["tf"][w]
            scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, doc in scores[:k]:
            if s <= 0:
                continue
            out.append((doc["text"], round(s, 4), doc["source"]))
        return out


class VectorRAGRetriever(RetrieverBase):
    """
    向量检索（预留）。需要提供 embedding 函数：
        embed_fn(texts: List[str]) -> List[List[float]]
    可由用户后续接入本地 sentence-transformers，或任意 embedding API。
    """

    def __init__(self, embed_fn=None, kb_dir: str = None):
        self.embed_fn = embed_fn
        self.kb_dir = kb_dir or config.KB_DIR
        self.chunks = []
        self.vectors = []
        self._built = False

    def build(self):
        if self._built:
            return
        if self.embed_fn is None:
            raise RuntimeError("VectorRAGRetriever requires embed_fn")
        if not os.path.isdir(self.kb_dir):
            self._built = True
            return
        raw = []
        for root, _, files in os.walk(self.kb_dir):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in SUPPORTED_EXT:
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                for chunk in _split_chunks(text):
                    raw.append((chunk, os.path.relpath(path, self.kb_dir)))
        if raw:
            texts = [r[0] for r in raw]
            self.vectors = self.embed_fn(texts)
            self.chunks = [{"text": r[0], "source": r[1]} for r in raw]
        self._built = True

    def retrieve(self, query: str, k: int = 4) -> list:
        self.build()
        if not self.chunks:
            return []
        qv = self.embed_fn([query])[0]
        scored = []
        for doc, vec in zip(self.chunks, self.vectors):
            sim = self._cosine(qv, vec)
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(d["text"], round(s, 4), d["source"]) for s, d in scored[:k]]

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0


def get_retriever(mode: str = "simple", embed_fn=None) -> RetrieverBase:
    """工厂：根据 config.ENABLE_RAG 与模式返回检索器。"""
    if not config.ENABLE_RAG:
        return None
    if mode == "vector":
        return VectorRAGRetriever(embed_fn=embed_fn)
    return SimpleRAGRetriever()
