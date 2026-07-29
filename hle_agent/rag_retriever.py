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


def _split_chunks(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


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
