#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hello-agents 1.0 兼容的轻量记忆与 RAG 后端。

MemoryTool / RAGTool 已在 1.0 中移除，本模块提供等价能力：
- 本地 JSON 记忆库（按 user_id 隔离）
- PDF 切块 + Embedding 检索（优先 Qdrant，否则本地向量库）
- 用 HelloAgentsLLM 生成答案
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from pypdf import PdfReader

from hello_agents import HelloAgentsLLM


@dataclass
class MemoryItem:
    id: str
    content: str
    memory_type: str = "working"
    importance: float = 0.5
    concept: str = "general"
    event_type: str = ""
    session_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SimpleMemoryStore:
    """按用户隔离的本地记忆存储。"""

    def __init__(self, user_id: str, root: str = "./memory_store"):
        self.user_id = user_id
        self.path = Path(root) / f"{user_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: List[MemoryItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.items = [MemoryItem(**item) for item in data]

    def _save(self) -> None:
        payload = [asdict(item) for item in self.items]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        concept: str = "general",
        event_type: str = "",
        session_id: str = "",
    ) -> str:
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            importance=importance,
            concept=concept or "general",
            event_type=event_type,
            session_id=session_id,
        )
        self.items.append(item)
        self._save()
        return f"✅ 已添加{memory_type}记忆: {content[:80]}"

    def search(self, query: str, limit: int = 5) -> str:
        query_l = query.lower()
        scored = []
        for item in self.items:
            score = item.importance
            if any(tok in item.content.lower() for tok in re.findall(r"[\w\u4e00-\u9fff]+", query_l)):
                score += 1.0
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[:limit]]
        if not top:
            return "暂无相关记忆。"
        lines = [
            f"- [{item.memory_type}] {item.content} (importance={item.importance})"
            for item in top
        ]
        return "\n".join(lines)

    def summary(self, limit: int = 10) -> str:
        if not self.items:
            return "暂无记忆。"
        recent = sorted(self.items, key=lambda x: x.created_at, reverse=True)[:limit]
        lines = [
            f"- [{item.memory_type}] {item.content}"
            for item in recent
        ]
        return "\n".join(lines)


class SimpleRAGEngine:
    """PDF -> chunk -> embed -> retrieve -> LLM ask。"""

    def __init__(self, namespace: str, root: str = "./rag_store"):
        self.namespace = namespace
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "chunks.json"
        self.vec_path = self.root / "vectors.npy"

        self.embed_model = os.getenv("EMBED_MODEL_NAME") or "text-embedding-v3"
        self.embed_client = OpenAI(
            api_key=os.getenv("EMBED_API_KEY") or os.getenv("LLM_API_KEY"),
            base_url=os.getenv("EMBED_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("LLM_MODEL_ID") or os.getenv("CORECODER_MODEL")
        if not all([api_key, base_url, model]):
            raise ValueError("缺少 LLM 配置：请设置 LLM_* 或 OPENAI_* / CORECODER_MODEL")
        self.llm = HelloAgentsLLM(model=model, api_key=api_key, base_url=base_url)

        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self._load()

        self.qdrant = None
        self.collection = f"{os.getenv('QDRANT_COLLECTION', 'hello_agents_vectors')}_{namespace}"
        self._maybe_init_qdrant()

    def _load(self) -> None:
        if self.meta_path.exists():
            self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.vec_path.exists():
            self.vectors = np.load(self.vec_path)

    def _save(self) -> None:
        self.meta_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.vectors is not None:
            np.save(self.vec_path, self.vectors)

    def _maybe_init_qdrant(self) -> None:
        url = (os.getenv("QDRANT_URL") or "").strip()
        api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
        if not url or "your-cluster" in url:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm

            self.qdrant = QdrantClient(url=url, api_key=api_key or None, timeout=30)
            # 延迟到首次写入时按向量维度建 collection
            self._qm = qm
        except Exception as e:
            print(f"⚠️ Qdrant 不可用，改用本地向量库: {e}")
            self.qdrant = None

    def _extract_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("未能从 PDF 提取到文本，请确认文件可复制文本")
        return text

    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text)
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(end - chunk_overlap, start + 1)
        return chunks

    def _embed(self, texts: List[str]) -> np.ndarray:
        # dashscope compatible embeddings API
        resp = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        vectors = [item.embedding for item in resp.data]
        arr = np.asarray(vectors, dtype=np.float32)
        # L2 normalize for cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

    def _ensure_qdrant_collection(self, dim: int) -> None:
        if self.qdrant is None:
            return
        qm = self._qm
        names = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in names:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def add_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> str:
        text = self._extract_pdf(file_path)
        pieces = self._chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not pieces:
            raise ValueError("文档切块为空")

        # batch embed
        batch_size = 16
        vectors = []
        for i in range(0, len(pieces), batch_size):
            vectors.append(self._embed(pieces[i : i + batch_size]))
        mat = np.vstack(vectors)

        start_id = len(self.chunks)
        new_chunks = []
        for i, piece in enumerate(pieces):
            new_chunks.append(
                {
                    "id": start_id + i,
                    "text": piece,
                    "source": os.path.basename(file_path),
                }
            )

        if self.vectors is None or len(self.chunks) == 0:
            self.chunks = new_chunks
            self.vectors = mat
        else:
            self.chunks.extend(new_chunks)
            self.vectors = np.vstack([self.vectors, mat])

        self._save()

        if self.qdrant is not None:
            self._ensure_qdrant_collection(mat.shape[1])
            points = []
            for chunk, vec in zip(new_chunks, mat):
                points.append(
                    self._qm.PointStruct(
                        id=chunk["id"],
                        vector=vec.tolist(),
                        payload={"text": chunk["text"], "source": chunk["source"]},
                    )
                )
            self.qdrant.upsert(collection_name=self.collection, points=points)

        return (
            f"✅ 已索引文档 {os.path.basename(file_path)}："
            f"{len(pieces)} 个片段，向量维度 {mat.shape[1]}，"
            f"后端={'qdrant+local' if self.qdrant else 'local'}"
        )

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or self.vectors is None:
            return []
        q = self._embed([query])[0]

        if self.qdrant is not None:
            try:
                hits = self.qdrant.search(
                    collection_name=self.collection,
                    query_vector=q.tolist(),
                    limit=limit,
                )
                return [
                    {
                        "text": hit.payload.get("text", ""),
                        "source": hit.payload.get("source", ""),
                        "score": float(hit.score),
                    }
                    for hit in hits
                ]
            except Exception as e:
                print(f"⚠️ Qdrant 检索失败，回退本地: {e}")

        scores = self.vectors @ q
        idx = np.argsort(-scores)[:limit]
        return [
            {
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
            }
            for i in idx
        ]

    def ask(self, question: str, limit: int = 5, **_: Any) -> str:
        hits = self.search(question, limit=limit)
        if not hits:
            return "知识库为空或未检索到相关内容，请先加载 PDF 文档。"

        context = "\n\n".join(
            f"[片段{i+1} | {h['source']} | score={h['score']:.3f}]\n{h['text']}"
            for i, h in enumerate(hits)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是文档问答助手。请只根据给定检索片段回答问题；"
                    "若片段不足以回答，请明确说明信息不足。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索片段：\n{context}",
            },
        ]
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        # 去掉部分模型返回的思考标签
        answer = re.sub(r"<think>[\s\S]*?</think>", "", answer).strip()
        refs = "\n".join(
            f"- {h['source']} (score={h['score']:.3f})" for h in hits
        )
        return f"{answer}\n\n---\n📚 参考片段：\n{refs}"

    def stats(self) -> str:
        backend = "qdrant+local" if self.qdrant else "local"
        dim = int(self.vectors.shape[1]) if self.vectors is not None else 0
        return (
            f"namespace={self.namespace}, chunks={len(self.chunks)}, "
            f"dim={dim}, backend={backend}"
        )
