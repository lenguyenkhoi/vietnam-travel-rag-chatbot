"""Ghép truy xuất + (xếp hạng lại) + sinh câu trả lời thành một luồng hỏi đáp hoàn chỉnh."""
import json
import time

from .config import (DEFAULT_EMBED, DEFAULT_LLM, DEFAULT_STRATEGY, PLACES_FILE, RERANK_CANDIDATES,
                     RERANK_MIN_SCORE, TOP_K)
from .generator import DEFAULT_PROMPT, REFUSAL, Generator
from .retriever import Retriever


class RAGPipeline:
    def __init__(self, embed_model=DEFAULT_EMBED, strategy=DEFAULT_STRATEGY, llm=DEFAULT_LLM,
                 rerank: str | None = None, prompt: str = DEFAULT_PROMPT, min_score: float | None = RERANK_MIN_SCORE):
        """rerank: tên mô hình trong RERANK_MODELS (ví dụ "bge-reranker-v2-m3") hoặc None để tắt.
        min_score: ngưỡng liên quan (chỉ dùng khi có re-ranker); None để tắt ngưỡng."""
        self.min_score = min_score
        self.retriever = Retriever(embed_model, strategy)
        self.generator = Generator(llm, prompt=prompt)
        self.reranker = None
        if rerank:
            from .reranker import Reranker
            self.reranker = Reranker(rerank)
        self.places = {}
        for line in PLACES_FILE.open(encoding="utf-8"):
            p = json.loads(line)
            self.places[p["doc_id"]] = p

    def set_llm(self, llm: str, prompt: str | None = None):
        prompt = prompt or self.generator.prompt
        if llm != self.generator.model or prompt != self.generator.prompt:
            self.generator = Generator(llm, prompt=prompt)

    def retrieve(self, question, k=TOP_K, method="hybrid", use_filters=False, max_per_doc=2, use_rerank=True):
        t0 = time.time()
        chunks, filters = self.retriever.search(
            question, k=k, method=method, use_filters=use_filters, max_per_doc=max_per_doc,
            reranker=self.reranker if use_rerank else None, n_candidates=RERANK_CANDIDATES)
        return chunks, filters, time.time() - t0

    def sources(self, chunks):
        """Danh sách nguồn theo đúng thứ tự [1], [2]... trong ngữ cảnh."""
        out = []
        for i, c in enumerate(chunks, 1):
            p = self.places.get(c["doc_id"], {})
            out.append({
                "so": i, "doc_id": c["doc_id"], "title": c["title"], "region": c.get("region"),
                "category": c.get("category"), "address": p.get("address"), "map_url": p.get("map_url"),
                "source_url": p.get("source_url"), "lat": p.get("lat"), "lng": p.get("lng"),
                "score": round(c["score"], 4), "text": c["text"],
            })
        return out

    def below_threshold(self, chunks, use_rerank=True) -> bool:
        """True nếu đã xếp hạng lại và đoạn tốt nhất vẫn kém liên quan (điểm < ngưỡng) -> nên từ chối."""
        if not (self.reranker and use_rerank and self.min_score is not None):
            return False
        return not chunks or chunks[0]["score"] < self.min_score

    def answer(self, question, k=TOP_K, method="hybrid", use_filters=False, use_rag=True, use_rerank=True):
        if not use_rag:
            t0 = time.time()
            ans = self.generator.generate(question, None)
            return {"answer": ans, "sources": [], "filters": {}, "t_retrieve": 0.0, "t_generate": time.time() - t0,
                    "chan_boi_nguong": False}
        chunks, filters, t_ret = self.retrieve(question, k, method, use_filters, use_rerank=use_rerank)
        if self.below_threshold(chunks, use_rerank):
            return {"answer": REFUSAL, "sources": [], "filters": filters, "t_retrieve": t_ret, "t_generate": 0.0,
                    "chan_boi_nguong": True}
        t0 = time.time()
        ans = self.generator.generate(question, chunks)
        return {"answer": ans, "sources": self.sources(chunks), "filters": filters,
                "t_retrieve": t_ret, "t_generate": time.time() - t0, "chan_boi_nguong": False}
