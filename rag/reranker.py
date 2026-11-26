"""Mô-đun xếp hạng lại (re-ranking) bằng cross-encoder.

Tìm kiếm lai (embedding + BM25) chấm điểm câu hỏi và từng đoạn một cách ĐỘC LẬP, nên khi câu hỏi nhắc
tên địa điểm, mọi đoạn của địa điểm đó đều được điểm gần như nhau. Cross-encoder đọc câu hỏi và đoạn văn
CÙNG LÚC nên phân biệt được đoạn nào thực sự chứa chi tiết được hỏi (ví dụ "năm tôn tạo, kinh phí").
Chi phí: chậm hơn nhiều, nên chỉ chạy trên RERANK_CANDIDATES đoạn tốt nhất của tìm kiếm lai.
"""
from .config import MODEL_CACHE, RERANK_MODELS


class Reranker:
    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        cfg = RERANK_MODELS[name]
        self.name = name
        self.model = CrossEncoder(cfg["hf"], device=device, max_length=cfg["max_len"], cache_folder=MODEL_CACHE)

    def rerank(self, query: str, chunks: list[dict], batch_size: int = 8) -> list[dict]:
        if not chunks:
            return chunks
        scores = self.model.predict([(query, c["text"]) for c in chunks], batch_size=batch_size,
                                    show_progress_bar=False)
        out = [{**c, "score_truoc": c.get("score"), "score": float(s)} for c, s in zip(chunks, scores)]
        return sorted(out, key=lambda c: -c["score"])
