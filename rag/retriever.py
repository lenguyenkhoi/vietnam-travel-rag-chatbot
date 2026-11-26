"""Mô-đun truy xuất: tìm kiếm ngữ nghĩa (FAISS), BM25 và tìm kiếm lai (RRF), có lọc theo tỉnh/danh mục."""
import json
import pickle
from collections import defaultdict

import faiss
import numpy as np

from .config import RRF_K, index_path
from .embedder import Embedder
from .text import norm_key, tokenize


class Retriever:
    def __init__(self, model: str, strategy: str, embedder: Embedder | None = None):
        d = index_path(model, strategy)
        if not d.exists():
            raise FileNotFoundError(f"Chưa có chỉ mục {d}. Hãy chạy scripts/04_build_index.py --model {model} --strategy {strategy}")
        self.model_name, self.strategy = model, strategy
        self.chunks = [json.loads(l) for l in (d / "chunks.jsonl").open(encoding="utf-8")]
        self.index = faiss.read_index(str(d / "faiss.index"))
        with (d / "bm25.pkl").open("rb") as fh:
            self.bm25 = pickle.load(fh)
        self.embedder = embedder or Embedder(model)
        self.regions = sorted({c["region"] for c in self.chunks if c.get("region")}, key=len, reverse=True)
        self.categories = sorted({c["category"] for c in self.chunks if c.get("category")}, key=len, reverse=True)

    # ---------- nhận diện bộ lọc trong câu hỏi ----------
    def detect_filters(self, query: str) -> dict:
        """Tìm tên tỉnh/thành xuất hiện trong câu hỏi (so khớp không dấu)."""
        # Một câu có thể nhắc nhiều tên tỉnh, ví dụ "Bảo tàng Hồ Chí Minh thành phố Huế ở Huế":
        # chọn tên xuất hiện SAU CÙNG (thường là cụm "ở <tỉnh>"), nếu bằng nhau thì chọn tên dài hơn.
        q = " " + norm_key(query) + " "
        aliases = {"hcm": "Hồ Chí Minh", "tphcm": "Hồ Chí Minh", "sai gon": "Hồ Chí Minh", "saigon": "Hồ Chí Minh"}
        names = [(norm_key(r), r) for r in self.regions] + [(a, r) for a, r in aliases.items() if r in self.regions]
        best = None
        for key, r in names:
            pos = q.rfind(f" {key} ")
            if pos >= 0 and (best is None or (pos, len(key)) > best[0]):
                best = ((pos, len(key)), r)
        return {"region": best[1]} if best else {}

    def _allowed(self, filters: dict):
        if not filters:
            return None
        return {i for i, c in enumerate(self.chunks)
                if all(c.get(k) == v for k, v in filters.items() if v)}

    # ---------- các phương pháp tìm kiếm ----------
    def semantic(self, query: str, k: int = 50, allowed=None, qvec=None):
        qv = qvec if qvec is not None else self.embedder.encode_queries([query])
        n = len(self.chunks) if allowed else k
        scores, ids = self.index.search(qv, min(n, len(self.chunks)))
        res = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0 and (allowed is None or i in allowed)]
        return res[:k]

    def keyword(self, query: str, k: int = 50, allowed=None):
        sc = self.bm25.get_scores(tokenize(query))
        if allowed is not None:
            mask = np.full(len(sc), -np.inf)
            idx = list(allowed)
            mask[idx] = sc[idx]
            sc = mask
        top = np.argsort(-sc)[:k]
        return [(int(i), float(sc[i])) for i in top if np.isfinite(sc[i]) and sc[i] > 0]

    def hybrid(self, query: str, k: int = 50, allowed=None, qvec=None):
        fused = defaultdict(float)
        for res in (self.semantic(query, k * 2, allowed, qvec), self.keyword(query, k * 2, allowed)):
            for rank, (i, _) in enumerate(res):
                fused[i] += 1.0 / (RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]

    def search(self, query: str, k: int = 5, method: str = "hybrid", use_filters: bool = True,
               dedupe_docs: bool = False, qvec=None, max_per_doc: int | None = None, filter_mode: str = "soft",
               reranker=None, n_candidates: int = 20):
        """Trả về danh sách chunk (dict) kèm điểm.

        dedupe_docs=True : mỗi địa điểm chỉ 1 chunk tốt nhất (dùng khi đánh giá ở mức địa điểm).
        max_per_doc=n    : mỗi địa điểm tối đa n chunk, giúp ngữ cảnh đa dạng hơn cho câu hỏi gợi ý.
        filter_mode      : "hard" = chỉ giữ địa điểm thuộc tỉnh nhắc trong câu hỏi;
                           "soft" = gộp (RRF) kết quả có lọc và không lọc, nên địa điểm đúng tỉnh được ưu tiên
                                    nhưng không bị loại khi câu hỏi nhắc tỉnh khác (ví dụ "từ TP.HCM đi Phú Quốc").
        reranker         : đối tượng Reranker; nếu có, lấy n_candidates đoạn tốt nhất (CHƯA giới hạn số đoạn
                           mỗi địa điểm) rồi xếp hạng lại bằng cross-encoder, sau đó mới áp max_per_doc và k.
        """
        if dedupe_docs:
            max_per_doc = 1
        filters = self.detect_filters(query) if use_filters else {}
        allowed = self._allowed(filters)
        if allowed is not None and not allowed:
            allowed = None
        pool = k * 6 if max_per_doc else k
        if reranker is not None:
            pool = max(pool, n_candidates)

        def run(allow):
            if method == "semantic":
                return self.semantic(query, pool, allow, qvec)
            if method == "bm25":
                return self.keyword(query, pool, allow)
            if method == "hybrid":
                return self.hybrid(query, pool, allow, qvec)
            raise ValueError(method)

        if allowed is not None and filter_mode == "soft":
            fused = defaultdict(float)
            for lst in (run(allowed), run(None)):
                for rank, (i, _) in enumerate(lst):
                    fused[i] += 1.0 / (RRF_K + rank + 1)
            res = sorted(fused.items(), key=lambda x: -x[1])[:pool]
        else:
            res = run(allowed)
        cands = [{**self.chunks[i], "score": s} for i, s in res]
        if reranker is not None:
            cands = reranker.rerank(query, cands[:n_candidates])
        out, per_doc = [], {}
        for c in cands:
            if max_per_doc and per_doc.get(c["doc_id"], 0) >= max_per_doc:
                continue
            per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
            out.append(c)
            if len(out) >= k:
                break
        return out, filters
