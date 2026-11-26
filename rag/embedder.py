"""Bao bọc mô hình embedding (sentence-transformers) với tiền tố truy vấn/đoạn văn phù hợp."""
from functools import lru_cache

import numpy as np

from .config import EMBED_MODELS


@lru_cache(maxsize=1)
def _segmenter():
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize


class Embedder:
    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        if name not in EMBED_MODELS:
            raise ValueError(f"Không có mô hình {name}. Chọn một trong {list(EMBED_MODELS)}")
        self.name = name
        self.cfg = EMBED_MODELS[name]
        self.model = SentenceTransformer(self.cfg["hf"], device=device)
        self.model.max_seq_length = self.cfg["max_len"]

    def _prep(self, texts, prefix):
        if self.cfg.get("word_segment"):
            seg = _segmenter()
            texts = [seg(t) for t in texts]
        return [prefix + t for t in texts]

    def encode_passages(self, texts, batch_size=16, show_progress=True) -> np.ndarray:
        x = self.model.encode(self._prep(texts, self.cfg["passage_prefix"]), batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=show_progress, convert_to_numpy=True)
        return x.astype("float32")

    def encode_queries(self, texts, batch_size=32) -> np.ndarray:
        x = self.model.encode(self._prep(texts, self.cfg["query_prefix"]), batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return x.astype("float32")
