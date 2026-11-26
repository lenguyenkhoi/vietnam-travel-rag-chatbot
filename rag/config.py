"""Cấu hình chung của hệ thống RAG."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
INDEX_DIR = PROJECT / "indexes"
EVAL_DIR = PROJECT / "eval"
REPORT_DIR = PROJECT / "reports"
PLACES_FILE = DATA_DIR / "places.jsonl"

# Mô hình embedding dùng để so sánh (đều miễn phí trên Hugging Face)
EMBED_MODELS = {
    "e5-base": {
        "hf": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_len": 512,
    },
    "bge-m3": {
        "hf": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "max_len": 512,
    },
    "vn-sbert": {
        "hf": "keepitreal/vietnamese-sbert",
        "query_prefix": "",
        "passage_prefix": "",
        "max_len": 256,
        "word_segment": True,  # mô hình PhoBERT cần tách từ tiếng Việt (pyvi)
    },
}
DEFAULT_EMBED = "e5-base"

# Mô hình xếp hạng lại (cross-encoder). Lưu ở thư mục MODEL_CACHE (máy thực tập: ổ E vì ổ C gần đầy).
RERANK_MODELS = {
    "bge-reranker-v2-m3": {"hf": "BAAI/bge-reranker-v2-m3", "max_len": 512},
}
DEFAULT_RERANK = "bge-reranker-v2-m3"
RERANK_CANDIDATES = 10  # số đoạn từ tìm kiếm lai đưa vào xếp hạng lại (10: ~20s/câu trên CPU, 20: ~55s, chất lượng như nhau khi thử)
# Ngưỡng liên quan: nếu đoạn tốt nhất sau xếp hạng lại có điểm < ngưỡng thì từ chối luôn, không gọi LLM.
# Chọn từ reports/06b_threshold.csv (18/11/2026): câu ngoài phạm vi có điểm cao nhất 0,252, câu trong phạm vi
# thấp nhất 0,734 -> mọi ngưỡng trong (0,26; 0,73) đều tách đúng 100%; chọn 0,5 ở giữa để dư biên hai phía.
RERANK_MIN_SCORE = 0.5
MODEL_CACHE = os.environ.get("RAG_MODEL_CACHE") or ("E:/huggingface" if Path("E:/").exists() else None)
DEFAULT_STRATEGY = "section"

# Mô hình ngôn ngữ chạy cục bộ qua Ollama
LLM_MODELS = ["qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b", "gemma2:2b"]
DEFAULT_LLM = "qwen2.5:3b"
OLLAMA_HOST = "http://localhost:11434"

# Truy xuất
TOP_K = 5
RRF_K = 60  # hằng số của Reciprocal Rank Fusion
MAX_CONTEXT_CHARS = 6000  # giới hạn ngữ cảnh đưa vào LLM (máy CPU)


def index_path(model: str, strategy: str) -> Path:
    return INDEX_DIR / f"{model}__{strategy}"
