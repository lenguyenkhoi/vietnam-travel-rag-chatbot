"""Hỏi đáp nhanh trên terminal.

Chạy:  python ask.py "Bảo tàng Áo Dài mở cửa mấy giờ?"
       python ask.py                      (chế độ hỏi liên tục, gõ 'q' để thoát)
Tùy chọn: --llm llama3.2:3b --method bm25 --k 5 --no-rag
"""
import argparse
import sys

from rag.config import DEFAULT_EMBED, DEFAULT_LLM, DEFAULT_STRATEGY, TOP_K
from rag.pipeline import RAGPipeline


def show(res):
    print("\n" + res["answer"] + "\n")
    for s in res["sources"]:
        print(f"  [{s['so']}] {s['title']} | {s['region'] or ''} | {s['category'] or ''} | điểm {s['score']}")
    extra = f" | lọc tỉnh: {res['filters']['region']}" if res["filters"].get("region") else ""
    print(f"  (tìm kiếm {res['t_retrieve']:.2f}s, sinh câu trả lời {res['t_generate']:.1f}s{extra})\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--method", default="hybrid", choices=["hybrid", "semantic", "bm25"])
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--no-rag", action="store_true")
    a = ap.parse_args()
    pipe = RAGPipeline(a.embed, a.strategy, a.llm)
    ask = lambda q: show(pipe.answer(q, k=a.k, method=a.method, use_rag=not a.no_rag))  # noqa: E731
    if a.question:
        ask(a.question)
        return
    while True:
        q = input("Câu hỏi> ").strip()
        if q.lower() in ("q", "quit", "exit", ""):
            break
        ask(q)


if __name__ == "__main__":
    main()
