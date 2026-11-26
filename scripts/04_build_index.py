"""Bước 4: Tạo embedding và xây chỉ mục FAISS + BM25.

Chạy:  python scripts/04_build_index.py --model e5-base --strategy section
       python scripts/04_build_index.py --model bge-m3 --strategy section
Đầu ra: indexes/<model>__<strategy>/{chunks.jsonl, faiss.index, bm25.pkl, meta.json}
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.chunking import chunk_place  # noqa: E402
from rag.config import DEFAULT_EMBED, DEFAULT_STRATEGY, EMBED_MODELS, PLACES_FILE, index_path  # noqa: E402
from rag.embedder import Embedder  # noqa: E402
from rag.text import tokenize  # noqa: E402


def encode_resumable(emb, texts, parts_dir: Path, batch: int, part_size: int):
    """Tạo embedding theo từng phần và lưu ra đĩa; nếu bị ngắt (tắt máy), lần sau chạy tiếp từ phần còn thiếu.

    Thư mục _parts lưu dấu vân tay của danh sách đoạn: nếu dữ liệu hoặc part_size thay đổi thì làm lại từ đầu.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha1(("\n".join(texts) + f"|{part_size}|{emb.name}").encode("utf-8")).hexdigest()
    fp_file = parts_dir / "fingerprint.txt"
    if not fp_file.exists() or fp_file.read_text() != fp:
        for f in parts_dir.glob("part_*.npy"):
            f.unlink()
        fp_file.write_text(fp)
    n_parts = math.ceil(len(texts) / part_size)
    out = []
    for i in range(n_parts):
        f = parts_dir / f"part_{i:04d}.npy"
        if f.exists():
            out.append(np.load(f))
            print(f"Phần {i + 1}/{n_parts}: đã có, bỏ qua", flush=True)
            continue
        t0 = time.time()
        v = emb.encode_passages(texts[i * part_size:(i + 1) * part_size], batch_size=batch, show_progress=False)
        tmp = parts_dir / f"part_{i:04d}.tmp.npy"
        np.save(tmp, v)
        tmp.replace(f)  # ghi xong mới đổi tên, tránh file hỏng khi bị ngắt giữa chừng
        out.append(v)
        print(f"Phần {i + 1}/{n_parts}: xong {len(v)} đoạn trong {time.time() - t0:.0f}s", flush=True)
    return np.vstack(out).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_EMBED, choices=list(EMBED_MODELS))
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=["doc", "fixed", "section"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--part-size", type=int, default=1000, help="số đoạn mỗi phần lưu tạm (để chạy tiếp khi bị ngắt)")
    args = ap.parse_args()

    places = [json.loads(l) for l in PLACES_FILE.open(encoding="utf-8")]
    chunks = [c for p in places for c in chunk_place(p, args.strategy)]
    print(f"{len(places)} địa điểm -> {len(chunks)} đoạn ({args.strategy})")

    out = index_path(args.model, args.strategy)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    t_bm25 = time.time() - t0

    t0 = time.time()
    emb = Embedder(args.model)
    t_load = time.time() - t0
    t0 = time.time()
    vecs = encode_resumable(emb, [c["text"] for c in chunks], out / "_parts", args.batch, args.part_size)
    t_embed = time.time() - t0

    index = faiss.IndexFlatIP(vecs.shape[1])  # vector đã chuẩn hóa -> tích vô hướng = cosine
    index.add(vecs)

    faiss.write_index(index, str(out / "faiss.index"))
    with (out / "bm25.pkl").open("wb") as fh:
        pickle.dump(bm25, fh)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = {
        "model": args.model, "hf": EMBED_MODELS[args.model]["hf"], "strategy": args.strategy,
        "n_places": len(places), "n_chunks": len(chunks), "dim": int(vecs.shape[1]),
        "seconds_bm25": round(t_bm25, 1), "seconds_load_model": round(t_load, 1), "seconds_embed": round(t_embed, 1),
        "chunks_per_second": round(len(chunks) / max(t_embed, 1e-6), 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
