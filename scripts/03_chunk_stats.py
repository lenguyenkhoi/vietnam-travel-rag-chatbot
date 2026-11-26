"""Bước 3: Chia đoạn theo 3 chiến lược và thống kê (dùng cho mục 3.2 của báo cáo).

Chạy:  python scripts/03_chunk_stats.py
Đầu ra: reports/03_chunk_stats.csv, data/chunks_<strategy>.jsonl
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.chunking import chunk_place  # noqa: E402
from rag.config import DATA_DIR, PLACES_FILE, REPORT_DIR  # noqa: E402

places = [json.loads(l) for l in PLACES_FILE.open(encoding="utf-8")]
rows = []
for strategy in ("doc", "fixed", "section"):
    chunks = [c for p in places for c in chunk_place(p, strategy)]
    with (DATA_DIR / f"chunks_{strategy}.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    lens = pd.Series([len(c["text"]) for c in chunks])
    rows.append({
        "chien_luoc": strategy, "so_dia_diem": len(places), "so_doan": len(chunks),
        "doan_moi_dia_diem": round(len(chunks) / len(places), 2),
        "do_dai_tb": int(lens.mean()), "trung_vi": int(lens.median()), "p90": int(lens.quantile(.9)), "max": int(lens.max()),
    })
df = pd.DataFrame(rows)
REPORT_DIR.mkdir(exist_ok=True)
df.to_csv(REPORT_DIR / "03_chunk_stats.csv", index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
