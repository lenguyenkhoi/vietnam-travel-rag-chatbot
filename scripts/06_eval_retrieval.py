"""Bước 6: Đánh giá mô-đun truy xuất (mục 4.1 của báo cáo).

Với mỗi chỉ mục (embedding x cách chia đoạn) và mỗi phương pháp (BM25 / ngữ nghĩa / lai / lai không lọc),
tính Recall@1,3,5,10, MRR@10, nDCG@10 ở mức địa điểm (mỗi địa điểm chỉ tính 1 lần).

Chạy:  python scripts/06_eval_retrieval.py                 (mọi chỉ mục đã xây)
       python scripts/06_eval_retrieval.py --index e5-base__section
Đầu ra: reports/06_retrieval_results.csv, reports/06_retrieval_by_type.csv, reports/figures/06_retrieval.png
"""
import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import EVAL_DIR, INDEX_DIR, REPORT_DIR  # noqa: E402
from rag.embedder import Embedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

KS = (1, 3, 5, 10)
# (nhãn, phương pháp, lọc tỉnh: False | "hard" | "soft")
METHODS = [("bm25", "bm25", "soft"), ("semantic", "semantic", "soft"), ("hybrid_khong_loc", "hybrid", False),
           ("hybrid_loc_cung", "hybrid", "hard"), ("hybrid_loc_mem", "hybrid", "soft")]


def load_testset():
    f = EVAL_DIR / "testset.csv"
    if not f.exists():
        f = EVAL_DIR / "testset_draft.csv"
        print("Chưa có eval/testset.csv, dùng bản nháp", f.name)
    df = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
    return df[df["loai"] != "ngoai_pham_vi"].reset_index(drop=True)


def metrics(ranked, relevant):
    rel = set(relevant)
    first = next((i for i, d in enumerate(ranked, 1) if d in rel), None)
    m = {f"Recall@{k}": float(first is not None and first <= k) for k in KS}
    m["MRR@10"] = 1.0 / first if first and first <= 10 else 0.0
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:10], 1) if d in rel)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), 10) + 1))
    m["nDCG@10"] = dcg / idcg if idcg else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="*", help="tên thư mục chỉ mục, ví dụ e5-base__section")
    args = ap.parse_args()
    names = args.index or sorted(d.name for d in INDEX_DIR.iterdir() if (d / "faiss.index").exists())
    ts = load_testset()
    print(f"{len(ts)} câu hỏi, {len(names)} chỉ mục: {names}")

    rows, per_q = [], []
    embedders = {}
    for name in names:
        model, strategy = name.split("__")
        emb = embedders.setdefault(model, Embedder(model))
        r = Retriever(model, strategy, embedder=emb)
        t0 = time.time()
        qvecs = emb.encode_queries(ts["cau_hoi"].tolist())
        t_enc = (time.time() - t0) / len(ts)
        for label, method, fmode in METHODS:
            t0 = time.time()
            for i, q in ts.iterrows():
                res, _ = r.search(q["cau_hoi"], k=10, method=method, use_filters=bool(fmode),
                                  filter_mode=fmode or "soft", dedupe_docs=True, qvec=qvecs[i:i + 1])
                ranked = [c["doc_id"] for c in res]
                m = metrics(ranked, q["doc_ids_dung"].split(";"))
                per_q.append({"chi_muc": name, "phuong_phap": label, "qid": q["qid"], "loai": q["loai"],
                              "dien_dat_lai": q.get("dien_dat_lai", ""), **m,
                              "top1": res[0]["title"] if res else ""})
            dt = (time.time() - t0) / len(ts) + (0 if method == "bm25" else t_enc)
            sub = pd.DataFrame([p for p in per_q if p["chi_muc"] == name and p["phuong_phap"] == label])
            row = {"chi_muc": name, "embedding": model, "chia_doan": strategy, "phuong_phap": label,
                   **{c: round(sub[c].mean(), 4) for c in sub.columns if "@" in c}, "giay_moi_cau": round(dt, 3)}
            rows.append(row)
            print({k: row[k] for k in ("chi_muc", "phuong_phap", "Recall@1", "Recall@5", "MRR@10", "giay_moi_cau")})

    REPORT_DIR.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(REPORT_DIR / "06_retrieval_results.csv", index=False, encoding="utf-8-sig")
    pq = pd.DataFrame(per_q)
    pq.to_csv(REPORT_DIR / "06_retrieval_per_question.csv", index=False, encoding="utf-8-sig")
    bt = pq.groupby(["chi_muc", "phuong_phap", "loai"])[["Recall@1", "Recall@5", "MRR@10"]].mean().round(4).reset_index()
    bt.to_csv(REPORT_DIR / "06_retrieval_by_type.csv", index=False, encoding="utf-8-sig")
    if pq["dien_dat_lai"].astype(bool).any():  # so sánh câu chép nguyên văn và câu diễn đạt lại
        rw = pq.groupby(["chi_muc", "phuong_phap", "dien_dat_lai"])[["Recall@1", "Recall@5", "MRR@10"]]
        rw = rw.mean().round(4).reset_index()
        rw.to_csv(REPORT_DIR / "06_retrieval_by_rewrite.csv", index=False, encoding="utf-8-sig")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = REPORT_DIR / "figures"
        fig_dir.mkdir(exist_ok=True)
        piv = res.pivot(index="chi_muc", columns="phuong_phap", values="MRR@10")
        ax = piv.plot(kind="bar", figsize=(9, 4.5), rot=0)
        ax.set_ylabel("MRR@10")
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.set_title("MRR@10 theo chỉ mục và phương pháp tìm kiếm")
        ax.legend(title="Phương pháp", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "06_retrieval.png", dpi=150)
    except Exception as e:
        print("Không vẽ được biểu đồ:", e)
    print("\n", res.to_string(index=False))


if __name__ == "__main__":
    main()
