"""Bước 6b: Đánh giá re-ranker và chọn ngưỡng liên quan (mục 3.3 mở rộng và 4.1 của báo cáo).

1. So sánh tìm kiếm lai (không lọc) TRƯỚC và SAU khi xếp hạng lại bằng cross-encoder (Recall@k, MRR@10)
   trên các câu trong phạm vi.
2. Ghi điểm re-ranker cao nhất của MỌI câu (kể cả câu ngoài phạm vi) để chọn NGƯỠNG LIÊN QUAN:
   nếu đoạn tốt nhất có điểm < ngưỡng thì chatbot từ chối luôn, không gọi LLM.
   Bảng ngưỡng: tỷ lệ từ chối NHẦM câu trong phạm vi và tỷ lệ từ chối ĐÚNG câu ngoài phạm vi.

Chạy:  python scripts/06b_eval_rerank.py         (~20 giây/câu trên CPU, ~1 giờ cho 192 câu; chạy tiếp được)
Đầu ra: reports/06b_rerank_per_question.csv, reports/06b_rerank_results.csv, reports/06b_threshold.csv,
        reports/figures/06b_threshold.png
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import DEFAULT_EMBED, DEFAULT_RERANK, DEFAULT_STRATEGY, EVAL_DIR, RERANK_CANDIDATES, REPORT_DIR  # noqa: E402
from rag.reranker import Reranker  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

KS = (1, 3, 5, 10)


def doc_metrics(ranked_docs, relevant):
    rel = set(relevant)
    first = next((i for i, d in enumerate(ranked_docs, 1) if d in rel), None)
    m = {f"Recall@{k}": float(first is not None and first <= k) for k in KS}
    m["MRR@10"] = 1.0 / first if first and first <= 10 else 0.0
    return m


def dedupe(chunks):
    seen, out = set(), []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            out.append(c["doc_id"])
    return out


def main():
    ts = pd.read_csv(EVAL_DIR / "testset.csv", encoding="utf-8-sig", dtype=str).fillna("")
    out_csv = REPORT_DIR / "06b_rerank_per_question.csv"
    rows = pd.read_csv(out_csv, encoding="utf-8-sig", dtype={"qid": str}).to_dict("records") if out_csv.exists() else []
    done = {r["qid"] for r in rows}
    print(f"{len(ts)} câu, đã có {len(done)} câu từ lần chạy trước")

    r = Retriever(DEFAULT_EMBED, DEFAULT_STRATEGY)
    rr = Reranker(DEFAULT_RERANK)
    for _, q in ts.iterrows():
        if q["qid"] in done:
            continue
        t0 = time.time()
        # ứng viên: tìm kiếm lai không lọc, chưa giới hạn số đoạn mỗi địa điểm (giống hệ thống thật)
        cands, _ = r.search(q["cau_hoi"], k=RERANK_CANDIDATES, method="hybrid", use_filters=False)
        reranked = rr.rerank(q["cau_hoi"], cands)
        dt = time.time() - t0
        row = {"qid": q["qid"], "loai": q["loai"], "dien_dat_lai": q.get("dien_dat_lai", ""),
               "diem_cao_nhat": round(reranked[0]["score"], 4) if reranked else 0.0, "giay": round(dt, 2),
               "top1_truoc": cands[0]["title"] if cands else "", "top1_sau": reranked[0]["title"] if reranked else ""}
        if q["loai"] != "ngoai_pham_vi":
            relevant = q["doc_ids_dung"].split(";")
            # mức địa điểm; chỉ xếp hạng lại trong RERANK_CANDIDATES đoạn, phần sau giữ thứ tự tìm kiếm lai
            for tag, lst in (("truoc", cands), ("sau", reranked)):
                for k, v in doc_metrics(dedupe(lst), relevant).items():
                    row[f"{k}_{tag}"] = v
        rows.append(row)
        print(f"{q['qid']} {q['loai']:13s} {dt:5.1f}s max={row['diem_cao_nhat']:.3f} | {row['top1_sau'][:50]}", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")  # lưu dần

    df = pd.DataFrame(rows)
    ins = df[df["loai"] != "ngoai_pham_vi"]
    oos = df[df["loai"] == "ngoai_pham_vi"]

    # 1. trước / sau xếp hạng lại
    res = []
    for grp_name, g in [("tat_ca", ins)] + [(f"loai={k}", v) for k, v in ins.groupby("loai")] + \
                       [(f"dien_dat_lai={k}", v) for k, v in ins.groupby("dien_dat_lai")]:
        row = {"nhom": grp_name, "so_cau": len(g)}
        for tag in ("truoc", "sau"):
            for k in ("Recall@1", "Recall@5", "MRR@10"):
                row[f"{k}_{tag}"] = round(g[f"{k}_{tag}"].mean(), 4)
        res.append(row)
    res = pd.DataFrame(res)
    res.to_csv(REPORT_DIR / "06b_rerank_results.csv", index=False, encoding="utf-8-sig")
    print("\n", res.to_string(index=False))

    # 2. bảng ngưỡng
    th = []
    for t in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        th.append({"nguong": t,
                   "tu_choi_nham_trong_pham_vi": round((ins["diem_cao_nhat"] < t).mean(), 4),
                   "tu_choi_dung_ngoai_pham_vi": round((oos["diem_cao_nhat"] < t).mean(), 4),
                   "so_cau_trong_pham_vi_bi_chan": int((ins["diem_cao_nhat"] < t).sum())})
    th = pd.DataFrame(th)
    th.to_csv(REPORT_DIR / "06b_threshold.csv", index=False, encoding="utf-8-sig")
    print("\n", th.to_string(index=False))
    print("\nĐiểm cao nhất, câu ngoài phạm vi:", sorted(oos["diem_cao_nhat"].round(3).tolist()))
    print("10 câu trong phạm vi có điểm thấp nhất:")
    print(ins.nsmallest(10, "diem_cao_nhat")[["qid", "loai", "diem_cao_nhat", "top1_sau"]].to_string(index=False))
    print(f"Thời gian xếp hạng lại trung bình: {df['giay'].mean():.1f}s/câu")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bins = [i / 20 for i in range(21)]
        ax.hist(ins["diem_cao_nhat"], bins=bins, alpha=0.7, label=f"Trong phạm vi ({len(ins)} câu)", color="#3b6ea5")
        ax.hist(oos["diem_cao_nhat"], bins=bins, alpha=0.8, label=f"Ngoài phạm vi ({len(oos)} câu)", color="#c0504d")
        ax.set_xlabel("Điểm re-ranker của đoạn tốt nhất")
        ax.set_ylabel("Số câu hỏi")
        ax.set_title("Phân bố điểm liên quan: câu trong và ngoài phạm vi")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        (REPORT_DIR / "figures").mkdir(exist_ok=True)
        plt.savefig(REPORT_DIR / "figures" / "06b_threshold.png", dpi=150)
    except Exception as e:
        print("Không vẽ được biểu đồ:", e)


if __name__ == "__main__":
    main()
