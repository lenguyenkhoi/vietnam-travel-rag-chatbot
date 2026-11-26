"""Bước 8: Vẽ biểu đồ cho chương 4 của báo cáo (nhãn tiếng Việt, bảng màu đã kiểm tra cho người mù màu).

Chạy:  python scripts/08_report_figures.py
Đầu ra: reports/figures/4_*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import REPORT_DIR  # noqa: E402

FIG = REPORT_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"  # 2 màu đầu của bảng màu phân loại (đã kiểm tra CVD)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS})


COMMA = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))  # dấu thập phân kiểu Việt Nam


def style(ax, grid_axis="x"):
    ax.xaxis.set_major_formatter(COMMA)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_retrieval():
    """Hình 4.1: MRR@10 theo phương pháp, câu chép nguyên văn và câu diễn đạt lại (biểu đồ chấm)."""
    d = pd.read_csv(REPORT_DIR / "06_retrieval_by_rewrite.csv")
    names = {"bm25": "BM25 (từ khóa)", "semantic": "Ngữ nghĩa (e5-base)", "hybrid_khong_loc": "Lai, không lọc tỉnh",
             "hybrid_loc_cung": "Lai, lọc tỉnh cứng", "hybrid_loc_mem": "Lai, lọc tỉnh mềm"}
    order = list(names)[::-1]
    p = d.pivot(index="phuong_phap", columns="dien_dat_lai", values="MRR@10").loc[order]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = range(len(order))
    ax.hlines(y, p["co"], p["khong"], color=AXIS, linewidth=2, zorder=1)
    ax.scatter(p["khong"], y, s=60, color=BLUE, zorder=3, label="Câu chép nguyên văn dữ liệu (63 câu)",
               edgecolor="white", linewidth=1.5)
    ax.scatter(p["co"], y, s=60, color=ORANGE, zorder=3, label="Câu diễn đạt lại (114 câu)",
               edgecolor="white", linewidth=1.5)
    for i, v in enumerate(p["co"]):
        ax.text(v - 0.004, i, f"{v:.3f}".replace(".", ","), va="center", ha="right", fontsize=8, color=INK2)
    ax.set_yticks(list(y), [names[k] for k in order])
    ax.set_xlim(0.85, 1.005)
    ax.set_xlabel("MRR@10 (càng gần 1 càng tốt)")
    style(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "4_1_retrieval_rewrite.png", dpi=200)
    plt.close()


def fig_threshold():
    """Hình 4.2: phân bố điểm re-ranker cao nhất của câu trong và ngoài phạm vi."""
    d = pd.read_csv(REPORT_DIR / "06b_rerank_per_question.csv")
    ins, oos = d[d["loai"] != "ngoai_pham_vi"], d[d["loai"] == "ngoai_pham_vi"]
    bins = [i / 20 for i in range(21)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(ins["diem_cao_nhat"], bins=bins, color=BLUE, label=f"Câu trong phạm vi ({len(ins)} câu)",
            edgecolor="white", linewidth=1)
    ax.hist(oos["diem_cao_nhat"], bins=bins, color=ORANGE, label=f"Câu ngoài phạm vi ({len(oos)} câu)",
            edgecolor="white", linewidth=1)
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.515, ax.get_ylim()[1] * 0.55, "ngưỡng 0,5", color=INK2, fontsize=8)
    ax.set_xlabel("Điểm liên quan của đoạn tốt nhất sau xếp hạng lại")
    ax.set_ylabel("Số câu hỏi")
    style(ax, "y")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.04, 0.98))
    plt.tight_layout()
    plt.savefig(FIG / "4_2_threshold.png", dpi=200)
    plt.close()


def grouped_bars(cats, s1, s2, l1, l2, xlabel, fname, figsize=(7, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    y = list(range(len(cats)))[::-1]
    h = 0.36
    b1 = ax.barh([v + h / 2 + 0.01 for v in y], s1, height=h, color=BLUE, label=l1)
    b2 = ax.barh([v - h / 2 - 0.01 for v in y], s2, height=h, color=ORANGE, label=l2)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%",
                    va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 105)
    ax.set_xlabel(xlabel)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=200)
    plt.close()


def fig_ablation():
    """Hình 4.3: tỷ lệ trả lời đúng của 4 cấu hình (bản rút gọn), qwen và llama."""
    d = pd.read_csv(REPORT_DIR / "07_ablation_rut_gon.csv")
    d = d[d["che_do"] == "co_rag"]
    labels = {"A: v1": "A. Prompt v1", "B: v2": "B. Prompt v2", "C: v2+rerank+nguong": "C. Prompt v2 + re-ranker + ngưỡng",
              "D: v1+rerank+nguong": "D. Prompt v1 + re-ranker + ngưỡng"}
    cats = list(labels)
    q = [100 * d[(d.cau_hinh == c) & (d.llm == "qwen2.5:3b")]["dung"].iloc[0] for c in cats]
    ll = [100 * d[(d.cau_hinh == c) & (d.llm == "llama3.2:3b")]["dung"].iloc[0] for c in cats]
    grouped_bars([labels[c] for c in cats], q, ll, "qwen2.5:3b", "llama3.2:3b",
                 "Tỷ lệ trả lời đúng câu trong phạm vi (%)", "4_3_ablation.png", figsize=(7, 3.4))


def fig_final():
    """Hình 4.4: có RAG và không RAG, 3 mô hình (bản đầy đủ)."""
    d = pd.read_csv(REPORT_DIR / "07_final_summary.csv").set_index("llm").loc[["qwen2.5:3b", "llama3.2:3b", "gemma2:2b"]]
    grouped_bars(list(d.index), list(100 * d["dung_co_rag_dieu_chinh"]), list(100 * d["dung_khong_rag_tu_dong"]),
                 "Có RAG (sau chấm tay)", "Không RAG", "Tỷ lệ trả lời đúng câu trong phạm vi (%)", "4_4_rag_vs_norag.png",
                 figsize=(7, 2.8))


if __name__ == "__main__":
    fig_retrieval()
    fig_threshold()
    fig_ablation()
    fig_final()
    print("Đã vẽ:", sorted(p.name for p in FIG.glob("4_*.png")))
