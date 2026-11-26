"""Bước 9: Vẽ sơ đồ cho báo cáo (kiến trúc hệ thống, quy trình dữ liệu tại công ty).

Chạy:  python scripts/09_diagrams.py
Đầu ra: reports/figures/2_1_kien_truc.png, reports/figures/1_1_quy_trinh.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import REPORT_DIR  # noqa: E402

FIG = REPORT_DIR / "figures"
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, accent=None, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=(accent + "1f") if accent else "white",
                                edgecolor=accent or EDGE, linewidth=1.6 if accent else 1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)


def arrow(ax, p1, p2, label=None, lpos=None, color=EDGE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.1,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(*(lpos or ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12)), label, ha="center", va="bottom",
                fontsize=8, color=INK2)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    # luồng chuẩn bị
    ax.text(0.1, 4.35, "LUỒNG CHUẨN BỊ (chạy một lần)", fontsize=9.5, fontweight="bold", color=INK)
    W, H, y1 = 1.7, 0.8, 3.6
    xs = [1.0, 3.0, 5.0, 7.0, 9.0]
    labels = ["Kho JSON Localis\n(8.840 file)", "Làm sạch,\nloại trùng\n(3.293 bản ghi)", "Chia đoạn theo mục\n(16.472 đoạn)",
              "Tạo embedding\n(multilingual-e5-base)", "Chỉ mục\nFAISS + BM25"]
    for x, t in zip(xs, labels):
        box(ax, x, y1, W, H, t, accent=BLUE if x == 9.0 else None)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + W / 2, y1), (b - W / 2, y1))
    # luồng hỏi đáp
    ax.text(0.1, 2.55, "LUỒNG HỎI ĐÁP (mỗi câu hỏi)", fontsize=9.5, fontweight="bold", color=INK)
    y2 = 1.75
    box(ax, 1.0, y2, W, H, "Câu hỏi\ncủa người dùng")
    box(ax, 3.0, y2, W, H, "Tìm kiếm lai\nngữ nghĩa + BM25\n(gộp bằng RRF)")
    box(ax, 5.0, y2, W, H, "Xếp hạng lại\n10 đoạn\n(bge-reranker-v2-m3)")
    # nút quyết định
    cx, cy = 6.85, y2
    ax.add_patch(Polygon([(cx - 0.75, cy), (cx, cy + 0.5), (cx + 0.75, cy), (cx, cy - 0.5)], closed=True,
                         facecolor="white", edgecolor=EDGE, linewidth=1.1))
    ax.text(cx, cy, "Điểm cao nhất\n≥ 0,5 ?", ha="center", va="center", fontsize=8.2, color=INK)
    box(ax, 9.0, y2, W, H, "LLM qua Ollama\n(prompt + 5 đoạn)\nTrả lời kèm nguồn", accent=ORANGE)
    box(ax, cx, 0.45, 1.7, 0.55, "Từ chối: “chưa tìm thấy\nthông tin trong dữ liệu”")
    arrow(ax, (1.0 + W / 2, y2), (3.0 - W / 2, y2))
    arrow(ax, (3.0 + W / 2, y2), (5.0 - W / 2, y2))
    arrow(ax, (5.0 + W / 2, y2), (cx - 0.75, y2))
    arrow(ax, (cx + 0.75, y2), (9.0 - W / 2, y2), "có", (cx + 0.95, y2 + 0.05))
    arrow(ax, (cx, cy - 0.5), (cx, 0.45 + 0.275), "không", (cx + 0.35, cy - 0.85))
    # chỉ mục phục vụ tìm kiếm
    arrow(ax, (9.0, y1 - H / 2), (3.0, y2 + H / 2), "tra cứu chỉ mục", (6.4, 2.95), color=BLUE)
    fig.tight_layout()
    fig.savefig(FIG / "2_1_kien_truc.png", dpi=220)
    plt.close(fig)


def process(steps, fname, title=None):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    gap = 10 / n
    W, H, y = gap - 0.35, 1.05, 0.9
    xs = [gap / 2 + i * gap for i in range(n)]
    for i, (x, t) in enumerate(zip(xs, steps)):
        box(ax, x, y, W, H, f"{i + 1}. {t}", fs=8)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + W / 2, y), (b - W / 2, y))
    if title:
        ax.text(0.05, 1.75, title, fontsize=9.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    if len(sys.argv) > 1:  # các bước quy trình truyền qua dòng lệnh, phân tách bằng "|"
        process([s.replace("\\n", "\n") for s in sys.argv[1].split("|")], "1_1_quy_trinh.png")
    print("Đã vẽ:", sorted(p.name for p in FIG.glob("[12]_*.png")))
