"""Bước 2: Phân tích khám phá dữ liệu (mục 3.1.2 của báo cáo).

Chạy:  python scripts/02_eda.py
Đầu ra: reports/figures/02_*.png, reports/02_missing.csv, reports/02_ban_do.html
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PLACES_FILE, REPORT_DIR  # noqa: E402

FIG = REPORT_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

df = pd.DataFrame([json.loads(l) for l in PLACES_FILE.open(encoding="utf-8")])
print(f"{len(df)} bản ghi; {df['type'].value_counts().to_dict()}")


def barh(series, title, fname, n=20, color="#3b6ea5"):
    s = series.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(s) + 1)))
    ax.barh(s.index, s.values, color=color)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150)
    plt.close()


barh(df["region"].fillna("(chưa rõ)").value_counts(), "Số địa điểm theo tỉnh/thành (top 20)", "02_theo_tinh.png")
barh(df["category"].fillna("(chưa rõ)").value_counts(), "Số địa điểm theo danh mục", "02_theo_danh_muc.png")

# Tỷ lệ thiếu dữ liệu theo trường
fields = ["subtitle", "category", "region", "address", "lat", "hours_label", "open_time", "price_label",
          "price_amount", "best_season", "duration_label", "services", "description", "rag_text", "map_url", "source_url"]
miss = {}
for f in fields:
    col = df[f]
    empty = col.isna() | col.apply(lambda v: v == [] or v == "")
    miss[f] = round(100 * empty.mean(), 1)
ms = pd.Series(miss).sort_values(ascending=False)
ms.rename("ty_le_thieu_%").to_csv(REPORT_DIR / "02_missing.csv", encoding="utf-8-sig")
barh(ms, "Tỷ lệ thiếu dữ liệu theo trường (%)", "02_ty_le_thieu.png", n=len(ms), color="#c0504d")

# Độ dài văn bản (số từ)
words = lambda s: len(str(s).split()) if isinstance(s, str) else 0  # noqa: E731
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, col in zip(axes, ["description", "rag_text"]):
    w = df[col].apply(words)
    ax.hist(w[w > 0], bins=40, color="#3b6ea5")
    ax.set_title(f"Số từ của {col} (trung vị {int(w[w > 0].median())})")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "02_do_dai_van_ban.png", dpi=150)
plt.close()

# Từ khóa phổ biến
tags = Counter(t.lower() for ts in df["tags"] for t in (ts or []))
pd.Series(dict(tags.most_common(30))).rename("so_lan").to_csv(REPORT_DIR / "02_top_tags.csv", encoding="utf-8-sig")
barh(pd.Series(dict(tags.most_common(25))), "25 từ khóa (tags) phổ biến nhất", "02_top_tags.png", n=25, color="#4f8a5b")

# Bản đồ phân bố
try:
    import folium
    from folium.plugins import MarkerCluster

    m = folium.Map(location=[16.0, 106.5], zoom_start=6, tiles="OpenStreetMap")
    cl = MarkerCluster().add_to(m)
    geo = df.dropna(subset=["lat", "lng"])
    for _, r in geo.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=3, fill=True,
                            popup=f"{r['title']} ({r['category'] or ''}, {r['region'] or ''})").add_to(cl)
    m.save(str(REPORT_DIR / "02_ban_do.html"))
    print(f"Bản đồ: {len(geo)} điểm có tọa độ")
except ImportError:
    print("Chưa cài folium, bỏ qua bản đồ")

print("Tỷ lệ thiếu (%):\n", ms.to_string())
print("Đã lưu hình vào", FIG)
