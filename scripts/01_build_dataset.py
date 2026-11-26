"""Bước 1: Gom, chuẩn hóa và loại trùng dữ liệu địa điểm/sự kiện của Localis.

Đầu vào : toàn bộ file *.json trong thư mục dữ liệu Localis (mặc định: thư mục cha của chatbot-rag).
Đầu ra  : data/places.jsonl (mỗi dòng một địa điểm) và reports/01_build_dataset.json (thống kê từng bước).

Chỉ ĐỌC file gốc, không sửa dữ liệu của công ty.

Chạy:  python scripts/01_build_dataset.py [--root D:/localis]
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

CATEGORIES = [
    "Không gian di sản", "Chợ địa phương", "Bảo tàng", "Văn hóa ẩm thực", "Làng cộng đồng", "Lễ hội",
    "Làng nghề", "Thiên nhiên & Mùa", "Nghệ thuật & Sáng tạo", "Nhà hát & Sân khấu", "Phòng tranh",
    "An toàn & Hỗ trợ", "Quà lưu niệm", "Phong tục & Nghi thức",
]
# Thư mục sao lưu / bản cũ / bản đã bỏ: bỏ qua hoàn toàn
SKIP_DIR = re.compile(r"^(_backup|_sao-luu|_ban-goc|_da_bo|old$|_scripts$|_VI_DU)", re.I)
PLACEHOLDER = re.compile(r"^\s*(đang cập nhật|chưa cập nhật|n/?a|none|null|-|không rõ)?\s*$", re.I)
REGION_ALIASES = {
    "tp. hồ chí minh": "Hồ Chí Minh", "thành phố hồ chí minh": "Hồ Chí Minh", "tp.hcm": "Hồ Chí Minh",
    "tp hcm": "Hồ Chí Minh", "hcm": "Hồ Chí Minh", "tp. cần thơ": "Cần Thơ", "tp. đà nẵng": "Đà Nẵng",
    "tp. hà nội": "Hà Nội", "tp. hải phòng": "Hải Phòng", "tp. huế": "Huế", "thừa thiên huế": "Huế",
}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(s or "").lower()).strip()


CAT_BY_KEY = {norm_key(c): c for c in CATEGORIES}
# Tên thư mục tiếng Anh / biến thể -> danh mục chuẩn
CAT_ALIASES = {
    "heritage": "Không gian di sản", "nature seasons": "Thiên nhiên & Mùa", "thien nhien cac mua": "Thiên nhiên & Mùa",
    "museum": "Bảo tàng", "museums": "Bảo tàng", "festival": "Lễ hội", "festivals": "Lễ hội",
    "craft village": "Làng nghề", "craft villages": "Làng nghề", "local market": "Chợ địa phương",
    "local markets": "Chợ địa phương", "cuisine": "Văn hóa ẩm thực", "food culture": "Văn hóa ẩm thực",
    "community village": "Làng cộng đồng", "community villages": "Làng cộng đồng",
    "art creativity": "Nghệ thuật & Sáng tạo", "theatre stage": "Nhà hát & Sân khấu", "gallery": "Phòng tranh",
    "galleries": "Phòng tranh", "souvenir": "Quà lưu niệm", "souvenirs": "Quà lưu niệm",
    "safety support": "An toàn & Hỗ trợ", "customs rituals": "Phong tục & Nghi thức",
}
for _k, _v in CAT_ALIASES.items():
    CAT_BY_KEY[_k] = _v
EVENT_LIKE = {"triển lãm", "biểu diễn", "workshop", "sự kiện"}


def clean(v):
    """Chuẩn hóa chuỗi: NFC, bỏ khoảng trắng thừa, giá trị giữ chỗ -> None."""
    if v is None:
        return None
    if isinstance(v, str):
        v = unicodedata.normalize("NFC", v).replace("\r\n", "\n").strip()
        v = re.sub(r"[ \t]+", " ", v)
        v = re.sub(r"\n{3,}", "\n\n", v)
        return None if PLACEHOLDER.match(v) else v
    return v


def to_float(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def norm_region(r):
    r = clean(r) if isinstance(r, str) else None
    if not r:
        return None
    return REGION_ALIASES.get(r.lower(), r)


def detect_category(d, path_parts, id2cat):
    c = d.get("category")
    if isinstance(c, dict) and c.get("name"):
        if c["name"].lower() in EVENT_LIKE:
            return "Sự kiện"
        return c["name"]
    if d.get("category_id") in id2cat:
        name = id2cat[d["category_id"]]
        return "Sự kiện" if name.lower() in EVENT_LIKE else name
    for part in reversed(path_parts):  # "KHÁNH HÒA/Bảo tàng", "Gia Lai - Làng nghề", "12. Nature & Seasons (...)"
        k = norm_key(part)
        k = re.sub(r"^\d+ ", "", k)
        for ck, cname in CAT_BY_KEY.items():
            if k == ck or k.endswith(" " + ck) or k.startswith(ck + " "):
                return cname
    for t in d.get("tags") or []:
        if isinstance(t, str) and norm_key(t) in CAT_BY_KEY:
            return CAT_BY_KEY[norm_key(t)]
    return None


def detect_region(d, path_parts, known_regions):
    r = norm_region(d.get("region"))
    if r:
        return r
    p = d.get("province")
    if isinstance(p, dict):
        p = p.get("name")
    r = norm_region(p) if isinstance(p, str) else None
    if r:
        return r
    by_key = {norm_key(x): x for x in known_regions}
    for part in path_parts:
        k = norm_key(part.split(" - ")[0])
        if k in by_key:
            return by_key[k]
    for t in d.get("tags") or []:
        if isinstance(t, str) and norm_key(t) in by_key:
            return by_key[norm_key(t)]
    return None


def is_place_record(d):
    return (
        isinstance(d, dict) and d.get("title") and ("rag_text" in d or "description" in d)
        and ("lat" in d or "address" in d or "region" in d)
    )


def quality(rec, mtime):
    """Điểm chọn bản tốt nhất khi trùng slug: mới nhất, có id trên admin, nội dung dài."""
    return (mtime, rec.get("admin_id") is not None, len(rec.get("rag_text") or "") + len(rec.get("description") or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT.parent), help="Thư mục chứa dữ liệu JSON của Localis")
    args = ap.parse_args()
    root = Path(args.root)

    stats = Counter()
    raw = []
    for f in root.rglob("*.json"):
        rel = f.relative_to(root)
        if rel.parts[0] == PROJECT.name:
            continue
        stats["file_json"] += 1
        if any(SKIP_DIR.match(p) for p in rel.parts[:-1]):
            stats["bo_qua_thu_muc_sao_luu"] += 1
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8-sig"))
        except Exception:
            stats["loi_doc_json"] += 1
            continue
        if not is_place_record(d):
            stats["khong_phai_dia_diem"] += 1
            continue
        stats["ban_ghi_dia_diem"] += 1
        raw.append((rel, f.stat().st_mtime, d))

    # bảng tra category_id -> tên danh mục, và danh sách tỉnh đã biết
    id2cat = {}
    known_regions = set()
    for _, _, d in raw:
        c = d.get("category")
        if isinstance(c, dict) and c.get("id") is not None and c.get("name"):
            id2cat[c["id"]] = c["name"]
        r = norm_region(d.get("region"))
        if r and "toàn quốc" not in r.lower():
            known_regions.add(r)

    best = {}
    for rel, mtime, d in raw:
        parts = rel.parts[:-1]
        rec = {
            "doc_id": clean(d.get("slug")) or norm_key(d.get("title", "")).replace(" ", "-"),
            "type": "su_kien" if "start_at" in d else "dia_diem",
            "title": clean(d.get("title")),
            "subtitle": clean(d.get("subtitle")),
            "category": detect_category(d, parts, id2cat) or ("Sự kiện" if "start_at" in d else None),
            "region": detect_region(d, parts, known_regions),
            "area": clean(d.get("area")),
            "address": clean(d.get("address")),
            "lat": to_float(d.get("lat")),
            "lng": to_float(d.get("lng")),
            "hours_label": clean(d.get("hours_label")),
            "open_time": clean(d.get("open_time")),
            "close_time": clean(d.get("close_time")),
            "duration_label": clean(d.get("duration_label")),
            "price_label": clean(d.get("price_label")),
            "price_amount": to_float(d.get("price_amount")),
            "best_season": clean(d.get("best_season")),
            "start_at": clean(d.get("start_at")),
            "end_at": clean(d.get("end_at")),
            "window_label": clean(d.get("window_label")),
            "venue": clean(d.get("venue")) if isinstance(d.get("venue"), str) else None,
            "services": [clean(s) for s in (d.get("services") or []) if isinstance(s, str) and clean(s)],
            "tags": [clean(t) for t in (d.get("tags") or []) if isinstance(t, str) and clean(t)],
            "description": clean(d.get("description")),
            "rag_text": clean(d.get("rag_text")),
            "map_url": clean(d.get("map_url")),
            "source_url": clean(d.get("source_url")),
            "admin_id": d.get("id"),
            "src_file": str(rel).replace("\\", "/"),
        }
        if not rec["title"] or not (rec["rag_text"] or rec["description"]):
            stats["thieu_noi_dung"] += 1
            continue
        # tọa độ ngoài lãnh thổ Việt Nam (xấp xỉ) -> bỏ tọa độ
        if rec["lat"] is not None and not (8.0 <= rec["lat"] <= 23.5 and 102.0 <= (rec["lng"] or 0) <= 110.5):
            stats["toa_do_khong_hop_le"] += 1
            rec["lat"] = rec["lng"] = None
        key = rec["doc_id"]
        if key in best:
            stats["trung_slug"] += 1
            if quality(rec, mtime) <= best[key][0]:
                continue
        best[key] = (quality(rec, mtime), rec)

    # Loại trùng lần 2: cùng tên chuẩn hóa + cùng tỉnh (khác slug)
    by_name = {}
    for _, rec in best.values():
        k = (norm_key(rec["title"]), rec["region"])
        if k in by_name:
            stats["trung_ten_va_tinh"] += 1
            old = by_name[k]
            if len(rec["rag_text"] or "") <= len(old["rag_text"] or ""):
                continue
        by_name[k] = rec
    places = sorted(by_name.values(), key=lambda r: (r["region"] or "~", r["category"] or "~", r["title"]))

    out = PROJECT / "data" / "places.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in places:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["dia_diem_cuoi_cung"] = len(places)
    summary = {
        "buoc_loc": dict(stats),
        "theo_loai": Counter(r["type"] for r in places),
        "theo_danh_muc": Counter(r["category"] or "(chưa rõ)" for r in places).most_common(),
        "theo_tinh": Counter(r["region"] or "(chưa rõ)" for r in places).most_common(),
    }
    rep = PROJECT / "reports" / "01_build_dataset.json"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    print(json.dumps(summary["buoc_loc"], ensure_ascii=False, indent=2))
    print("Theo loại:", dict(summary["theo_loai"]))
    print("Danh mục:", summary["theo_danh_muc"])
    print("Số tỉnh:", len(summary["theo_tinh"]), summary["theo_tinh"][:10])
    print("Đã ghi", out)


if __name__ == "__main__":
    main()
