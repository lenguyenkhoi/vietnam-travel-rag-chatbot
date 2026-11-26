"""Chia văn bản địa điểm thành các đoạn (chunk) để tạo chỉ mục.

Ba chiến lược để so sánh trong báo cáo:
- "doc"     : mỗi địa điểm 1 đoạn (thẻ thông tin + phần đầu rag_text).
- "fixed"   : cửa sổ cố định CHUNK_SIZE ký tự, chồng lấn OVERLAP ký tự.
- "section" : chia theo các mục có tiêu đề in hoa trong rag_text (NHẬN DIỆN, TRẢI NGHIỆM...),
              gộp mục ngắn, tách mục dài theo đoạn văn. (mặc định)

Mọi chiến lược đều thêm 1 "thẻ thông tin" có cấu trúc (địa chỉ, giờ, giá, mùa...) cho mỗi địa điểm
và gắn tiêu đề địa điểm vào đầu mỗi đoạn để giữ ngữ cảnh.
"""
import re

CHUNK_SIZE = 1200
OVERLAP = 200
DOC_CHARS = 1800

# Dòng tiêu đề mục: chữ in hoa, đứng riêng một dòng hoặc kết thúc bằng dấu ":"
_HEAD = re.compile(r"^\s*([A-ZÀ-Ỹ0-9][A-ZÀ-Ỹ0-9 &/()+,\-–—]{2,60})\s*:?\s*$")


def place_header(p: dict) -> str:
    bits = [p["title"]]
    meta = ", ".join(x for x in (p.get("category"), p.get("region")) if x)
    return f"{bits[0]} ({meta})" if meta else bits[0]


def info_card(p: dict) -> str:
    """Thẻ thông tin có cấu trúc, giúp trả lời câu hỏi thực tế (địa chỉ, giờ, giá)."""
    rows = [
        ("Tên", p.get("title")),
        ("Giới thiệu ngắn", p.get("subtitle")),
        ("Loại", "Sự kiện" if p.get("type") == "su_kien" else "Địa điểm"),
        ("Danh mục", p.get("category")),
        ("Tỉnh/Thành phố", p.get("region")),
        ("Khu vực", p.get("area")),
        ("Địa chỉ", p.get("address")),
        ("Giờ mở cửa", p.get("hours_label") or _hours(p)),
        ("Thời gian tham quan", p.get("duration_label")),
        ("Giá vé", p.get("price_label")),
        ("Mùa đẹp nhất", p.get("best_season")),
        ("Thời gian diễn ra", p.get("window_label") or _when(p)),
        ("Địa điểm tổ chức", p.get("venue")),
        ("Dịch vụ, hoạt động", "; ".join(p.get("services") or []) or None),
        ("Từ khóa", ", ".join(p.get("tags") or []) or None),
    ]
    return "\n".join(f"{k}: {v}" for k, v in rows if v)


def _hours(p):
    if p.get("open_time") and p.get("close_time"):
        return f"{p['open_time']} - {p['close_time']}"
    return None


def _when(p):
    if p.get("start_at"):
        return f"{p['start_at']} đến {p.get('end_at') or '?'}"
    return None


def split_sections(text: str) -> list[str]:
    lines = (text or "").split("\n")
    sections, cur = [], []
    for ln in lines:
        if _HEAD.match(ln) and sum(c.isupper() for c in ln) >= 3 and cur:
            sections.append("\n".join(cur).strip())
            cur = []
        cur.append(ln)
    if cur:
        sections.append("\n".join(cur).strip())
    return [s for s in sections if s]


def _split_long(text: str, size: int) -> list[str]:
    """Tách đoạn dài theo đoạn văn, sau đó theo câu, để mỗi phần <= size."""
    if len(text) <= size:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|(?<=[.!?])\s+", text):
        if len(cur) + len(para) + 1 > size and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    out = []
    for p in parts:  # câu quá dài: cắt cứng
        out.extend(p[i:i + size] for i in range(0, len(p), size))
    return out


def fixed_windows(text: str, size=CHUNK_SIZE, overlap=OVERLAP) -> list[str]:
    text = text or ""
    if len(text) <= size:
        return [text] if text else []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text) - overlap, step)]


def chunk_place(p: dict, strategy: str = "section") -> list[dict]:
    head = place_header(p)
    body = p.get("rag_text") or p.get("description") or ""
    chunks = [("the_thong_tin", info_card(p))]
    if strategy == "doc":
        txt = (p.get("description") or "") + "\n\n" + body
        chunks = [("toan_bo", info_card(p) + "\n\n" + txt[:DOC_CHARS])]
    elif strategy == "fixed":
        chunks += [("doan", w) for w in fixed_windows(body)]
    elif strategy == "section":
        buf = ""
        for sec in split_sections(body):
            for piece in _split_long(sec, CHUNK_SIZE):
                if len(buf) + len(piece) + 2 > CHUNK_SIZE and buf:
                    chunks.append(("muc", buf.strip()))
                    buf = ""
                buf += piece + "\n\n"
        if buf.strip():
            chunks.append(("muc", buf.strip()))
    else:
        raise ValueError(f"Chiến lược không hợp lệ: {strategy}")

    out = []
    for i, (kind, txt) in enumerate(chunks):
        if not txt.strip():
            continue
        out.append({
            "chunk_id": f"{p['doc_id']}#{i}",
            "doc_id": p["doc_id"],
            "kind": kind,
            "title": p["title"],
            "category": p.get("category"),
            "region": p.get("region"),
            "text": f"[{head}]\n{txt.strip()}",
        })
    return out
