"""Tiện ích xử lý văn bản tiếng Việt."""
import re
import unicodedata


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def norm_key(s: str) -> str:
    """Chuỗi không dấu, chữ thường, chỉ chữ số và khoảng trắng (dùng để so khớp tên)."""
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(s).lower()).strip()


_WORD = re.compile(r"[0-9a-zà-ỹđ]+", re.I)


def tokenize(text: str) -> list[str]:
    """Tách token cho BM25: âm tiết chữ thường + cặp âm tiết liền kề (bigram).

    Tiếng Việt có từ ghép nhiều âm tiết ("bảo tàng", "lễ hội"), nên bigram giúp BM25
    phân biệt tốt hơn mà không cần bộ tách từ.
    """
    sy = _WORD.findall(unicodedata.normalize("NFC", text or "").lower())
    return sy + [a + "_" + b for a, b in zip(sy, sy[1:])]
