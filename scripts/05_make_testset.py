"""Bước 5: Sinh bộ câu hỏi kiểm thử NHÁP (cần người rà soát lại trước khi dùng cho báo cáo).

Các loại câu hỏi:
- thong_tin     : hỏi thông tin thực tế (địa chỉ, giờ mở cửa, giá vé) của 1 địa điểm cụ thể.
- mo_ta_an      : mô tả địa điểm nhưng GIẤU tên, hỏi "đó là nơi nào" (kiểm tra tìm kiếm theo nghĩa).
- goi_y         : gợi ý theo danh mục + tỉnh (nhiều đáp án đúng).
- tu_nhien      : câu hỏi tự nhiên do LLM viết từ 1 đoạn văn (tùy chọn --llm).
- ngoai_pham_vi : câu hỏi không có trong dữ liệu; hệ thống phải từ chối.

Chạy:  python scripts/05_make_testset.py               (nhanh, không cần LLM)
       python scripts/05_make_testset.py --llm 40      (thêm 40 câu do qwen2.5:3b viết, ~10 phút)
Đầu ra: eval/testset_draft.csv  -> rà soát, sửa, rồi lưu thành eval/testset.csv
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.chunking import chunk_place  # noqa: E402
from rag.config import DEFAULT_LLM, EVAL_DIR, PLACES_FILE  # noqa: E402
from rag.text import norm_key  # noqa: E402

OUT_OF_SCOPE = [
    "Giá vé máy bay từ TP.HCM đi Paris tháng sau là bao nhiêu?",
    "Thời tiết ở Tokyo ngày mai thế nào?",
    "Cách nấu món mì Ý sốt kem nấm?",
    "Tỷ giá đô la Mỹ hôm nay là bao nhiêu?",
    "Ai là người đầu tiên đặt chân lên Mặt Trăng?",
    "Kết quả trận bóng đá Ngoại hạng Anh tối qua?",
    "Làm sao để xin visa du lịch Hàn Quốc?",
    "Khách sạn 5 sao nào ở Singapore rẻ nhất?",
    "Tháp Eiffel cao bao nhiêu mét?",
    "Cách cài đặt Python trên máy Mac?",
    "Lịch thi đấu SEA Games năm sau?",
    "Giá vàng SJC hôm nay bao nhiêu?",
    "Viết giúp tôi một bài thơ về mùa thu Hà Nội.",
    "Bảo tàng Louvre mở cửa mấy giờ?",
    "Công thức tính diện tích hình tròn là gì?",
]

LLM_PROMPT = """Dưới đây là một đoạn thông tin về địa điểm "{title}" ({region}).
Hãy viết MỘT câu hỏi tự nhiên bằng tiếng Việt mà một du khách có thể hỏi, sao cho câu trả lời nằm trong đoạn này.
Sau đó viết câu trả lời ngắn (1 câu) lấy từ đoạn này.
Trả về đúng định dạng 2 dòng:
CÂU HỎI: ...
TRẢ LỜI: ...

ĐOẠN THÔNG TIN:
{text}"""


def good(v, minlen=4):
    return isinstance(v, str) and len(v) >= minlen and not re.search(r"đang cập nhật|liên hệ", v, re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="số câu mỗi loại thong_tin / mo_ta_an")
    ap.add_argument("--n-goi-y", type=int, default=30)
    ap.add_argument("--llm", type=int, default=0, help="số câu tự nhiên do LLM viết (0 = bỏ qua)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    places = [json.loads(l) for l in PLACES_FILE.open(encoding="utf-8")]
    places = [p for p in places if p["type"] == "dia_diem" and p.get("region") and p.get("category")]
    name_count = defaultdict(int)
    for p in places:
        name_count[norm_key(p["title"])] += 1
    rows = []

    # 1. thông tin thực tế
    pool = places[:]
    rnd.shuffle(pool)
    makers = [
        ("address", "Địa chỉ của {t} ở {r} là ở đâu?"),
        ("hours_label", "{t} ({r}) mở cửa vào giờ nào?"),
        ("price_label", "Giá vé tham quan {t} ở {r} là bao nhiêu?"),
    ]
    used = set()
    for field, tpl in makers * args.n:
        if sum(r["loai"] == "thong_tin" for r in rows) >= args.n:
            break
        for p in pool:
            if p["doc_id"] in used or not good(p.get(field)):
                continue
            used.add(p["doc_id"])
            rows.append({"loai": "thong_tin", "truong": field, "cau_hoi": tpl.format(t=p["title"], r=p["region"]),
                         "doc_ids_dung": p["doc_id"], "dap_an_tham_chieu": p[field]})
            break

    # 2. mô tả giấu tên
    pool2 = [p for p in places if good(p.get("subtitle"), 30) and p["doc_id"] not in used
             and norm_key(p["title"]) not in norm_key(p["subtitle"])]
    for p in rnd.sample(pool2, min(args.n, len(pool2))):
        rows.append({"loai": "mo_ta_an", "truong": "subtitle",
                     "cau_hoi": f"{p['subtitle'].rstrip('.')}. Đó là địa điểm nào ở {p['region']}?",
                     "doc_ids_dung": p["doc_id"], "dap_an_tham_chieu": p["title"]})

    # 3. gợi ý theo danh mục + tỉnh
    groups = defaultdict(list)
    for p in places:
        groups[(p["category"], p["region"])].append(p["doc_id"])
    combos = [k for k, v in groups.items() if len(v) >= 3]
    for cat, reg in rnd.sample(combos, min(args.n_goi_y, len(combos))):
        rows.append({"loai": "goi_y", "truong": "category+region",
                     "cau_hoi": f"Gợi ý giúp mình vài địa điểm thuộc nhóm {cat.lower()} ở {reg}.",
                     "doc_ids_dung": ";".join(groups[(cat, reg)]), "dap_an_tham_chieu": f"{cat} - {reg}"})

    # 4. câu hỏi tự nhiên do LLM viết
    if args.llm:
        import ollama
        # Lưu từng câu ngay khi sinh xong: nếu bị dừng giữa chừng, lần chạy sau dùng lại, không gọi LLM lần nữa
        cache_file = EVAL_DIR / "_tu_nhien_cache.jsonl"
        EVAL_DIR.mkdir(exist_ok=True)
        cache = {}
        if cache_file.exists():
            for line in cache_file.open(encoding="utf-8"):
                c = json.loads(line)
                cache[c["chunk_id"]] = c
        print(f"Đã có sẵn {len(cache)} câu tự nhiên trong cache")
        cands = [p for p in places if p["doc_id"] not in used and name_count[norm_key(p["title"])] == 1]
        for p in rnd.sample(cands, min(args.llm * 2, len(cands))):
            if sum(r["loai"] == "tu_nhien" for r in rows) >= args.llm:
                break
            secs = [c for c in chunk_place(p, "section") if c["kind"] == "muc"]
            if not secs:
                continue
            ch = rnd.choice(secs)
            if ch["chunk_id"] in cache:
                q, a = cache[ch["chunk_id"]]["q"], cache[ch["chunk_id"]]["a"]
            else:
                msg = LLM_PROMPT.format(title=p["title"], region=p["region"], text=ch["text"][:1500])
                out = ollama.chat(model=DEFAULT_LLM, messages=[{"role": "user", "content": msg}],
                                  options={"temperature": 0.3})["message"]["content"]
                mq = re.search(r"CÂU HỎI:\s*(.+)", out)
                ma = re.search(r"TRẢ LỜI:\s*(.+)", out)
                q, a = (mq.group(1).strip(), ma.group(1).strip()) if mq and ma else (None, None)
                with cache_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"chunk_id": ch["chunk_id"], "q": q, "a": a}, ensure_ascii=False) + "\n")
            if q and a:
                rows.append({"loai": "tu_nhien", "truong": ch["chunk_id"], "cau_hoi": q,
                             "doc_ids_dung": p["doc_id"], "dap_an_tham_chieu": a})
                print(len(rows), q[:100], flush=True)

    # 5. ngoài phạm vi
    for q in OUT_OF_SCOPE:
        rows.append({"loai": "ngoai_pham_vi", "truong": "", "cau_hoi": q, "doc_ids_dung": "",
                     "dap_an_tham_chieu": "TỪ CHỐI"})

    df = pd.DataFrame(rows)
    df.insert(0, "qid", [f"Q{i:03d}" for i in range(1, len(df) + 1)])
    df["da_kiem_tra"] = ""
    EVAL_DIR.mkdir(exist_ok=True)
    out = EVAL_DIR / "testset_draft.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df["loai"].value_counts().to_string())
    print("Đã ghi", out, "-> rà soát rồi lưu thành eval/testset.csv")


if __name__ == "__main__":
    main()
