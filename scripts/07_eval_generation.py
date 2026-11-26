"""Bước 7: Đánh giá câu trả lời của các LLM (mục 4.2 và 4.3 của báo cáo).

- Cùng một kết quả truy xuất (lai, top-5) được đưa cho mọi LLM để so sánh công bằng.
- Chỉ số tự động:
    dung_tu_dong : câu trả lời chứa đáp án tham chiếu (độ phủ âm tiết >= 0.6; câu gợi ý: nêu ít nhất 1 địa điểm đúng)
    tu_choi      : câu trả lời là câu từ chối
    giay         : thời gian sinh câu trả lời
- Tùy chọn --judge <model>: thêm LLM làm giám khảo chấm "đúng" và "trung thực với ngữ cảnh".
- File kết quả có sẵn cột để CHẤM TAY một mẫu nhằm kiểm chứng chấm tự động.

Chạy:  python scripts/07_eval_generation.py --n 40                       (mọi LLM đã cài, có RAG)
       python scripts/07_eval_generation.py --n 40 --no-rag-baseline     (thêm so sánh không RAG)
       python scripts/07_eval_generation.py --llms qwen2.5:3b --judge qwen2.5:7b
Đầu ra: reports/07_generation_answers.csv, reports/07_generation_summary.csv
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import ollama
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import DEFAULT_EMBED, DEFAULT_STRATEGY, EVAL_DIR, LLM_MODELS, REPORT_DIR  # noqa: E402
from rag.config import RERANK_MIN_SCORE  # noqa: E402
from rag.generator import DEFAULT_PROMPT, PROMPTS, REFUSAL, Generator, build_context  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

REFUSE_PAT = re.compile(
    r"chưa tìm thấy thông tin|không tìm thấy thông tin|không có thông tin|không có trong (dữ liệu|ngữ cảnh)|không biết"
    r"|không thể cung cấp|không thể trả lời|không có (quyền truy cập|khả năng|dữ liệu)|nằm ngoài phạm vi", re.I)
SYL = re.compile(r"[0-9a-zà-ỹđ]+", re.I)
PROVINCES = ["Hồ Chí Minh", "Hà Nội", "Hải Phòng", "Đà Nẵng", "Huế", "Cần Thơ", "Lâm Đồng", "Quảng Ninh", "Quảng Trị",
             "Tây Ninh", "Vĩnh Long", "An Giang", "Gia Lai", "Đồng Nai", "Nghệ An", "Đồng Tháp", "Cao Bằng", "Thanh Hóa",
             "Lạng Sơn", "Tuyên Quang", "Khánh Hòa", "Quảng Ngãi", "Lào Cai", "Hưng Yên", "Cà Mau", "Hà Tĩnh",
             "Phú Thọ", "Bắc Ninh", "Ninh Bình", "Điện Biên", "Đắk Lắk", "Thái Nguyên", "Sơn La", "Lai Châu"]

JUDGE_PROMPT = """Bạn là giám khảo chấm điểm hệ thống hỏi đáp. Chỉ trả về JSON.
CÂU HỎI: {q}
ĐÁP ÁN THAM CHIẾU: {ref}
NGỮ CẢNH ĐƯỢC CUNG CẤP CHO HỆ THỐNG:
{ctx}
CÂU TRẢ LỜI CỦA HỆ THỐNG: {ans}

Chấm 2 tiêu chí (0 hoặc 1):
- "dung": câu trả lời đúng và phù hợp với đáp án tham chiếu.
- "trung_thuc": mọi thông tin trong câu trả lời đều có trong ngữ cảnh (không bịa).
Trả về: {{"dung": 0 hoặc 1, "trung_thuc": 0 hoặc 1}}"""


def syllables(s):
    return [w.lower() for w in SYL.findall(s or "")]


def auto_correct(row, answer, titles_by_id):
    if row["loai"] == "ngoai_pham_vi":
        return float(bool(REFUSE_PAT.search(answer)))
    if row["loai"] == "goi_y":
        # đúng nếu nêu được ít nhất 1 địa điểm đúng; chấp nhận tên rút gọn (bỏ tên tỉnh và phần trong ngoặc),
        # ví dụ "Chợ Ga" cho "Chợ Ga Hải Phòng", "Chợ Đà Lạt" cho "Chợ Đà Lạt (Lâm Đồng)"
        a = answer.lower()
        for d in row["doc_ids_dung"].split(";"):
            t = titles_by_id.get(d, "")
            if not t:
                continue
            names = {t.lower(), re.sub(r"\s*\(.*?\)", "", t).strip().lower()}
            for prov in PROVINCES:
                names |= {n.replace(prov.lower(), "").strip(" -–,") for n in list(names)}
            if any(len(n) >= 5 and n in a for n in names):
                return 1.0
        return 0.0
    ref = syllables(row["dap_an_tham_chieu"])[:40]
    if not ref:
        return 0.0
    ans = set(syllables(answer))
    return float(sum(w in ans for w in ref) / len(ref) >= 0.6)


def judge(model, q, ref, ctx, ans):
    try:
        out = ollama.chat(model=model, format="json", options={"temperature": 0},
                          messages=[{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, ctx=ctx[:4000], ans=ans)}])
        j = json.loads(out["message"]["content"])
        return float(j.get("dung", 0)), float(j.get("trung_thuc", 0))
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="số câu hỏi (chia đều theo loại)")
    ap.add_argument("--llms", nargs="*", help="mặc định: mọi mô hình trong LLM_MODELS đã cài")
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rag-baseline", action="store_true")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="", help="hậu tố tên file kết quả, ví dụ 'rut_gon' -> 07_generation_*_rut_gon.csv")
    ap.add_argument("--fresh", action="store_true", help="bỏ kết quả cũ, chạy lại từ đầu (mặc định: chạy tiếp)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, choices=list(PROMPTS), help="phiên bản prompt (v1, v2)")
    ap.add_argument("--rerank", default="none", help="tên re-ranker, ví dụ bge-reranker-v2-m3 (mặc định: none)")
    ap.add_argument("--min-score", type=float, default=RERANK_MIN_SCORE,
                    help="ngưỡng liên quan khi có re-ranker; < 0 để tắt ngưỡng")
    ap.add_argument("--rescore", action="store_true",
                    help="chỉ chấm lại từ file câu trả lời đã có (khi sửa cách chấm), không gọi LLM")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""

    if args.rescore:
        rescore(sfx)
        return

    installed = {m["model"] for m in ollama.list()["models"]}
    llms = args.llms or [m for m in LLM_MODELS if m in installed]
    print("LLM:", llms)

    f = EVAL_DIR / "testset.csv"
    f = f if f.exists() else EVAL_DIR / "testset_draft.csv"
    ts = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
    per = max(1, args.n // ts["loai"].nunique())
    sample = ts.groupby("loai", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=args.seed))
    sample = sample.reset_index(drop=True)
    print(sample["loai"].value_counts().to_dict())

    rerank = None if args.rerank.lower() == "none" else args.rerank
    min_score = args.min_score if args.min_score >= 0 else None
    print(f"Cấu hình: prompt={args.prompt}, rerank={rerank}, ngưỡng={min_score if rerank else '-'}")
    pipe = RAGPipeline(args.embed, args.strategy, llms[0], rerank=rerank, prompt=args.prompt, min_score=min_score)
    titles = {d: p["title"] for d, p in pipe.places.items()}
    # tìm kiếm 1 lần cho mỗi câu, dùng chung cho mọi LLM để so sánh công bằng
    retrieved, gated, t_ret = {}, {}, {}
    for _, q in sample.iterrows():
        chunks, _, dt_r = pipe.retrieve(q["cau_hoi"], k=args.k, method="hybrid")
        retrieved[q["qid"]], gated[q["qid"]], t_ret[q["qid"]] = chunks, pipe.below_threshold(chunks), dt_r
    print(f"Bị chặn bởi ngưỡng liên quan: {sum(gated.values())}/{len(gated)} câu")

    out_csv = REPORT_DIR / f"07_generation_answers{sfx}.csv"
    rows = []
    if out_csv.exists() and not args.fresh:  # chạy tiếp: giữ các câu đã trả lời ở lần chạy trước
        rows = pd.read_csv(out_csv, encoding="utf-8-sig", dtype={"qid": str}).fillna("").to_dict("records")
        print(f"Chạy tiếp: đã có {len(rows)} câu trả lời trong {out_csv.name}")
    done = {(r["llm"], r["che_do"], r["qid"]) for r in rows}
    modes = [("co_rag", True)] + ([("khong_rag", False)] if args.no_rag_baseline else [])
    for llm in llms:
        if all((llm, m, q) in done for m, _ in modes for q in sample["qid"]):
            continue  # mô hình này đã chạy xong ở lần trước
        gen = Generator(llm, prompt=args.prompt)
        gen.generate("Xin chào", None)  # nạp mô hình vào RAM trước khi đo thời gian
        for mode, use_rag in modes:
            for _, q in sample.iterrows():
                if (llm, mode, q["qid"]) in done:
                    continue
                chunks = retrieved[q["qid"]] if use_rag else None
                blocked = use_rag and gated[q["qid"]]
                t0 = time.time()
                try:
                    ans = REFUSAL if blocked else gen.generate(q["cau_hoi"], chunks)
                except Exception as e:
                    ans = f"[LỖI] {e}"
                dt = time.time() - t0
                row = {"llm": llm, "che_do": mode, "qid": q["qid"], "loai": q["loai"], "cau_hoi": q["cau_hoi"],
                       "dap_an_tham_chieu": q["dap_an_tham_chieu"], "tra_loi": ans, "giay": round(dt, 2),
                       "tu_choi": float(bool(REFUSE_PAT.search(ans))),
                       "dung_tu_dong": auto_correct(q, ans, titles),
                       "nguon_top1": chunks[0]["title"] if chunks else "",
                       "prompt": args.prompt if use_rag else "", "rerank": (rerank or "") if use_rag else "",
                       "chan_boi_nguong": float(blocked), "giay_tim_kiem": round(t_ret[q["qid"]], 2) if use_rag else 0.0}
                if args.judge:
                    ctx = build_context(chunks) if chunks else "(không có ngữ cảnh)"
                    row["dung_giam_khao"], row["trung_thuc_giam_khao"] = judge(args.judge, q["cau_hoi"],
                                                                               q["dap_an_tham_chieu"], ctx, ans)
                row["cham_tay_dung"] = ""
                row["cham_tay_trung_thuc"] = ""
                rows.append(row)
                print(f"[{llm} {mode}] {q['qid']} {dt:5.1f}s dung={row['dung_tu_dong']:.0f} | {ans[:80]!r}")
                pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")  # lưu dần

    summarize(pd.DataFrame(rows), sfx)


def rescore(sfx):
    """Chấm lại cột tu_choi / dung_tu_dong từ file câu trả lời đã lưu, rồi tính lại bảng tổng hợp."""
    import json
    from rag.config import PLACES_FILE
    out_csv = REPORT_DIR / f"07_generation_answers{sfx}.csv"
    df = pd.read_csv(out_csv, encoding="utf-8-sig", dtype={"qid": str})
    df["tra_loi"] = df["tra_loi"].fillna("")
    ts = pd.read_csv(EVAL_DIR / "testset.csv", encoding="utf-8-sig", dtype=str).fillna("").set_index("qid")
    titles = {}
    for line in PLACES_FILE.open(encoding="utf-8"):
        p = json.loads(line)
        titles[p["doc_id"]] = p["title"]
    df["tu_choi"] = df["tra_loi"].apply(lambda a: float(bool(REFUSE_PAT.search(a))))
    df["dung_tu_dong"] = [auto_correct(ts.loc[q], a, titles) for q, a in zip(df["qid"], df["tra_loi"])]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Đã chấm lại {len(df)} câu trả lời trong {out_csv.name}")
    summarize(df, sfx)


def summarize(df, sfx):
    summ = []
    for (llm, mode), g in df.groupby(["llm", "che_do"]):
        gi, go = g[g["loai"] != "ngoai_pham_vi"], g[g["loai"] == "ngoai_pham_vi"]
        s = {"llm": llm, "che_do": mode, "so_cau": len(g),
             "ty_le_dung_trong_pham_vi": round(gi["dung_tu_dong"].mean(), 3),
             "ty_le_tu_choi_dung_ngoai_pham_vi": round(go["tu_choi"].mean(), 3) if len(go) else None,
             "ty_le_tu_choi_nham": round(gi["tu_choi"].mean(), 3),
             "giay_tb": round(g["giay"].mean(), 2), "giay_p90": round(g["giay"].quantile(.9), 2)}
        if "giay_tim_kiem" in g:  # có từ khi thêm re-ranker; tổng thời gian = tìm kiếm + sinh câu trả lời
            tk = pd.to_numeric(g["giay_tim_kiem"], errors="coerce").fillna(0)
            s["giay_tim_kiem_tb"] = round(tk.mean(), 2)
            s["giay_tong_tb"] = round((tk + g["giay"]).mean(), 2)
            s["ty_le_chan_boi_nguong"] = round(pd.to_numeric(g["chan_boi_nguong"], errors="coerce").fillna(0).mean(), 3)
        if "dung_giam_khao" in g:
            s["dung_giam_khao"] = round(gi["dung_giam_khao"].mean(), 3)
            s["trung_thuc_giam_khao"] = round(gi["trung_thuc_giam_khao"].mean(), 3)
        for loai, gl in g.groupby("loai"):
            s[f"dung_{loai}"] = round(gl["dung_tu_dong"].mean(), 3)
        summ.append(s)
    sm = pd.DataFrame(summ)
    sm.to_csv(REPORT_DIR / f"07_generation_summary{sfx}.csv", index=False, encoding="utf-8-sig")
    print("\n", sm.to_string(index=False))


if __name__ == "__main__":
    main()
