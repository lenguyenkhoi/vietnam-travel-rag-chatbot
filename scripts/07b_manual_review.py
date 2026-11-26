"""Bước 7b: Chấm tay (hỗ trợ bởi AI, sinh viên kiểm tra lại) các câu trả lời có RAG bị bộ chấm tự động đánh SAI
trong lần đánh giá đầy đủ (tag day_du), rồi tính độ chính xác sau điều chỉnh.

Lý do: bộ chấm tự động so khớp âm tiết nên bỏ sót câu đúng nhưng diễn đạt khác (ví dụ "Không, vào chợ không phải
mua vé" so với đáp án "Vào chợ miễn phí"). Đã kiểm tra thêm 15 câu ngẫu nhiên được chấm ĐÚNG: cả 15 đúng thật,
nên chỉ cần chấm lại nhóm bị chấm SAI.
Thang điểm: 1 = đúng; 0,5 = đúng một phần; 0 = sai. Kèm loại lỗi để phân tích (mục 4.2.2).

Chạy:  python scripts/07b_manual_review.py
Đầu ra: reports/07_cham_tay_day_du.csv, reports/07_final_summary.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import REPORT_DIR  # noqa: E402

# (llm, qid): (điểm, loại lỗi, ghi chú)
R = {
    ("qwen2.5:3b", "Q130"): (1, "cham_tu_dong_bo_sot", "Liệt kê đúng phong tục có trong dữ liệu"),
    ("qwen2.5:3b", "Q026"): (0, "tu_choi_nham", "Dữ liệu có thông tin 'không áp dụng, tùy nhà vườn'"),
    ("qwen2.5:3b", "Q018"): (0, "hieu_sai_cau_hoi", "'Bán bao nhiêu' bị hiểu là số lượng (3.000 trứng/ngày) thay vì giá 8.000đ"),
    ("qwen2.5:3b", "Q021"): (0, "tu_choi_nham", "Dữ liệu ghi 'vào chợ miễn phí'"),
    ("qwen2.5:3b", "Q036"): (0, "bia_thong_tin", "Bịa 'vài chục nghìn đồng/người'; dữ liệu ghi dạo làng miễn phí"),
    ("qwen2.5:3b", "Q155"): (0.5, "dung_mot_phan", "Chỉ nêu 1 trong 2 nguồn giờ mở cửa (7h–17h)"),
    ("qwen2.5:3b", "Q177"): (0, "tu_choi_nham", "Dữ liệu có thông tin: chưa công bố phí, phố cổ là không gian mở"),
    ("llama3.2:3b", "Q026"): (0, "tu_choi_nham", "Dữ liệu có thông tin 'không áp dụng, tùy nhà vườn'"),
    ("llama3.2:3b", "Q018"): (0, "hieu_sai_cau_hoi", "'Bán bao nhiêu' bị hiểu là số lượng thay vì giá"),
    ("llama3.2:3b", "Q021"): (1, "cham_tu_dong_bo_sot", "'Không phải mua vé' đúng với 'vào chợ miễn phí'"),
    ("llama3.2:3b", "Q036"): (0, "tu_choi_nham", "Nói không có thông tin; dữ liệu ghi dạo làng miễn phí"),
    ("llama3.2:3b", "Q155"): (0.5, "dung_mot_phan", "Nêu 7h–16h, ghép lẫn hai nguồn"),
    ("llama3.2:3b", "Q185"): (0.5, "dung_mot_phan", "Nêu ngày âm lịch hằng năm, thiếu ngày dương lịch năm 2026"),
    ("llama3.2:3b", "Q177"): (0.5, "khang_dinh_qua_muc", "Khẳng định 'không mất phí' trong khi dữ liệu chỉ ghi chưa có công bố"),
    ("gemma2:2b", "Q130"): (1, "cham_tu_dong_bo_sot", "Liệt kê đúng phong tục có trong dữ liệu"),
    ("gemma2:2b", "Q145"): (0, "tra_loi_rong", "Mô hình trả về chuỗi rỗng"),
    ("gemma2:2b", "Q026"): (1, "cham_tu_dong_bo_sot", "'Không có giờ mở cửa chính thức' đúng"),
    ("gemma2:2b", "Q038"): (0, "bia_thong_tin", "Nói đóng cửa 17h00; dữ liệu ghi 17:30"),
    ("gemma2:2b", "Q033"): (1, "cham_tu_dong_bo_sot", "'Miễn phí' đúng với 'không bán vé'"),
    ("gemma2:2b", "Q036"): (1, "cham_tu_dong_bo_sot", "'Miễn phí' đúng với dữ liệu"),
    ("gemma2:2b", "Q155"): (0.5, "dung_mot_phan", "Chỉ nêu 1 trong 2 nguồn giờ mở cửa (8h–16h)"),
    ("gemma2:2b", "Q177"): (0.5, "khang_dinh_qua_muc", "Khẳng định 'miễn phí' trong khi dữ liệu chỉ ghi chưa có công bố"),
}


def main():
    d = pd.read_csv(REPORT_DIR / "07_generation_answers_day_du.csv", encoding="utf-8-sig", dtype={"qid": str})
    rag = d[d["che_do"] == "co_rag"].copy()
    wrong = rag[(rag["dung_tu_dong"] == 0) & (rag["loai"] != "ngoai_pham_vi")]
    missing = {(r.llm, r.qid) for r in wrong.itertuples()} - set(R)
    assert not missing, f"Chưa chấm tay: {missing}"

    rv = pd.DataFrame([{"llm": k[0], "qid": k[1], "cham_tay_dung": v[0], "loai_loi": v[1], "ghi_chu": v[2]}
                       for k, v in R.items()])
    rv = rv.merge(rag[["llm", "qid", "loai", "cau_hoi", "dap_an_tham_chieu", "tra_loi"]], on=["llm", "qid"], how="left")
    rv.to_csv(REPORT_DIR / "07_cham_tay_day_du.csv", index=False, encoding="utf-8-sig")

    rag = rag.merge(rv[["llm", "qid", "cham_tay_dung"]], on=["llm", "qid"], how="left", suffixes=("_cu", ""))
    rag["dung_dieu_chinh"] = rag["cham_tay_dung"].fillna(rag["dung_tu_dong"])

    rows = []
    for llm, g in d.groupby("llm"):
        gr = rag[(rag["llm"] == llm) & (rag["loai"] != "ngoai_pham_vi")]
        go = d[(d["llm"] == llm) & (d["loai"] == "ngoai_pham_vi")]
        gn = d[(d["llm"] == llm) & (d["che_do"] == "khong_rag") & (d["loai"] != "ngoai_pham_vi")]
        rows.append({
            "llm": llm,
            "dung_co_rag_tu_dong": round(gr["dung_tu_dong"].mean(), 3),
            "dung_co_rag_dieu_chinh": round(gr["dung_dieu_chinh"].mean(), 3),
            "dung_khong_rag_tu_dong": round(gn["dung_tu_dong"].mean(), 3),
            "tu_choi_dung_ngoai_pham_vi_co_rag": round(go[go["che_do"] == "co_rag"]["tu_choi"].mean(), 3),
            "tu_choi_dung_ngoai_pham_vi_khong_rag": round(go[go["che_do"] == "khong_rag"]["tu_choi"].mean(), 3),
            "giay_tb_co_rag": round((gr["giay"] + pd.to_numeric(gr["giay_tim_kiem"], errors="coerce").fillna(0)).mean(), 1),
            "giay_tb_khong_rag": round(gn["giay"].mean(), 1),
        })
    sm = pd.DataFrame(rows)
    sm.to_csv(REPORT_DIR / "07_final_summary.csv", index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))
    print("\nLoại lỗi (sau chấm tay, chỉ tính câu có điểm < 1):")
    print(rv[rv["cham_tay_dung"] < 1].groupby(["loai_loi", "llm"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
