"""Bước 5b: Áp dụng kết quả rà soát bộ câu hỏi nháp -> eval/testset.csv

Rà soát ngày 17/11/2026, thực hiện với sự hỗ trợ của AI (Claude), sinh viên kiểm tra lại.
Mỗi câu: GIỮ / SỬA (viết lại câu hỏi hoặc đáp án) / XÓA, kèm lý do.
Cột dien_dat_lai = "co" nếu câu hỏi đã được viết lại bằng lời khác (không chép nguyên văn dữ liệu),
dùng để so sánh kết quả tìm kiếm giữa câu chép nguyên văn và câu diễn đạt lại (mục 4.1).

Chạy:  python scripts/05b_apply_review.py
Đầu ra: eval/testset.csv, eval/review_decisions.csv (nhật ký rà soát), reports/05_review_summary.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import EVAL_DIR, REPORT_DIR  # noqa: E402

# qid: (quyết định, câu hỏi mới hoặc None, đáp án mới hoặc None, lý do)
X, S = "xoa", "sua"
D = {
    # ---------------- thong_tin ----------------
    "Q001": (S, "Lululola Show ở Đà Lạt nằm ở địa chỉ nào?", None, "Diễn đạt tự nhiên hơn"),
    "Q002": (S, "Khu du lịch Lung Cột Cầu ở Cần Thơ đón khách từ mấy giờ đến mấy giờ?", None, "Diễn đạt tự nhiên hơn"),
    "Q003": (S, "Vào Bảo tàng Tôn Đức Thắng có mất tiền vé không?", None, "Diễn đạt tự nhiên hơn"),
    "Q005": (S, "Lễ hội đền Đô ở Bắc Ninh diễn ra vào ngày nào?", None, "Lễ hội không có 'giờ mở cửa'; đáp án là ngày tổ chức"),
    "Q006": (S, "Đi Vịnh Hạ Long thì vé thắng cảnh bao nhiêu tiền?", None, "Diễn đạt tự nhiên hơn"),
    "Q008": (S, "Mấy giờ thì ghé làng bánh tráng An Ngãi được?", None, "Diễn đạt tự nhiên hơn"),
    "Q009": (X, None, None, "'Thông tin hữu ích' là trang số khẩn cấp, không phải địa điểm; hỏi giá vé vô nghĩa"),
    "Q012": (X, None, None, "Di sản phi vật thể (hát dân ca), không có giá vé"),
    "Q013": (S, "Biển Hải Ninh thuộc xã nào của Quảng Trị?", None, "Diễn đạt tự nhiên hơn"),
    "Q015": (S, "Tham quan đình Phú Lễ ở Vĩnh Long có tốn phí không?", None, "Diễn đạt tự nhiên hơn"),
    "Q018": (S, "Hột vịt lộn Hoàng Vân bán bao nhiêu tiền một quả?", None, "Quán ăn: hỏi giá món thay vì 'giá vé tham quan'"),
    "Q020": (S, "Chợ Bà Rịa họp từ mấy giờ?", None, "Diễn đạt tự nhiên hơn"),
    "Q021": (S, "Vào chợ Đầm Nha Trang có phải mua vé không?", None, "Diễn đạt tự nhiên hơn"),
    "Q023": (X, None, None, "Lễ hội: 'giờ mở cửa' không áp dụng, đáp án không có thông tin"),
    "Q024": (S, "Đậu hũ thối Hồ Nam ở Sài Gòn giá bao nhiêu một phần?", None, "Quán ăn: hỏi giá món thay vì 'giá vé tham quan'"),
    "Q030": (S, "Bảo tàng tỉnh Lai Châu có thu vé không?", None, "Diễn đạt tự nhiên hơn"),
    "Q032": (S, "Bệnh viện Đa khoa tỉnh Tuyên Quang khám bệnh vào giờ nào?", None, "Bệnh viện: 'giờ khám' thay vì 'giờ mở cửa'"),
    "Q035": (S, "Chợ đêm Đà Lạt đông nhất vào khung giờ nào?", None, "Diễn đạt tự nhiên hơn"),
    "Q038": (S, "Bảo tàng Áo Dài mấy giờ đóng cửa?", None, "Diễn đạt tự nhiên hơn"),
    "Q039": (S, "Một tô mì bò ở Jeffrey's Kitchen giá khoảng bao nhiêu?", None, "Quán ăn: hỏi giá món thay vì 'giá vé tham quan'"),
    "Q040": (S, "Lăng Ông Bà Chiểu nằm trên đường nào?", None, "Diễn đạt tự nhiên hơn"),
    "Q041": (X, None, None, "Lễ hội: 'giờ mở cửa' không áp dụng, đáp án không có thông tin"),
    "Q045": (S, "Vé vào Thung lũng Vàng ở Đà Lạt cho người lớn và trẻ em là bao nhiêu?", None, "Diễn đạt tự nhiên hơn"),
    "Q051": (S, "Vào chùa Thiên Mụ có mất phí không?", None, "Diễn đạt tự nhiên hơn"),
    "Q054": (X, None, None, "Đáp án là chi phí đi lại (máy bay, tàu), không phải giá vé tham quan"),
    "Q055": (S, "Chùa Linh Sơn Tiên Thạch nằm ở đâu trên núi Bà Đen?", None, "Diễn đạt tự nhiên hơn"),
    "Q057": (S, "Ăn ở Ba Hí Sài Gòn tốn khoảng bao nhiêu?", None, "Quán ăn: hỏi giá món thay vì 'giá vé tham quan'"),
    # ---------------- mo_ta_an: Q061-Q090 viết lại, Q091-Q120 giữ nguyên văn ----------------
    "Q061": (S, "Nhóm nào ở Sài Gòn đứng sau lễ hội đầu tiên ở Việt Nam về không gian công cộng?", None, "Viết lại bằng lời khác"),
    "Q062": (S, "Ở TP.HCM có bảo tàng nào trưng bày về hậu quả chiến tranh?", None, "Viết lại bằng lời khác"),
    "Q063": (S, "Ở Đà Lạt có khu vui chơi nào dưới chân núi Langbiang mang văn hóa người K'Ho?", None, "Viết lại bằng lời khác"),
    "Q064": (S, "Làng người Jrai ở Ia Ly (Gia Lai) nổi tiếng nghề dệt thổ cẩm tên là gì?", None, "Viết lại bằng lời khác"),
    "Q065": (S, "Nông trường bò sữa gắn với thương hiệu sữa Mộc Châu là nơi nào?", None, "Viết lại bằng lời khác"),
    "Q066": (S, "Ngọn núi nào ở thành phố Ninh Bình có nhiều bài thơ khắc trên vách đá?", None, "Viết lại bằng lời khác"),
    "Q067": (S, "Lăng vua nào ở Huế nằm giữa đồng lúa và quay về hướng tây bắc?", None, "Viết lại bằng lời khác"),
    "Q068": (S, "Ở Biên Hòa có ngôi chùa nào thờ tượng thần Vishnu của người Chăm?", None, "Viết lại bằng lời khác"),
    "Q069": (S, "Tháng Giêng đi hội chùa ở Chí Linh thì có lễ hội nào?", None, "Viết lại bằng lời khác"),
    "Q070": (S, "Ngôi chùa Khmer lâu đời nhất ở Sóc Trăng là chùa nào?", None, "Viết lại bằng lời khác"),
    "Q071": (S, "Lễ hội nào ở TP.HCM tôn vinh áo dài vào tháng 3 hằng năm?", None, "Viết lại; đây là lễ hội chứ không phải địa điểm"),
    "Q072": (S, "Trên lòng hồ thủy điện Hòa Bình có chợ nổi nào họp sáng Chủ nhật?", None, "Viết lại bằng lời khác"),
    "Q073": (S, "Ở Cao Bằng, liên hoan dân ca và chiếu phim của tỉnh thường tổ chức ở đâu?", None, "Viết lại bằng lời khác"),
    "Q074": (S, "Có ngôi chùa nào nằm trên đường từ Ayun Pa đi Krông Pa không?", None, "Viết lại bằng lời khác"),
    "Q075": (S, "Ở Cát Bà có làng chài nào chỉ đi được bằng thuyền hoặc đường rừng?", None, "Viết lại bằng lời khác"),
    "Q076": (S, "Ngôi chùa nào ở Tiền Giang cũ từng là nơi vua Gia Long trốn tránh?", None, "Viết lại bằng lời khác"),
    "Q077": (S, "Chợ đầu mối nào ở TP.HCM nhập hàng nghìn tấn nông sản mỗi đêm?", None, "Viết lại bằng lời khác"),
    "Q078": (S, "Đi Tây Ninh muốn mua muối tôm, bánh tráng thì ghé chợ nào?", None, "Viết lại bằng lời khác"),
    "Q079": (S, "Ở Côn Đảo muốn mua hải sản buổi tối thì ra chợ nào?", None, "Viết lại bằng lời khác"),
    "Q080": (S, "Ở Mũi Né có con suối nào nước màu đỏ cam để lội chân trần?", None, "Viết lại bằng lời khác"),
    "Q081": (S, "Ở Gia Nghĩa có nơi nào trưng bày về âm thanh của đá, gió, nước không?", None, "Viết lại bằng lời khác"),
    "Q082": (S, "Muốn ngắm hoàng hôn trên biển ở Hà Tiên thì nên ra bãi nào?", None, "Viết lại bằng lời khác"),
    "Q083": (S, "Chợ nào ở Sài Gòn có tên bắt nguồn từ 'Phúc Lâm Thị'?", None, "Viết lại bằng lời khác"),
    "Q084": (S, "Lễ hội nào ở Hải Phòng có nghi thức các chức sắc thề giữ liêm chính?", None, "Viết lại; đây là lễ hội chứ không phải địa điểm"),
    "Q085": (X, None, None, "Phần giới thiệu là ghi chú nội bộ ('chưa tìm được nguồn độc lập'), không phải mô tả địa điểm"),
    "Q086": (S, "Muốn ăn bánh canh Trảng Bàng đúng gốc thì nên ghé chợ nào?", None, "Viết lại bằng lời khác"),
    "Q087": (S, "Sân khấu do ca sĩ Thanh Bùi lập ở Quận 3 tên là gì?", None, "Viết lại bằng lời khác"),
    "Q088": (S, "Bảo tàng quân sự nào ở TP.HCM giới thiệu chín tỉnh thành Đông Nam Bộ?", None, "Viết lại bằng lời khác"),
    "Q089": (S, "Ở Quảng Ninh còn nơi nào xem được cọc gỗ của trận Bạch Đằng năm 1288?", None, "Viết lại bằng lời khác"),
    "Q090": (S, "Nhà thờ nào ở Gò Vấp có kiến trúc Byzantine với mái vòm?", None, "Viết lại bằng lời khác"),
    # ---------------- goi_y: viết lại theo cách du khách hỏi ----------------
    "Q121": (S, "Lào Cai có những di tích, di sản nào đáng ghé?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q122": (S, "Đến Hải Phòng nên đi chợ nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q123": (S, "Đi Lâm Đồng nên mua đặc sản gì làm quà?", None, "Mục này là sản phẩm làm quà, không phải địa điểm"),
    "Q124": (S, "Thanh Hóa có những cảnh đẹp thiên nhiên nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q125": (S, "Hải Phòng có những lễ hội nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q126": (S, "Quảng Ninh có những chùa, di tích nào nên đến?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q127": (S, "Đi Sơn La ngắm cảnh thiên nhiên ở đâu?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q128": (S, "Ở Điện Biên có những chợ nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q129": (S, "Gần Đà Nẵng có làng nghề truyền thống nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q130": (S, "Người Tày ở Nghĩa Đô (Lào Cai) có những phong tục, nét văn hóa gì?", None, "Mục này là phong tục, không phải địa điểm"),
    "Q131": (S, "Hải Phòng có bãi biển, cảnh đẹp nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q132": (S, "Quảng Ngãi có cảnh đẹp thiên nhiên nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q133": (S, "Ở Vĩnh Long có làng nghề nào tham quan được?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q134": (S, "Đà Nẵng có những điểm thiên nhiên nào nên đi?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q135": (S, "Phú Thọ có làng nghề truyền thống nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q136": (S, "Gợi ý vài di tích, không gian di sản ở Hà Nội.", None, "Bỏ khuôn 'nhóm ...'"),
    "Q137": (S, "Đi chợ ở Đà Lạt thì nên ghé đâu?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q138": (S, "Đà Nẵng có chợ nào nổi tiếng?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q139": (S, "Huế có những bảo tàng nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q140": (S, "Sơn La có di tích lịch sử nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q141": (S, "Ở Đà Nẵng có làng du lịch cộng đồng nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q142": (S, "Hà Nội có không gian nghệ thuật nào hay?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q143": (S, "Quảng Trị có những cảnh thiên nhiên nào đẹp?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q144": (S, "Đồng Nai có những chợ nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q145": (S, "Cao Bằng có di tích nào nên ghé?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q146": (S, "Quảng Trị có bản làng du lịch cộng đồng nào?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q147": (S, "Bắc Ninh có ngôi chùa, di tích cổ nào nổi tiếng?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q148": (S, "Cao Bằng có cảnh thiên nhiên nào đẹp?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q149": (S, "Bắc Ninh có hồ hay cảnh thiên nhiên nào để đi chơi?", None, "Bỏ khuôn 'nhóm ...'"),
    "Q150": (S, "An Giang có những ngôi chùa, di tích nào?", None, "Bỏ khuôn 'nhóm ...'"),
    # ---------------- tu_nhien ----------------
    "Q152": (S, "Ở Làng Văn hóa – Du lịch các dân tộc Việt Nam có thuê xe đạp đi tham quan được không?", None,
             "Câu hỏi thiếu tên địa điểm"),
    "Q153": (X, None, None, "Câu hỏi phụ thuộc đoạn văn ('đoạn thông tin'), không tự nhiên"),
    "Q154": (X, None, None, "DỮ LIỆU MÂU THUẪN: trường address ghi Tùng Thiện Vương, P. Phú Định; rag_text ghi Phạm Thế Hiển, P. Chánh Hưng"),
    "Q156": (S, "Từ Hà Nội đi núi Trầm mất bao lâu?", None, "Câu gốc sai ngữ pháp"),
    "Q158": (X, None, None, "Câu hỏi lặp lại đáp án; nội dung nguồn là mô tả tự sinh ('theo tên chưa rõ chuyên khoa')"),
    "Q159": (S, "Gần Khu bảo tồn thiên nhiên Bắc Hướng Hóa có điểm nào kết hợp tham quan?", None, "Diễn đạt rõ hơn"),
    "Q160": (S, "Miệng núi lửa Thới Lới ở Lý Sơn hình thành cách đây bao lâu?", None, "Câu gốc gộp 2 ý, diễn đạt chưa đúng"),
    "Q161": (None, None, "09:00-18:00 (giờ tham khảo, cần xác nhận)", "Đáp án gốc nhắc 'nguồn bậc 3' (thuật ngữ nội bộ)"),
    "Q163": (S, "Căn cứ Bộ Chỉ huy Miền Tà Thiết được tôn tạo năm nào và kinh phí bao nhiêu?",
             "Năm 2015, tổng mức đầu tư hơn 300 tỷ đồng", "Câu gốc mơ hồ; đã đối chiếu đáp án với dữ liệu"),
    "Q164": (S, "Chợ Cao Lãnh nằm ở địa chỉ nào?", None, "Diễn đạt tự nhiên hơn"),
    "Q165": (S, "Chùa Bửu Sơn ở Biên Hòa mở cửa giờ nào?",
             "Google Maps ghi mở cả ngày (chưa có nguồn chính thức xác nhận)", "Đáp án nêu rõ mức độ tin cậy"),
    "Q167": (X, None, None, "Đáp án không chứa địa chỉ; đoạn nguồn chỉ là danh sách nguồn tham khảo"),
    "Q168": (S, "Lễ hội Hoa đăng Thiên Cấm Sơn ở núi Cấm tổ chức khi nào?", None, "Câu gốc quá chung"),
    "Q170": (X, None, None, "Dữ liệu không có địa chỉ cụ thể; đáp án là lời nhắc kiểm tra lại, khó chấm"),
    "Q171": (S, "Gần Di tích Ban Dân y Trung ương Cục miền Nam (Tây Ninh) có điểm nào kết hợp tham quan?", None,
             "Câu hỏi thiếu tên địa điểm ('địa điểm này')"),
    "Q172": (S, "Đến Bến đá Kỳ Cùng ở Lạng Sơn có cần đặt chỗ trước không?",
             "Không cần đặt chỗ trước, khách có thể đến tự do", "Câu gốc mơ hồ"),
    "Q174": (S, "Vé vào Thạch Động ở Hà Tiên bao nhiêu tiền?",
             "Khoảng 20.000 VND/người (dao động 5.000–20.000 VND)", "Diễn đạt tự nhiên hơn; đáp án theo dữ liệu"),
    "Q175": (None, None, "Thôn Đèo De, xã Phú Đình, tỉnh Thái Nguyên", "Đáp án gốc viết sai tên xã (Phú Đinh)"),
    "Q176": (X, None, None, "Đáp án sai: nguồn chỉ ghi 'cần CTV xác nhận', không khẳng định có hỗ trợ"),
    "Q177": (S, "Tham quan phố cổ Đồng Văn có mất phí không?",
             "Chưa có công bố chính thức về phí tham quan; phố cổ là không gian mở", "Câu gốc mơ hồ"),
    "Q180": (S, "Từ cồn cát Quang Phú có đi tiếp ra biển Nhật Lệ được không?",
             "Được, hai nơi nằm cùng tuyến đường Trương Pháp", "Câu gốc sai ngữ pháp"),
    "Q181": (S, "Danh thắng Kẽm Trống có gì đặc biệt và nên đi mùa nào?", None, "Diễn đạt rõ hơn"),
    "Q188": (X, None, None, "Dữ liệu không xác minh được; đáp án khó chấm"),
    "Q189": (S, "Kho nhiên liệu VK98 hiện có đón khách không?",
             "Đã tạm dừng đón khách từ tháng 4/2025; chưa xác minh được đã mở lại hay chưa",
             "Đáp án gốc không trả lời câu hỏi"),
}


def main():
    draft = pd.read_csv(EVAL_DIR / "testset_draft.csv", encoding="utf-8-sig", dtype=str).fillna("")
    unknown = set(D) - set(draft["qid"])
    assert not unknown, f"qid không có trong bản nháp: {unknown}"
    rows, log = [], []
    for _, r in draft.iterrows():
        act, q_new, a_new, why = D.get(r["qid"], ("giu", None, None, ""))
        if act is None:  # chỉ sửa đáp án
            act = "sua"
        log.append({"qid": r["qid"], "loai": r["loai"], "quyet_dinh": act, "cau_hoi_cu": r["cau_hoi"],
                    "cau_hoi_moi": q_new or "", "dap_an_moi": a_new or "", "ly_do": why})
        if act == X:
            continue
        rr = r.to_dict()
        rewritten = bool(q_new)
        if q_new:
            rr["cau_hoi"] = q_new
        if a_new:
            rr["dap_an_tham_chieu"] = a_new
        # câu tự nhiên do LLM viết, ngoài phạm vi: vốn không chép nguyên văn dữ liệu
        rr["dien_dat_lai"] = "co" if rewritten or r["loai"] in ("tu_nhien", "ngoai_pham_vi") else "khong"
        rr["da_kiem_tra"] = "AI + SV"
        rows.append(rr)

    ts = pd.DataFrame(rows)
    ts.to_csv(EVAL_DIR / "testset.csv", index=False, encoding="utf-8-sig")
    lg = pd.DataFrame(log)
    lg.to_csv(EVAL_DIR / "review_decisions.csv", index=False, encoding="utf-8-sig")
    summ = lg.pivot_table(index="loai", columns="quyet_dinh", values="qid", aggfunc="count", fill_value=0)
    summ["tong_nhap"] = summ.sum(axis=1)
    summ["con_lai"] = summ["tong_nhap"] - summ.get(X, 0)
    summ.loc["TỔNG"] = summ.sum()
    summ.to_csv(REPORT_DIR / "05_review_summary.csv", encoding="utf-8-sig")
    print(summ.to_string())
    print("\nCâu viết lại / nguyên văn trong testset:", ts["dien_dat_lai"].value_counts().to_dict())
    print("Đã ghi eval/testset.csv,", len(ts), "câu")


if __name__ == "__main__":
    main()
