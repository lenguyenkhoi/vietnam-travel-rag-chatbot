# Chatbot hỏi đáp du lịch và văn hóa Việt Nam (RAG, dữ liệu Localis)

Hệ thống hỏi đáp tiếng Việt về địa điểm, lễ hội, ẩm thực, làng nghề..., trả lời dựa trên dữ liệu JSON của Localis và có trích nguồn.
Toàn bộ chạy trên máy cá nhân, **miễn phí**: mô hình embedding từ Hugging Face, FAISS, BM25 và mô hình ngôn ngữ chạy qua Ollama.

## Kiến trúc

```
Chuẩn bị (chạy 1 lần)
  file JSON Localis ─► gom, làm sạch, loại trùng ─► chia đoạn + gắn tỉnh/danh mục ─► embedding ─► chỉ mục FAISS + BM25

Hỏi đáp
  câu hỏi ─► nhận diện tỉnh/thành ─► tìm kiếm lai (FAISS + BM25, gộp bằng RRF) ─► top-k đoạn
          ─► prompt "chỉ dùng ngữ cảnh" ─► LLM (Ollama) ─► câu trả lời + nguồn [1], [2]...
```

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `rag/config.py` | Cấu hình: danh sách mô hình, top-k, đường dẫn |
| `rag/chunking.py` | 3 chiến lược chia đoạn: `doc`, `fixed`, `section` |
| `rag/embedder.py` | Mô hình embedding (e5-base, bge-m3, vn-sbert) |
| `rag/retriever.py` | Tìm kiếm ngữ nghĩa, BM25, lai (RRF), lọc theo tỉnh |
| `rag/generator.py` | Prompt và gọi LLM qua Ollama |
| `rag/pipeline.py` | Ghép truy xuất + sinh câu trả lời |
| `app.py` | Giao diện demo Streamlit |
| `scripts/01_build_dataset.py` | Gom, làm sạch, loại trùng dữ liệu → `data/places.jsonl` |
| `scripts/02_eda.py` | Phân tích khám phá dữ liệu → `reports/figures/` |
| `scripts/03_chunk_stats.py` | Thống kê 3 cách chia đoạn |
| `scripts/04_build_index.py` | Tạo embedding và chỉ mục |
| `scripts/05_make_testset.py` | Sinh bộ câu hỏi kiểm thử nháp |
| `scripts/06_eval_retrieval.py` | Đánh giá tìm kiếm: Recall@k, MRR, nDCG |
| `scripts/07_eval_generation.py` | Đánh giá câu trả lời của các LLM |

Các script chỉ **đọc** file JSON gốc của công ty, không sửa dữ liệu gốc.

## Dữ liệu và kết quả không công khai

Dữ liệu địa điểm là dữ liệu nội bộ của Công ty Cổ phần Công nghệ Du lịch Localis, chỉ được dùng cho mục đích học tập
và thực hiện báo cáo thực tập. Vì vậy kho mã này **không** chứa:

- dữ liệu gốc và dữ liệu đã xử lý (`data/`), chỉ mục FAISS/BM25 (`indexes/`);
- bộ câu hỏi kiểm thử (`eval/`), câu trả lời của mô hình và kết quả chấm từng câu;
- bản đồ, thống kê và biểu đồ mô tả dữ liệu (`reports/02_*`).

Thư mục `reports/` chỉ giữ các kết quả tổng hợp (chỉ số trung bình) và biểu đồ đánh giá dùng trong báo cáo.

## Cài đặt

1. Cài thư viện Python: `pip install -r requirements.txt`
2. Cài [Ollama](https://ollama.com) rồi tải mô hình: `ollama pull qwen2.5:3b`
   (tùy chọn để so sánh: `llama3.2:3b`, `gemma2:2b`, `qwen2.5:7b`)

## Chạy từng bước

```bash
cd chatbot-rag
python scripts/01_build_dataset.py                      # ~1 phút
python scripts/02_eda.py
python scripts/03_chunk_stats.py
python scripts/04_build_index.py --model e5-base --strategy section   # CPU: khá lâu; bị ngắt thì chạy lại sẽ làm tiếp
python scripts/05_make_testset.py                       # thêm --llm 40 để có câu hỏi tự nhiên
#   -> mở eval/testset_draft.csv, rà soát, lưu thành eval/testset.csv
python scripts/06_eval_retrieval.py
python scripts/07_eval_generation.py --n 40 --no-rag-baseline
streamlit run app.py
```

Để so sánh các mô hình embedding và cách chia đoạn, chạy thêm bước 4 với `--model vn-sbert`, `--model bge-m3`
hoặc `--strategy doc` / `--strategy fixed`, rồi chạy lại bước 6.

## Lưu ý về máy cấu hình thấp

- Máy RAM 16GB: nên đóng bớt Chrome/ứng dụng khác khi tạo embedding và dùng `--batch 8`.
- `bge-m3` (2,3GB) và `qwen2.5:7b` (4,7GB) tốn nhiều RAM và ổ đĩa; có thể chạy trên Google Colab (GPU miễn phí).
- Ảnh trong dữ liệu lấy từ báo chí nên hệ thống chỉ dùng văn bản.
