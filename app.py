"""Giao diện demo chatbot hỏi đáp du lịch và văn hóa (Streamlit).

Chạy:  streamlit run app.py
"""
import time

import streamlit as st

from rag.config import DEFAULT_RERANK, EMBED_MODELS, INDEX_DIR, LLM_MODELS, RERANK_MIN_SCORE, TOP_K
from rag.generator import DEFAULT_PROMPT, PROMPTS, REFUSAL
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="Chatbot du lịch văn hóa Localis", page_icon="🧭", layout="wide")


def available_indexes():
    out = []
    for d in sorted(INDEX_DIR.glob("*__*")):
        if (d / "faiss.index").exists():
            m, s = d.name.split("__")
            if m in EMBED_MODELS:
                out.append((m, s))
    return out


def installed_llms():
    try:
        import ollama
        names = {m["model"] for m in ollama.list()["models"]}
    except Exception:
        return []
    return [m for m in LLM_MODELS if m in names] + sorted(n for n in names if n not in LLM_MODELS)


@st.cache_resource(show_spinner="Đang tải chỉ mục, mô hình embedding và re-ranker...")
def load_pipeline(embed, strategy):
    try:
        return RAGPipeline(embed, strategy, rerank=DEFAULT_RERANK)
    except Exception as e:  # chưa tải re-ranker: vẫn chạy được, chỉ tắt xếp hạng lại
        st.warning(f"Không nạp được re-ranker ({e}); chạy không xếp hạng lại.")
        return RAGPipeline(embed, strategy, rerank=None)


# ---------------- thanh bên ----------------
st.sidebar.title("Cấu hình")
idx = available_indexes()
if not idx:
    st.error("Chưa có chỉ mục. Hãy chạy: python scripts/04_build_index.py")
    st.stop()
labels = [f"{m} / {s}" for m, s in idx]
default_i = labels.index("e5-base / section") if "e5-base / section" in labels else 0
choice = st.sidebar.selectbox("Chỉ mục (embedding / cách chia đoạn)", labels, index=default_i)
embed, strategy = idx[labels.index(choice)]

llms = installed_llms()
if not llms:
    st.error("Không kết nối được Ollama. Hãy mở ứng dụng Ollama rồi tải mô hình: ollama pull qwen2.5:3b")
    st.stop()
llm = st.sidebar.selectbox("Mô hình ngôn ngữ (Ollama)", llms)
method = st.sidebar.radio("Phương pháp tìm kiếm", ["hybrid", "semantic", "bm25"],
                          format_func={"hybrid": "Lai (ngữ nghĩa + BM25)", "semantic": "Ngữ nghĩa", "bm25": "Từ khóa BM25"}.get)
k = st.sidebar.slider("Số đoạn ngữ cảnh (top-k)", 1, 10, TOP_K)
use_filters = st.sidebar.checkbox("Tự lọc theo tỉnh/thành nhắc trong câu hỏi", False,
                                  help="Đánh giá cho thấy tắt lọc cho kết quả tốt hơn (mục 4.1)")
use_rag = st.sidebar.checkbox("Dùng RAG (bỏ chọn để so sánh LLM trả lời trực tiếp)", True)

pipe = load_pipeline(embed, strategy)
use_rerank = st.sidebar.checkbox("Xếp hạng lại bằng re-ranker", pipe.reranker is not None,
                                 disabled=pipe.reranker is None,
                                 help="Chính xác hơn với câu hỏi chi tiết nhưng chậm thêm khoảng 20 giây trên CPU")
use_threshold = st.sidebar.checkbox(f"Từ chối khi điểm liên quan < {RERANK_MIN_SCORE}", True,
                                    disabled=not use_rerank,
                                    help="Câu hỏi không có dữ liệu liên quan sẽ được từ chối ngay, không gọi LLM")
prompt = st.sidebar.selectbox("Phiên bản prompt", list(PROMPTS), index=list(PROMPTS).index(DEFAULT_PROMPT))
pipe.set_llm(llm, prompt)
pipe.min_score = RERANK_MIN_SCORE if use_threshold else None

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(pipe.places):,} địa điểm · {len(pipe.retriever.chunks):,} đoạn văn bản")

# ---------------- khung chat ----------------
st.title("🧭 Hỏi đáp du lịch và văn hóa Việt Nam")
st.caption("Trả lời dựa trên dữ liệu địa điểm của Localis · mô hình chạy cục bộ, miễn phí")

examples = [
    "Lễ hội nào ở miền Tây diễn ra vào tháng 8 âm lịch?",
    "Bảo tàng Áo Dài mở cửa mấy giờ, giá vé bao nhiêu?",
    "Gợi ý vài làng nghề truyền thống ở Huế",
    "Đi Lâm Đồng mùa nào đẹp nhất?",
]
cols = st.columns(len(examples))
for c, ex in zip(cols, examples):
    if c.button(ex, use_container_width=True):
        st.session_state["pending"] = ex

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if turn["sources"]:
            with st.expander(f"Nguồn ({len(turn['sources'])})"):
                for s in turn["sources"]:
                    st.markdown(f"**[{s['so']}] {s['title']}** · {s['region'] or ''} · {s['category'] or ''}")

q = st.chat_input("Nhập câu hỏi về địa điểm, lễ hội, ẩm thực...") or st.session_state.pop("pending", None)
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        chunks, filters, sources, t_ret = None, {}, [], 0.0
        if use_rag:
            chunks, filters, t_ret = pipe.retrieve(q, k=k, method=method, use_filters=use_filters,
                                                   use_rerank=use_rerank)
            sources = pipe.sources(chunks)
        t0 = time.time()
        blocked = use_rag and pipe.below_threshold(chunks, use_rerank)
        if blocked:
            answer = REFUSAL
            st.write(answer)
            sources = []
        else:
            answer = st.write_stream(pipe.generator.stream(q, chunks))
        t_gen = time.time() - t0
        st.caption(f"Tìm kiếm {t_ret:.1f}s · sinh câu trả lời {t_gen:.1f}s"
                   + (f" · điểm liên quan cao nhất {chunks[0]['score'] if chunks else 0:.2f} < ngưỡng, không gọi LLM"
                      if blocked else "")
                   + (f" · lọc: {filters['region']}" if filters.get("region") else ""))
        if sources:
            with st.expander(f"Nguồn ({len(sources)})", expanded=True):
                for s in sources:
                    links = " · ".join(x for x in (
                        f"[Bản đồ]({s['map_url']})" if s.get("map_url") else "",
                        f"[Bài gốc]({s['source_url']})" if s.get("source_url") else "") if x)
                    st.markdown(f"**[{s['so']}] {s['title']}** · {s['region'] or ''} · {s['category'] or ''}  \n"
                                f"{s.get('address') or ''} {links}")
                    with st.popover("Xem đoạn văn bản"):
                        st.text(s["text"])
                pts = [{"lat": s["lat"], "lon": s["lng"]} for s in sources if s.get("lat") and s.get("lng")]
                if pts:
                    st.map(pts, size=200, zoom=None)
    st.session_state.history.append({"q": q, "a": answer, "sources": sources})
