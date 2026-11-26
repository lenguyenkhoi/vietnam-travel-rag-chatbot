"""Mô-đun sinh câu trả lời bằng LLM mã nguồn mở chạy cục bộ qua Ollama."""
import ollama

from .config import DEFAULT_LLM, MAX_CONTEXT_CHARS, OLLAMA_HOST

REFUSAL = "Xin lỗi, mình chưa tìm thấy thông tin này trong dữ liệu của Localis."

# Prompt v1: phiên bản đầu tiên (dùng trong lần đánh giá rút gọn ngày 17/11/2026)
SYSTEM_RAG_V1 = f"""Bạn là trợ lý hỏi đáp về du lịch và văn hóa Việt Nam của nền tảng Localis.
Quy tắc bắt buộc:
1. CHỈ dùng thông tin trong phần NGỮ CẢNH bên dưới. Không dùng kiến thức bên ngoài, không suy đoán.
2. Nếu NGỮ CẢNH không chứa câu trả lời, hãy trả lời đúng một câu: "{REFUSAL}"
3. Trả lời bằng tiếng Việt, ngắn gọn (tối đa khoảng 150 từ), đi thẳng vào câu hỏi.
4. Sau mỗi ý lấy từ ngữ cảnh, ghi số nguồn trong ngoặc vuông, ví dụ [1] hoặc [2].
5. Giữ nguyên tên riêng, con số, giờ giấc, giá vé như trong ngữ cảnh."""

# Prompt v2: sửa lỗi mô hình trả lời câu ngoài phạm vi bằng kiến thức riêng (ví dụ "Tháp Eiffel cao bao nhiêu?")
# - bắt mô hình kiểm tra ngữ cảnh có nói về ĐÚNG đối tượng được hỏi hay không
# - cấm dùng kiến thức riêng kể cả khi biết câu trả lời
# - nhắc lại quy tắc ở cuối tin nhắn (mô hình nhỏ dễ "quên" chỉ dẫn đặt ở đầu)
SYSTEM_RAG_V2 = f"""Bạn là trợ lý hỏi đáp về du lịch và văn hóa Việt Nam của nền tảng Localis.
Quy tắc bắt buộc:
1. CHỈ dùng thông tin trong phần NGỮ CẢNH. Tuyệt đối không dùng kiến thức riêng của bạn, KỂ CẢ KHI bạn biết câu trả lời.
2. Trước khi trả lời, kiểm tra NGỮ CẢNH có nói về ĐÚNG địa điểm, sự kiện hoặc chủ đề mà câu hỏi hỏi không.
   Nếu không có, hoặc câu hỏi không liên quan đến du lịch, văn hóa Việt Nam, hãy trả lời đúng một câu: "{REFUSAL}"
3. Nếu NGỮ CẢNH chỉ có một phần thông tin, chỉ trả lời phần đó và nói rõ phần nào chưa có thông tin.
4. Trả lời bằng tiếng Việt, ngắn gọn (tối đa khoảng 150 từ), đi thẳng vào câu hỏi.
5. Sau mỗi ý lấy từ ngữ cảnh, ghi số nguồn trong ngoặc vuông, ví dụ [1] hoặc [2]. Không ghi số nguồn cho thông tin không có trong đoạn đó.
6. Giữ nguyên tên riêng, con số, giờ giấc, giá vé như trong ngữ cảnh."""

REMINDER_V2 = (f'Nhắc lại: chỉ dùng NGỮ CẢNH ở trên. Nếu NGỮ CẢNH không nói về đúng đối tượng trong câu hỏi, '
               f'trả lời: "{REFUSAL}"')
PROMPTS = {"v1": (SYSTEM_RAG_V1, ""), "v2": (SYSTEM_RAG_V2, REMINDER_V2)}
# Đánh giá rút gọn 18/11/2026: v2 làm mô hình 3B từ chối nhầm nhiều (llama 63%, qwen 31%) trong khi câu ngoài
# phạm vi đã được ngưỡng liên quan của re-ranker xử lý -> dùng lại v1 làm mặc định.
DEFAULT_PROMPT = "v1"
SYSTEM_RAG = SYSTEM_RAG_V1

SYSTEM_NO_RAG = """Bạn là trợ lý hỏi đáp về du lịch và văn hóa Việt Nam.
Trả lời bằng tiếng Việt, ngắn gọn (tối đa khoảng 150 từ). Nếu không biết chắc, hãy nói là không biết."""


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, used = [], 0
    for i, c in enumerate(chunks, 1):
        txt = c["text"]
        if used + len(txt) > max_chars:
            txt = txt[: max(0, max_chars - used)]
        if not txt:
            break
        parts.append(f"[{i}] {txt}")
        used += len(txt)
    return "\n\n".join(parts)


class Generator:
    def __init__(self, model: str = DEFAULT_LLM, temperature: float = 0.1, num_ctx: int = 4096,
                 prompt: str = DEFAULT_PROMPT):
        self.model = model
        self.prompt = prompt
        self.system, self.reminder = PROMPTS[prompt]
        self.client = ollama.Client(host=OLLAMA_HOST)
        self.options = {"temperature": temperature, "num_ctx": num_ctx}

    def _messages(self, question: str, chunks: list[dict] | None):
        if chunks is None:
            return [{"role": "system", "content": SYSTEM_NO_RAG}, {"role": "user", "content": question}]
        ctx = build_context(chunks)
        tail = f"\n\n{self.reminder}" if self.reminder else ""
        user = f"NGỮ CẢNH:\n{ctx}\n\nCÂU HỎI: {question}{tail}\n\nTRẢ LỜI:"
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def generate(self, question: str, chunks: list[dict] | None) -> str:
        r = self.client.chat(model=self.model, messages=self._messages(question, chunks), options=self.options)
        return r["message"]["content"].strip()

    def stream(self, question: str, chunks: list[dict] | None):
        for part in self.client.chat(model=self.model, messages=self._messages(question, chunks),
                                     options=self.options, stream=True):
            yield part["message"]["content"]
