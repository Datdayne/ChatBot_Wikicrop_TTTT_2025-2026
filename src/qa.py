import numpy as np
import faiss
import json
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from config_loader import load_config

# --- CẤU HÌNH ---
config = load_config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "..", "faiss.index")
META_FILE = os.path.join(BASE_DIR, "..", "docs.json")

RERANK_MODEL = config["model"]["RERANK_MODEL"]
MODEL_NAME = config["model"]["embedding_model"]

# --- LOAD MODEL & DATA ---
print(f"⏳ Đang tải models...\n   - Embedding: {MODEL_NAME}\n   - Reranker: {RERANK_MODEL}")
embedder = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL)

# Load FAISS
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    raise FileNotFoundError("❌ Không tìm thấy file faiss.index! Hãy chạy ingest.py trước.")

# Load Metadata
if os.path.exists(META_FILE):
    with open(META_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
else:
    docs = []

# ==============================================================================
# 1. RETRIEVE & RERANK (CÓ LỌC NGƯỠNG ĐIỂM)
# ==============================================================================
def retrieve(query, top_k=30, rerank_top_n=5, score_threshold=0.0):
    """
    Tìm kiếm và lọc kết quả.
    - score_threshold: Ngưỡng điểm tối thiểu. Nếu điểm < 0 (hoặc thấp hơn), bỏ qua.
    """
    # 1. Embedding Query
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")

    # 2. Tìm kiếm thô bằng FAISS
    D, I = index.search(qv, top_k)
    
    candidates = []
    # Lấy ra danh sách candidate, bỏ qua -1 (không tìm thấy)
    for idx in I[0]:
        if idx != -1 and idx < len(docs):
            candidates.append(docs[idx])

    if not candidates:
        return []

    # 3. Rerank bằng CrossEncoder (Chính xác hơn Cosine)
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # 4. Sắp xếp và LỌC (Filtering)
    results = []
    # Ghép (candidate, score) lại và sort giảm dần
    ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    for i, (doc, score) in enumerate(ranked_candidates):
        if score < score_threshold:
            continue  # Bỏ qua kết quả kém
            
        if len(results) >= rerank_top_n:
            break # Đã đủ số lượng cần lấy

        results.append({
            "rank": len(results) + 1,
            "source": doc["source"],
            "rep_type": doc["rep_type"],
            "score": float(score),
            "text": doc["text"]
        })

    return results

# ==============================================================================
# 2. BUILD PROMPT 
# ==============================================================================
def make_prompt(query: str, retrieved: list, role: str = "Chuyên gia nông nghiệp") -> str:
    if not retrieved:
        # Nếu không có tài liệu nào vượt qua ngưỡng điểm
        return None

    # Ghép ngữ cảnh
    parts = []
    for i, r in enumerate(retrieved, 1):
        # Làm sạch text một chút
        clean_text = r["text"].replace("\n", " ").strip()
        parts.append(f"Tài liệu [{i}] (Nguồn: {r['source']}):\n{clean_text}")

    context = "\n\n".join(parts)

    # Prompt Engineering: "Guardrails" (Hàng rào bảo vệ)
    prompt = (
        f"Bạn là {role}. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các tài liệu được cung cấp dưới đây.\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n"
        f"Câu hỏi: {query}\n\n"
        f"Yêu cầu:\n"
        f"1. CHỈ sử dụng thông tin trong các tài liệu trên để trả lời.\n"
        f"2. Nếu tài liệu không chứa câu trả lời, hãy nói: 'Xin lỗi, tôi không tìm thấy thông tin trong cơ sở dữ liệu'.\n"
        f"3. Không tự bịa đặt thông tin hoặc dùng kiến thức bên ngoài.\n"
        f"4. Trả lời ngắn gọn, súc tích và trích dẫn nguồn (Ví dụ: [Tài liệu 1]).\n"
        f"Câu trả lời:"
    )
    return prompt

# ==============================================================================
# 3. CALL OLLAMA 
# ==============================================================================
def call_ollama(prompt: str, model: str = "qwen2.5", temperature: float = 0.3) -> str:
    """
    Gọi Ollama. Mặc định dùng qwen2.5 (nếu máy yếu dùng qwen2.5:3b)
    Temperature thấp (0.3) để model bớt "sáng tạo" lung tung.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature, 
            "num_predict": 1024
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "Lỗi: Model không phản hồi.")
    except Exception as e:
        return f"Lỗi kết nối Ollama: {e}"

# ==============================================================================
# 4. MAIN FLOW
# ==============================================================================
def answer(query: str, model: str = "qwen2.5", debug: bool = True) -> str:
    try:
        retrieved = retrieve(query, top_k=30, rerank_top_n=5, score_threshold=0.0)

        if debug:
            print(f"\n=== 🔍 Debug: Tìm thấy {len(retrieved)} tài liệu phù hợp ===")
            for r in retrieved:
                print(f"[{r['score']:.4f}] {r['source']} ({r['rep_type']})")
            print("===================================================\n")

        # 2. Tạo Prompt
        prompt = make_prompt(query, retrieved)
        
        # Nếu không có tài liệu nào qua được vòng gửi xe
        if prompt is None:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu của bạn (Điểm tin cậy quá thấp)."

        # 3. Gọi LLM
        return call_ollama(prompt, model=model)

    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}"