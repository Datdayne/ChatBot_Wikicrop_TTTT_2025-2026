import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from config_loader import load_config
import db  # Import module database mới

# --- CẤU HÌNH ---
config = load_config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "..", "faiss.index")

# Config path to DB is handled inside db.py via config loader, so we just use db module.

RERANK_MODEL = config["model"]["RERANK_MODEL"]
MODEL_NAME = config["model"]["embedding_model"]

# Config Performance
USE_RERANKER = config["vector_db"].get("use_reranker", True)
RETRIEVAL_TOP_K = config["vector_db"].get("retrieval_top_k", 30)
RERANK_TOP_N = config["vector_db"].get("rerank_top_n", 5)

# --- LOAD MODEL & DATA ---
print(f"⏳ Đang tải models...\n   - Embedding: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME)

if USE_RERANKER:
    print(f"   - Reranker: {RERANK_MODEL}")
    reranker = CrossEncoder(RERANK_MODEL)
else:
    print("   - Reranker: OFF (Chế độ Fast Mode)")
    reranker = None

# Load FAISS
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    raise FileNotFoundError("❌ Không tìm thấy file faiss.index! Hãy chạy ingest.py trước.")

# Không load docs.json nữa vì đã chuyển sang SQLite (lazy load)

def reload_index():
    """Reload FAISS index from disk (dùng sau khi Ingest)"""
    global index
    if os.path.exists(INDEX_FILE):
        print("🔄 Reloading FAISS index...")
        index = faiss.read_index(INDEX_FILE)
    else:
        print("⚠️ Không tìm thấy index để reload.")

# ==============================================================================
# 1. RETRIEVE & RERANK (CÓ LỌC NGƯỠNG ĐIỂM)
# ==============================================================================
def retrieve(query, top_k=RETRIEVAL_TOP_K, rerank_top_n=RERANK_TOP_N, score_threshold=0.0):
    """
    Tìm kiếm và lọc kết quả.
    - score_threshold: Ngưỡng điểm tối thiểu. Nếu điểm < 0 (hoặc thấp hơn), bỏ qua.
    """
    # 1. Embedding Query (Thêm prefix query: cho E5)
    qv = embedder.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")

    # 2. Tìm kiếm thô bằng FAISS
    D, I = index.search(qv, top_k)
    
    # Lấy ra danh sách ID hợp lệ, bỏ qua -1
    valid_ids = [int(idx) for idx in I[0] if idx != -1]
    
    # Truy vấn nội dung từ SQLite theo ID
    candidates = db.get_documents_by_ids(valid_ids)

    if not candidates:
        return []

    # 3. Rerank (Nếu bật)
    if USE_RERANKER:
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        
        # Ghép (candidate, score) lại và sort giảm dần
        ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    else:
        # Nếu tắt reranker, giữ nguyên thứ tự FAISS (Khoảng cách Euclid: càng nhỏ càng tốt, nhưng FAISS inner product: càng lớn càng tốt)
        # Tuy nhiên index đang là inner product hay L2? Thường mặc định là L2 nếu không nói gì. 
        # Nhưng ở đây ta cứ giả sử FAISS trả về theo thứ tự tốt nhất rồi.
        # Gán score giả định giảm dần để logic bên dưới hoạt động
        ranked_candidates = [(c, 1.0 - (i*0.01)) for i, c in enumerate(candidates)]

    # 4. Sắp xếp và LỌC (Filtering)
    results = []
    
    for i, (doc, score) in enumerate(ranked_candidates):
        # Nếu dùng reranker thì mới care threshold chặt chẽ, 
        # còn không dùng reranker thì score là giả định, nên bỏ qua check threshold âm
        if USE_RERANKER and score < score_threshold:
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
        retrieved = retrieve(query)

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