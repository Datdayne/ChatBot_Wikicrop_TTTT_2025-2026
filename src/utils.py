import requests
import json
import torch
from keybert import KeyBERT
from config_loader import load_config

# --- LOAD CONFIG ---
config = load_config()

# Lấy tên model từ config
SUMMARY_MODEL_NAME = config["model"].get("summary_model") # Có thể là None
KEYWORDS_MODEL_NAME = config["model"].get("keywords_model", "intfloat/multilingual-e5-small")
LLM_MODEL_NAME = config["model"].get("llm_model", "qwen2.5") # Dùng cho fallback

# ==============================================================================
# 1. KHỞI TẠO KEYBERT (Trích xuất từ khóa)
# ==============================================================================
print(f"Loading Keyword Model: {KEYWORDS_MODEL_NAME}...")
try:
    kw_model = KeyBERT(model=KEYWORDS_MODEL_NAME)
except Exception as e:
    print(f"⚠️ Lỗi tải KeyBERT: {e}. Sẽ bỏ qua bước trích xuất từ khóa.")
    kw_model = None

# ==============================================================================
# 2. KHỞI TẠO MODEL TÓM TẮT (Xử lý thông minh)
# ==============================================================================
hf_summary_model = None
hf_tokenizer = None

if SUMMARY_MODEL_NAME:
    # TRƯỜNG HỢP 1: Có cấu hình model riêng (Tốn RAM)
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print(f"Loading Summary Model: {SUMMARY_MODEL_NAME}...")
        
        # Tự động chọn thiết bị (GPU nếu có, không thì CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load đúng tokenizer theo model (không hardcode VietAI nữa)
        hf_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
        hf_summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME).to(device)
        
    except Exception as e:
        print(f"⚠️ Không tải được model tóm tắt HF: {e}. Sẽ chuyển sang dùng Ollama.")
        SUMMARY_MODEL_NAME = None 
else:
    # TRƯỜNG HỢP 2: Config để null -> Dùng Ollama (Tiết kiệm RAM)
    print(f"ℹ️ Chế độ tiết kiệm RAM: Sử dụng {LLM_MODEL_NAME} (Ollama) để tóm tắt.")

# ==============================================================================
# HÀM TÓM TẮT (Hybrid: HuggingFace hoặc Ollama)
# ==============================================================================
def extract_summary(text, max_len=150):
    """
    Tự động chọn cách tóm tắt dựa trên config.
    """
    if not text or not text.strip():
        return ""

    # CÁCH 1: Dùng Model chuyên dụng (Nếu đã load thành công)
    if hf_summary_model and hf_tokenizer:
        try:
            input_text = "Tóm tắt: " + text
            device = hf_summary_model.device
            
            input_ids = hf_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(device)
            
            output = hf_summary_model.generate(
                input_ids, 
                max_length=max_len, 
                min_length=20,
                num_beams=4, 
                early_stopping=True
            )
            return hf_tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Lỗi HF Summary: {e}")
            return text[:500] # Fallback cắt ngắn

    # CÁCH 2: Dùng Ollama (Nếu không có model riêng)
    else:
        return _summarize_with_ollama(text)

def _summarize_with_ollama(text):
    """Gọi API Ollama để tóm tắt"""
    url = "http://localhost:11434/api/generate"
    
    # Prompt ép Qwen tóm tắt ngắn gọn
    prompt = (
        f"Bạn là trợ lý tóm tắt văn bản. Hãy viết một đoạn tóm tắt ngắn gọn (khoảng 2-3 câu) "
        f"bằng tiếng Việt cho nội dung sau đây:\n\n{text[:2000]}\n\n" # Giới hạn input
        f"Tóm tắt:"
    )

    payload = {
        "model": LLM_MODEL_NAME, # qwen2.5
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3, # Thấp để tập trung
            "num_predict": 150  # Giới hạn độ dài output
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # Nếu model trả về rỗng hoặc lỗi, lấy text gốc cắt ngắn
            return result if result else text[:300] + "..."
        else:
            return text[:300] + "..."
    except:
        return text[:300] + "..."

# ==============================================================================
# HÀM TRÍCH XUẤT TỪ KHÓA
# ==============================================================================
def extract_keywords(text, top_k=10):
    if not kw_model:
        return ""
        
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),    # Cụm 1-2 từ
            stop_words=None,                 
            top_n=top_k
        )
        # keywords là list [(từ, điểm)]
        return ", ".join([kw for kw, _ in keywords])
    except Exception as e:
        print(f"Lỗi KeyBERT: {e}")
        return ""