import os
import json
import uuid
import numpy as np
import faiss
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from extractors import auto_extract
from utils import extract_summary, extract_keywords
from config_loader import load_config

# --- CONFIG ---
config = load_config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "..", "faiss.index")
META_FILE = os.path.join(BASE_DIR, "..", "docs.json")
DATA_DIR = os.path.join(BASE_DIR, "..", "data_output")

# Cấu hình API Wiki
WIKI_API_URL = "http://localhost/wikicrop/api.php"

MODEL_NAME = config["model"]["embedding_model"]
print(f"🔄 Đang tải model: {MODEL_NAME}...")
embedder = SentenceTransformer(MODEL_NAME)
dimension = embedder.get_sentence_embedding_dimension()

# --- CHUNKING ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# --- LOAD FAISS ---
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    print("📂 Tải index cũ...")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
else:
    print("✨ Tạo index mới...")
    index = faiss.IndexFlatIP(dimension)
    docs = []

# Tập hợp các nguồn đã xử lý để tránh trùng
processed_sources = set(d['full_path'] for d in docs)

# ==============================================================================
# PHẦN 1: XỬ LÝ NỘI DUNG (Chunk -> Embed)
# ==============================================================================
# Sửa dòng định nghĩa hàm: thêm force_update=False
def process_content(text, source_name, full_identifier, source_type="file", force_update=False):
    """Hàm chung để xử lý văn bản -> Chunk -> Embed"""
    
    # Logic kiểm tra trùng lặp:
    # Nếu KHÔNG PHẢI là ép buộc (force=False) VÀ đã tồn tại -> Thì mới bỏ qua
    if not force_update:
        if source_type == "file" and full_identifier in processed_sources:
            return [], []
        # Với Wiki, nếu không force thì cũng bỏ qua nếu đã có
        if source_type == "wiki" and full_identifier in processed_sources:
             return [], []

    if not text or not text.strip():
        return [], []

    # ... (Phần chunking và embedding bên dưới giữ nguyên) ...

    # 1. Chunking
    if len(text) > 1200:
        chunks = text_splitter.split_text(text)
    else:
        chunks = [text]

    vecs = []
    metas = []

    # 2. Embedding từng chunk
    for i, chunk_text in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        
        display_source = source_name
        if len(chunks) > 1:
            display_source += f" (Đoạn {i+1})"

        # Embed nội dung chunk (Dùng chính chunk_text làm input cho chính xác)
        # Nếu muốn dùng summary, có thể bỏ comment các dòng dưới
        # summary_text = extract_summary(chunk_text)
        embed_input = chunk_text 

        vec = embedder.encode(embed_input, normalize_embeddings=True)

        meta = {
            "id": doc_id,
            "source": display_source,
            "rep_type": "wiki_content" if source_type == "wiki" else "file_content",
            "text": chunk_text,
            "full_path": full_identifier
        }
        
        vecs.append(vec)
        metas.append(meta)

    return vecs, metas

# ==============================================================================
# PHẦN 2: HÚT DỮ LIỆU TỪ MEDIAWIKI API (Đã nâng cấp Pagination)
# ==============================================================================
def fetch_all_wiki_pages():
    """Lấy TOÀN BỘ bài viết từ Wiki (Xử lý phân trang chuẩn + Lấy raw wikitext)"""
    print(f"🌐 Đang kết nối tới Wiki: {WIKI_API_URL}")
    
    session = requests.Session()
    
    # Tham số cơ bản (Chưa có token phân trang)
    base_params = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": "max",     # Lấy tối đa số lượng mỗi lần gọi
        "prop": "revisions",   # Lấy phiên bản sửa đổi (raw content)
        "rvprop": "content",   # Nội dung
        "rvslots": "main",     # Slot chính
        "format": "json"
    }

    results = []
    last_continue = {} # Biến lưu dấu vết để lật trang
    page_count = 0

    # --- VÒNG LẶP VÉT CẠN (Pagination Loop) ---
    while True:
        # Trộn tham số cơ bản với token tiếp theo (nếu có)
        params = {**base_params, **last_continue}
        
        try:
            resp = session.get(WIKI_API_URL, params=params)
            data = resp.json()
            
            # Xử lý lỗi API nếu có
            if "error" in data:
                print(f"❌ API Error: {data['error']}")
                break

            # 1. Xử lý dữ liệu đợt này
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page_data in pages.items():
                title = page_data.get("title", "")
                
                # Bỏ qua các trang hệ thống (Namespace != 0)
                ns = page_data.get("ns", 0)
                if ns != 0: continue

                # Lấy nội dung thô (Wikitext) từ cấu trúc JSON
                content = ""
                try:
                    revisions = page_data.get("revisions", [])
                    if revisions:
                        content = revisions[0]["slots"]["main"]["*"]
                except KeyError:
                    pass

                if content:
                    fake_url = f"wiki://{title}"
                    results.append((title, content, fake_url))
                    page_count += 1
            
            print(f"   ... Đã quét được {page_count} bài viết...")

            # 2. Kiểm tra xem còn trang sau không? (Quan trọng)
            if 'continue' in data:
                last_continue = data['continue'] # Lấy token để đi tiếp vòng sau
            else:
                break # Hết dữ liệu rồi, thoát vòng lặp

        except Exception as e:
            print(f"❌ Lỗi khi quét Wiki: {e}")
            break
            
    print(f"✅ Đã tải xong TẤT CẢ {len(results)} bài viết từ Wiki.")
    return results

def ingest_wiki():
    print("\n--- 🌍 BẮT ĐẦU QUÉT WIKI ONLINE ---")
    pages = fetch_all_wiki_pages()
    
    if not pages:
        print("⚠️ Không tìm thấy bài viết nào trên Wiki hoặc lỗi kết nối.")
        return
    
    new_vectors = []
    new_metas = []

    for title, content, url in tqdm(pages, desc="Processing Wiki", unit="page"):
        v, m = process_content(content, f"Wiki: {title}", url, source_type="wiki")
        if v:
            new_vectors.extend(v)
            new_metas.extend(m)

    if new_vectors:
        _save_batch(new_vectors, new_metas)
        print(f"🎉 Đã thêm {len(new_metas)} đoạn văn từ Wiki vào bộ nhớ.")
    else:
        print("⏩ Không có dữ liệu mới từ Wiki để cập nhật.")

# ==============================================================================
# PHẦN 3: HÚT DỮ LIỆU TỪ FILE LOCAL
# ==============================================================================
def ingest_local_files(root_folder=DATA_DIR):
    print(f"\n--- 📂 BẮT ĐẦU QUÉT FILE LOCAL ({root_folder}) ---")
    docx_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.lower().endswith(".docx") and not f.startswith("~$"):
                docx_files.append(os.path.join(dirpath, f))

    new_vectors = []
    new_metas = []
    
    for path in tqdm(docx_files, desc="Processing Files", unit="file"):
        try:
            raw_text = auto_extract(path)
            v, m = process_content(raw_text, os.path.basename(path), path, source_type="file")
            if v:
                new_vectors.extend(v)
                new_metas.extend(m)
        except Exception as e:
            print(f"Lỗi file {path}: {e}")

    if new_vectors:
        _save_batch(new_vectors, new_metas)
        print(f"🎉 Đã thêm {len(new_metas)} đoạn văn từ File vào bộ nhớ.")

# --- Helper lưu đĩa ---
def _save_batch(vectors, metas):
    if not vectors: return
    vecs_np = np.vstack(vectors).astype("float32")
    index.add(vecs_np)
    docs.extend(metas)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    
    for m in metas:
        processed_sources.add(m['full_path'])

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Lưu ý: Nên xóa file faiss.index và docs.json trước khi chạy nếu muốn làm mới hoàn toàn
    
    # 1. Quét file docx
    ingest_local_files()
    
    # 2. Quét bài viết trên Wiki
    ingest_wiki()
    
    print("\n✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HỌC DỮ LIỆU!")