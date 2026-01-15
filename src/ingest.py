import os
import uuid
import numpy as np
import faiss
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from extractors import auto_extract
from config_loader import load_config
import db  # Import module database mới

# --- CONFIG ---
config = load_config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "..", "faiss.index")
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
if os.path.exists(INDEX_FILE):
    print("📂 Tải index cũ...")
    index = faiss.read_index(INDEX_FILE)
else:
    print("✨ Tạo index mới...")
    # Dùng IndexIDMap để quản lý ID thủ công (khớp vói DB)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

# Lấy danh sách nguồn đã xử lý từ DB
processed_sources = db.get_all_full_paths()

def remove_from_processed(full_path):
    if full_path in processed_sources:
        processed_sources.remove(full_path)

# ==============================================================================
# PHẦN 1: XỬ LÝ NỘI DUNG (Chunk -> Embed)
# ==============================================================================
def process_content(text, source_name, full_identifier, source_type="file", force_update=False):
    """Hàm chung để xử lý văn bản -> Chunk -> Embed"""
    
    if not force_update:
        if full_identifier in processed_sources:
            return [], []

    if not text or not text.strip():
        return [], []

    # 1. Chunking
    if len(text) > 1200:
        chunks = text_splitter.split_text(text)
    else:
        chunks = [text]

    vecs = []
    # Thay vì lưu meta dict hoàn chỉnh, ta lưu dữ liệu raw để insert DB
    db_entries = []

    # 2. Embedding từng chunk
    for i, chunk_text in enumerate(chunks):
        doc_uuid = str(uuid.uuid4())
        
        display_source = source_name
        if len(chunks) > 1:
            display_source += f" (Đoạn {i+1})"

        embed_input = f"passage: {chunk_text}" 
        vec = embedder.encode(embed_input, normalize_embeddings=True)

        entry = {
            "doc_uuid": doc_uuid,
            "source": display_source,
            "rep_type": "wiki_content" if source_type == "wiki" else "file_content",
            "text": chunk_text,
            "full_path": full_identifier
        }
        
        vecs.append(vec)
        db_entries.append(entry)

    return vecs, db_entries

# ==============================================================================
# PHẦN 2: HÚT DỮ LIỆU TỪ MEDIAWIKI API
# ==============================================================================
def fetch_all_wiki_pages():
    """Lấy TOÀN BỘ bài viết từ Wiki"""
    print(f"🌐 Đang kết nối tới Wiki: {WIKI_API_URL}")
    
    session = requests.Session()
    base_params = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": "max",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json"
    }

    results = []
    last_continue = {}
    page_count = 0

    while True:
        params = {**base_params, **last_continue}
        try:
            resp = session.get(WIKI_API_URL, params=params)
            data = resp.json()
            
            if "error" in data:
                print(f"❌ API Error: {data['error']}")
                break

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                title = page_data.get("title", "")
                ns = page_data.get("ns", 0)
                if ns != 0: continue

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

            if 'continue' in data:
                last_continue = data['continue']
            else:
                break

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
    new_db_entries = []

    for title, content, url in tqdm(pages, desc="Processing Wiki", unit="page"):
        v, m = process_content(content, f"Wiki: {title}", url, source_type="wiki")
        if v:
            new_vectors.extend(v)
            new_db_entries.extend(m)

    if new_vectors:
        save_batch(new_vectors, new_db_entries)
        print(f"🎉 Đã thêm {len(new_db_entries)} đoạn văn từ Wiki vào bộ nhớ.")
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
    new_db_entries = []
    
    for path in tqdm(docx_files, desc="Processing Files", unit="file"):
        try:
            raw_text = auto_extract(path)
            v, m = process_content(raw_text, os.path.basename(path), path, source_type="file")
            if v:
                new_vectors.extend(v)
                new_db_entries.extend(m)
        except Exception as e:
            print(f"Lỗi file {path}: {e}")

    if new_vectors:
        save_batch(new_vectors, new_db_entries)
        print(f"🎉 Đã thêm {len(new_db_entries)} đoạn văn từ File vào bộ nhớ.")

# --- Helper lưu đĩa ---
def save_batch(vectors, db_entries):
    """
    vectors: list of numpy arrays
    db_entries: list of dicts (chưa có ID)
    """
    if not vectors: return
    
    # Lấy ID bắt đầu hiện tại từ DB (để khớp với FAISS index)
    start_id = db.get_doc_count()
    
    # Gán ID cho các entry mới
    final_db_entries = []
    for i, entry in enumerate(db_entries):
        entry['id'] = start_id + i
        final_db_entries.append(entry)
        # update tracking set
        processed_sources.add(entry['full_path'])

    # 1. Thêm vào FAISS (với ID cụ thể)
    vecs_np = np.vstack(vectors).astype("float32")
    ids_np = np.array([e['id'] for e in final_db_entries], dtype=np.int64)
    index.add_with_ids(vecs_np, ids_np)
    
    faiss.write_index(index, INDEX_FILE)

    # 2. Thêm vào SQLite
    db.add_documents_batch(final_db_entries)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Đảm bảo DB được khởi tạo
    db.init_db()
    
    # 1. Quét file docx
    ingest_local_files()
    
    # 2. Quét bài viết trên Wiki
    ingest_wiki()
    
    print("\n✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HỌC DỮ LIỆU!")