import sqlite3
import os
import json
from config_loader import load_config

# Load config
config = load_config()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lấy đường dẫn DB từ config hoặc mặc định
DB_PATH = config.get("vector_db", {}).get("metadata_path", "docs.db")
if not os.path.isabs(DB_PATH):
    DB_PATH = os.path.join(BASE_DIR, "..", DB_PATH)

def init_db():
    """Khởi tạo database và bảng documents nếu chưa có"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            doc_uuid TEXT,
            text TEXT,
            source TEXT,
            rep_type TEXT,
            full_path TEXT
        )
    ''')
    # Index để tìm kiếm nhanh theo full_path (tránh trùng lặp)
    c.execute('CREATE INDEX IF NOT EXISTS idx_full_path ON documents(full_path)')
    conn.commit()
    conn.close()

def get_doc_count():
    """Lấy tổng số documents hiện có (dùng để tính ID tiếp theo cho FAISS)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT MAX(id) FROM documents")
    result = c.fetchone()[0]
    conn.close()
    return (result + 1) if result is not None else 0

def get_all_full_paths():
    """Lấy danh sách tất cả các full_path đã xử lý"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT full_path FROM documents")
    rows = c.fetchall()
    conn.close()
    return {r[0] for r in rows}

def add_documents_batch(documents):
    """
    Thêm nhiều documents vào DB.
    documents: list of dict {'id': int, 'doc_uuid': str, 'text': str, ...}
    """
    if not documents:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    data = []
    for doc in documents:
        # doc['id'] ở đây là FAISS ID (integer 0, 1, 2...)
        data.append((
            doc['id'],
            doc['doc_uuid'],
            doc['text'],
            doc['source'],
            doc['rep_type'],
            doc['full_path']
        ))

    c.executemany('''
        INSERT INTO documents (id, doc_uuid, text, source, rep_type, full_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data)
    
    conn.commit()
    conn.close()

def get_documents_by_ids(ids):
    """
    Lấy thông tin documents theo danh sách FAISS IDs.
    Trả về list dict, giữ đúng thứ tự (nếu cần thiết có thể sort lại ở client)
    """
    if not ids:
        return []

    # SQLite không đảm bảo thứ tự trả về theo IN (...), nên ta lấy về rồi map lại
    placeholders = ",".join("?" * len(ids))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Để truy cập theo tên cột
    c = conn.cursor()
    
    query = f"SELECT * FROM documents WHERE id IN ({placeholders})"
    c.execute(query, ids)
    rows = c.fetchall()
    conn.close()

    # Convert to dict map
    doc_map = {}
    for r in rows:
        doc_map[r['id']] = {
            "id": r['id'],
            "doc_uuid": r['doc_uuid'],
            "text": r['text'],
            "source": r['source'],
            "rep_type": r['rep_type'],
            "full_path": r['full_path']
        }

    # Trả về list theo đúng thứ tự requested ids (filter out None)
    results = []
    for i in ids:
        if i in doc_map:
            results.append(doc_map[i])
            
    return results

def delete_documents_by_path(full_path):
    """
    Xóa tất cả documents có full_path khớp (dùng khi cập nhật bài viết).
    Trả về danh sách ID đã xóa để đồng bộ xóa bên FAISS.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Lấy danh sách ID cần xóa
    c.execute("SELECT id FROM documents WHERE full_path = ?", (full_path,))
    rows = c.fetchall()
    deleted_ids = [r[0] for r in rows]
    
    if deleted_ids:
        # 2. Xóa dữ liệu
        c.execute("DELETE FROM documents WHERE full_path = ?", (full_path,))
        conn.commit()
    
    conn.close()
    return deleted_ids

# Initialize on import (optional, but good for safety)
init_db()
