import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn
import faiss
import json
import numpy as np

import qa      
import ingest  
import db # Import module DB
from config_loader import load_config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model dữ liệu (Dùng để validate thủ công)
class IngestRequest(BaseModel):
    title: str
    content: str
    url: str

class QuestionRequest(BaseModel):
    query: str

# --- API HỎI ĐÁP ---
@app.post("/ask")
async def ask_endpoint(request: QuestionRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Câu hỏi rỗng")
    try:
        bot_response = qa.answer(request.query)
        return {"answer": bot_response}
    except Exception as e:
        print(f"❌ Lỗi server: {e}")
        return {"answer": "Xin lỗi, hệ thống đang gặp sự cố."}

@app.post("/ingest")
async def ingest_endpoint(raw_request: Request):
    try:
        # 1. Đọc dữ liệu thô (Bytes)
        body_bytes = await raw_request.body()
        
        # 2. Convert Bytes sang JSON (Dictionary)
        data = json.loads(body_bytes)
        
        # 3. Validate dữ liệu khớp với mẫu IngestRequest
        request_data = IngestRequest(**data)
        
    except Exception as e:
        print(f"\n❌ Lỗi đọc dữ liệu JSON: {e}")
        try:
            print(f"📦 Body gốc: {body_bytes.decode('utf-8')}")
        except: pass
        raise HTTPException(status_code=422, detail="Dữ liệu gửi lên không phải JSON hợp lệ")

    # --- BẮT ĐẦU XỬ LÝ (Dùng biến request_data) ---
    print(f"📥 Đang xử lý bài viết: {request_data.title}")
    
    try:
        # 1. XÓA DỮ LIỆU CŨ TỪ SQLite & Update Index
        # Xóa khỏi DB và lấy IDs đã xóa
        deleted_ids = db.delete_documents_by_path(request_data.url)
        
        # Cập nhật set tracking của ingest
        ingest.remove_from_processed(request_data.url)

        if deleted_ids:
            print(f"   ♻️ Đã xóa {len(deleted_ids)} chunk cũ từ DB.")
            # Xóa khỏi memory index của ingest
            ingest.index.remove_ids(np.array(deleted_ids, dtype=np.int64))

        # 2. TẠO DỮ LIỆU MỚI
        vecs, db_entries = ingest.process_content(
            request_data.content, 
            f"Wiki: {request_data.title}", 
            request_data.url, 
            source_type="wiki",
            force_update=True 
        )

        if not vecs:
            print("⚠️ Nội dung rỗng sau khi xử lý.")
            # Nếu có xóa mà không có mới -> Save index hiện tại (đã remove) xuống đĩa
            if deleted_ids:
                 faiss.write_index(ingest.index, ingest.INDEX_FILE)
                 qa.reload_index()
            return {"status": "warning", "message": "Nội dung rỗng."}

        # 3. LƯU VÀO DB & DISK 
        # ingest.save_batch tự động add vào ingest.index, save disk và insert DB
        ingest.save_batch(vecs, db_entries)
        
        # 4. RELOAD QA INDEX (Để chatbot tìm thấy ngay)
        qa.reload_index()

        print(f"✅ Đã học xong: {request_data.title}")
        return {"status": "success", "chunks": len(vecs)}

    except Exception as e:
        print(f"❌ Lỗi Ingest Logic: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🚀 Server Chatbot RAG (Robust Mode) đang chạy tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)