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
        # 1. XÓA DỮ LIỆU CŨ
        indices_to_remove = []
        for i, doc in enumerate(qa.docs):
            if doc.get('full_path') == request_data.url:
                indices_to_remove.append(i)
        
        if indices_to_remove:
            print(f"   ♻️ Xóa {len(indices_to_remove)} chunk cũ...")
            qa.index.remove_ids(np.array(indices_to_remove, dtype=np.int64))
            for i in sorted(indices_to_remove, reverse=True):
                del qa.docs[i]

        # 2. TẠO DỮ LIỆU MỚI
        vecs, metas = ingest.process_content(
            request_data.content, 
            f"Wiki: {request_data.title}", 
            request_data.url, 
            source_type="wiki",
            force_update=True 
        )

        if not vecs:
            print("⚠️ Nội dung rỗng sau khi xử lý.")
            return {"status": "warning", "message": "Nội dung rỗng."}

        # 3. NẠP VÀO RAM
        vecs_np = np.vstack(vecs).astype("float32")
        qa.index.add(vecs_np)
        qa.docs.extend(metas)

        # 4. LƯU XUỐNG ĐĨA
        faiss.write_index(qa.index, ingest.INDEX_FILE)
        with open(ingest.META_FILE, "w", encoding="utf-8") as f:
            json.dump(qa.docs, f, ensure_ascii=False, indent=2)
        
        for m in metas:
            ingest.processed_sources.add(m['full_path'])

        print(f"✅ Đã học xong: {request_data.title}")
        return {"status": "success", "chunks": len(vecs)}

    except Exception as e:
        print(f"❌ Lỗi Ingest Logic: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("🚀 Server Chatbot RAG (Robust Mode) đang chạy tại http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)