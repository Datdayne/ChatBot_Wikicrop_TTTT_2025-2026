# 🚀 Hướng dẫn khởi chạy hệ thống RAG

# Dự án sử dụng Ollama với mô hình gwen 2.5

## Mô tả

Dự án này sử dụng mô hình **Gemma Qwen2.5** chạy qua **Ollama** để thực hiện các tác vụ AI cục bộ.

## Yêu cầu môi trường

- Python 3.12
- Ollama (đã cài sẵn)
- Mô hình: `gwen2.5'

## Cài đặt

### 1. Cài Ollama

Tải và cài từ: [https://ollama.ai](https://ollama.ai)

### 2. Tải mô hình Gwen 2.5

ollama pull qwen2.5

## 2️⃣ Cài đặt thư viện cần thiết

Chạy lệnh sau trong thư mục dự án:
pip install -r requirements.txt

3 Khởi chạy server API
Chạy:
python ./src/server.py
Mặc định server chạy tại:
👉 http://127.0.0.1:8000

Lưu Ý : chỉ chạy mỗi "python ./src/server.py"
