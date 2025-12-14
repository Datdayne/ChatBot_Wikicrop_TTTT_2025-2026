import os
import re
from docx import Document
import split_heading_data as shd

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def process_all_docx_in_tree(root_dir):
    """Duyệt toàn bộ cây thư mục và xử lý tất cả file .docx"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".docx") and not filename.startswith("~$"):
                input_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(dirpath, root_dir)
                output_dir = os.path.join(root_dir, "data_output", rel_path, filename[:-5])
                print(f"📄 Đang xử lý: {input_path}")
                shd.split_docx_by_content(input_path)
    print(f"📄 đã xử lý xong")


# --- Chạy thử ---

root_folder = "data"   # 👉 Thay bằng thư mục gốc chứa các file .docx
process_all_docx_in_tree(root_folder)