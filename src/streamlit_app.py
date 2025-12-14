import streamlit as st
import requests

st.title("Chatbot Trồng Cây")
q = st.text_input("Câu hỏi")
if st.button("Hỏi"):
    resp = requests.post("http://localhost:8000/ask", json={"query": q}).json()
    st.write(resp["answer"])

upload = st.file_uploader("Upload tài liệu", type=["pdf","txt","docx"])
if upload:
    files = {"file": (upload.name, upload.getvalue())}
    r = requests.post("http://localhost:8000/upload", files=files)
    st.write(r.json())