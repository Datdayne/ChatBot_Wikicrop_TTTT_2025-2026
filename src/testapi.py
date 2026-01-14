import requests
import json

# Đổi link nếu cổng của bạn khác
API_URL = "http://localhost/wikicrop/api.php"

def test_wiki():
    print(f"ang kết nối tới: {API_URL}...")
    
    params = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": "10",      # Test thử 10 bài
        "prop": "extracts",    # Lấy nội dung
        "explaintext": 1,      # Chuyển về text thuần
        "exsectionformat": "plain",
        "format": "json"
    }

    try:
        resp = requests.get(API_URL, params=params)
        data = resp.json()
        
        # In ra cấu trúc JSON gốc để soi lỗi
        # print("Dữ liệu thô:", json.dumps(data, indent=2)) 

        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            print("❌ KHÔNG TÌM THẤY BÀI VIẾT NÀO! (Query trả về rỗng)")
            return

        print(f"✅ Tìm thấy {len(pages)} trang trong API:")
        print("-" * 40)
        
        for pid, info in pages.items():
            title = info.get("title", "Không tiêu đề")
            content = info.get("extract", "")
            ns = info.get("ns", -99)
            
            print(f"📄 ID: {pid} | Title: {title} | Namespace: {ns}")
            if content:
                print(f"   📝 Nội dung ({len(content)} ký tự): {content[:100]}...")
            else:
                print("   ⚠️  CÓ TIÊU ĐỀ NHƯNG KHÔNG CÓ NỘI DUNG (Rỗng)")
                
        print("-" * 40)

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    test_wiki()