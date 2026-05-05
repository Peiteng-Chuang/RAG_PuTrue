import fitz  # PyMuPDF
### 測試抓取PDF的CAD圖所在分頁

def get_cad_pages(pdf_path, threshold=200):
    """
    掃描 PDF 並回傳含有大量向量路徑（疑似 CAD 圖）的頁碼列表。
    
    Args:
        pdf_path (str): PDF 檔案路徑。
        threshold (int): 向量路徑數量的閾值。建築圖紙通常含有數千條路徑。
        
    Returns:
        list: 含有圖資的頁碼列表（從 0 開始索引）。
    """
    cad_page_indices = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            
            # 1. 檢查是否有點陣圖片 (Raster images)
            # has_images = len(page.get_images()) > 0
            
            # 2. 檢查向量路徑數量 (Vector paths)
            # get_drawings() 抓取所有線條、曲線等 CAD 特徵
            drawings = page.get_drawings()
            path_count = len(drawings)
            
            if path_count > threshold:
                cad_page_indices.append(page_index)
                print(f"[偵測成功] page [{page_index} ]: vector_path_count {path_count}")
            else:
                # 可選：這裡可以輸出輕量頁面的資訊以便除錯
                pass
                
        doc.close()
        return cad_page_indices

    except Exception as e:
        print(f"處理檔案時發生錯誤: {e}")
        return []

# --- 執行部分 ---
file_path = "C:/Users/Peiteng.Chuang/Desktop/璞真RAG/本因坊結案報告(提審)95.08.01.pdf"
cad_pages = get_cad_pages(file_path, threshold=300) # 建議建築圖紙設 300-500

print("-" * 30)
print(f"總共偵測到 {len(cad_pages)} 頁含有 CAD 或大量圖資。")
print(f"index 列表: {cad_pages}")