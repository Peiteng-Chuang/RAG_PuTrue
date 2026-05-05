from unstructured.partition.xlsx import partition_xlsx

def extract_excel_rag(file_path):
    # partition_xlsx 會自動解析所有分頁
    elements = partition_xlsx(filename=file_path)
    
    # 這裡會得到一系列的 Element 物件，包含表格內容
    for element in elements:
        if "Table" in str(type(element)):
            # 獲取該表格的 HTML 或 Text 表現形式
            print(element.metadata.to_dict()) # 這裡包含分頁名稱等資訊
            print(element.text)

# 範例使用
if __name__ == "__main__": 
    file_path = "C:/Users/Peiteng.Chuang/Desktop/璞真RAG/01完工結案報告(建築)本因坊/本因坊結案報告(提審)95.08.01.xls"
    extract_excel_rag(file_path)