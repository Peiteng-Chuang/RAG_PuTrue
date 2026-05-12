import os
from opendataloader_pdf import convert

# 設定路徑
input_file = "/app/test2.pdf"
output_directory = "/app/rag_results"

# 確保輸出目錄存在
os.makedirs(output_directory, exist_ok=True)
os.makedirs(os.path.join(output_directory, "images"), exist_ok=True)

if os.path.exists(input_file):
    print(f"🚀 啟動 OpenDataLoader 2.2.1 引擎...")
    try:
        # 根據 help() 修正後的參數
        convert(
            input_path=input_file,
            output_dir=output_directory,
            format="markdown",             # 之前報錯就是因為寫成 output_format
            image_output="external",       # 圖片輸出為獨立檔案
            image_dir=os.path.join(output_directory, "images"),
            image_format="png"
        )
        
        print(f"✨ 轉換成功！")
        print(f"📁 請到 Windows 資料夾下的 rag_results 查看結果。")
        
    except Exception as e:
        print(f"❌ 執行失敗: {e}")
else:
    print(f"❌ 找不到檔案: {input_file}")