import sys
import os
from pathlib import Path
from pypdf import PdfReader
import fitz  # PyMuPDF

def run_pypdf_extract(pdf_path, base_output):
    output_dir = Path(base_output) / "pypdf_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reader = PdfReader(pdf_path)
    total = 0
    for page_num, page in enumerate(reader.pages, start=1):
        for img_idx, img_obj in enumerate(page.images, start=1):
            filename = f"p{page_num}_{img_idx}_{img_obj.name}"
            # 有些物件名可能包含路徑字元，清理一下
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                filename += ".png"
            
            with open(output_dir / filename, "wb") as f:
                f.write(img_obj.data)
            total += 1
    return total

def run_fitz_extract(pdf_path, base_output):
    output_dir = Path(base_output) / "fitz_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 模式 1: 直接提取原始圖片物件 (高解析度原始檔)
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"] # 自動偵測副檔名
            
            filename = f"p{page_num+1}_{img_idx}_raw.{ext}"
            with open(output_dir / filename, "wb") as f:
                f.write(image_bytes)
            total += 1
            
        # 模式 2: 如果該頁是掃描件(沒圖片物件)，則渲染整頁為高解析度圖片 (300 DPI)
        if not image_list:
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72)) # 放大 4.16 倍達 300 DPI
            filename = f"p{page_num+1}_page_render.png"
            pix.save(output_dir / filename)
            total += 1
    return total

if __name__ == "__main__":
    target_pdf = "./test2.pdf" # Docker 內的路徑
    compare_dir = "./fitz_pypdf_compare"
    
    print(f"🧐 開始比對測試 PDF: {target_pdf}")
    
    # 執行 pypdf
    c1 = run_pypdf_extract(target_pdf, compare_dir)
    print(f"✅ pypdf 完成: 提取 {c1} 張圖片")
    
    # 執行 fitz
    c2 = run_fitz_extract(target_pdf, compare_dir)
    print(f"✅ fitz 完成: 提取/渲染 {c2} 張圖片")
    
    print(f"\n📁 比對結果已儲存至: {compare_dir}")
    print("💡 提示：請觀察 fitz_output 裡的圖片，解析度與色彩穩定度通常會優於 pypdf。")