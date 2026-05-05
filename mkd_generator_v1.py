import fitz
import os
import shutil
import torch
import gc
import sys
from pathlib import Path

# 導入 Marker 核心組件
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def process_pdf_with_marker_api(pdf_path):
    pdf_path = os.path.abspath(pdf_path)
    pdf_name = Path(pdf_path).stem
    
    tmp_dir = Path("./tmp").absolute()
    output_base = Path("./mkdata").absolute()
    img_dir = output_base / f"{pdf_name}_image"
    md_file_path = output_base / f"{pdf_name}.md"

    for d in [tmp_dir, output_base, img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    vector_indices = []
    md_lines = []

    print(f"🚀 開始掃描文件: {pdf_name} (共 {len(doc)} 頁)")

    # --- 階段 1: Fitz 基礎提取 ---
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        md_lines.append(f"## 第[{page_idx}]頁\n\n")
        
        # 提取文字
        text = page.get_text().strip()
        md_lines.append(f"{text}\n\n")
        
        # 提取 Fitz 普通圖片
        img_list = page.get_images(full=True)
        for img_seq, img_info in enumerate(img_list):
            try:
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
                img_filename = f"{pdf_name}_{page_idx}-{img_seq}.png"
                img_save_path = img_dir / img_filename
                
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # 檢查檔案是否已存在，避免多進程衝突寫入
                if not img_save_path.exists():
                    pix.save(str(img_save_path))
                
                md_lines.append(f"![{img_filename}]({pdf_name}_image/{img_filename})\n")
                pix = None
            except Exception as e:
                print(f"圖片提取跳過: {e}")

        # 判定向量圖
        if len(page.get_drawings()) > 200:
            vector_indices.append(page_idx)
            md_lines.append(f"\n")
        
        md_lines.append("\n---\n")

    # --- 階段 2: 使用 Marker API 處理向量圖頁面 ---
    if vector_indices:
        print(f"📊 偵測到 {len(vector_indices)} 頁向量圖，準備啟動 Marker...")
        
        artifact_dict = None
        converter = None
        
        try:
            # 準備暫存 PDF
            tmp_pdf_path = tmp_dir / f"{pdf_name}_v_temp.pdf"
            v_doc = fitz.open()
            v_doc.insert_pdf(doc, from_page=0, to_page=len(doc)-1)
            v_doc.select(vector_indices)
            v_doc.save(str(tmp_pdf_path))
            v_doc.close()

            # 載入模型
            artifact_dict = create_model_dict()
            converter = PdfConverter(artifact_dict=artifact_dict)
            
            with torch.no_grad():
                rendered = converter(str(tmp_pdf_path))
                # 這裡只提取圖片，忽略 text
                _, _, marker_images = text_from_rendered(rendered)

            if marker_images:
                for img_name, img_obj in marker_images.items():
                    v_img_filename = f"{pdf_name}_vector_{img_name}"
                    v_img_path = img_dir / v_img_filename
                    
                    if isinstance(img_obj, bytes):
                        with open(v_img_path, "wb") as f: f.write(img_obj)
                    else:
                        img_obj.save(v_img_path)
            
            # 更新記號
            for v_idx in vector_indices:
                placeholder = f""
                for i, line in enumerate(md_lines):
                    if placeholder in line:
                        md_lines[i] = f"> [!IMPORTANT] 偵測到複雜向量圖表，已由 Marker 完成解析。\n"

        except Exception as e:
            print(f"❌ Marker API 階段報錯: {e}")
        finally:
            # 強制回收
            if converter: del converter
            if artifact_dict: del artifact_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("🧹 VRAM 已釋放")

    # --- 階段 3: 存檔 ---
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    
    doc.close()
    print(f"🎊 處理完成: {md_file_path}")

# ==========================================
# 🔥 關鍵修復：必須加上這個判斷，防止 Windows 多進程崩潰
# ==========================================
if __name__ == "__main__":
    target_file = "C:/Users/Peiteng.Chuang/Desktop/璞真RAG/本因坊結案報告(提審)95.08.01.pdf"
    process_pdf_with_marker_api(target_file)