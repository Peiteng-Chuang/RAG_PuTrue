import re
import fitz
import os
import shutil
import torch
import gc
from pathlib import Path

# 導入 Marker 核心組件
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def process_pdf_by_insertion(pdf_path):
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
    
    # --- 階段 1：掃描並建立基礎骨架 ---
    print(f"📂 階段 1：建立基礎骨架...")
    with open(md_file_path, "w", encoding="utf-8") as f:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            is_vector = len(page.get_drawings()) > 200
            
            f.write(f"## 第[{page_idx}]頁\n\n")
            
            if is_vector:
                vector_indices.append(page_idx)
                f.write(f"[[MARKER_INSERT_PAGE_{page_idx}]]\n")
            else:
                f.write(f"{page.get_text().strip()}\n\n")
                # 處理一般圖片
                for img_seq, img_info in enumerate(page.get_images(full=True)):
                    try:
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc, xref)
                        img_filename = f"{pdf_name}_{page_idx}-{img_seq}.png"
                        if pix.n - pix.alpha > 3: pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(str(img_dir / img_filename))
                        f.write(f"![{img_filename}]({pdf_name}_image/{img_filename})\n")
                    except: continue
            
            f.write("\n---\n")

    # --- 階段 2：針對向量頁面「逐頁」執行 Marker ---
    if vector_indices:
        print(f"🚀 階段 2：針對 {len(vector_indices)} 個頁面進行單獨深度解析...")
        
        # 預先載入模型 (放在循環外，避免重複載入)
        artifact_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=artifact_dict)

        for page_idx in vector_indices:
            print(f"正在處理第 {page_idx} 頁...")
            try:
                # 單頁提取
                v_doc = fitz.open()
                v_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                tmp_single_pdf = tmp_dir / f"temp_p{page_idx}.pdf"
                v_doc.save(str(tmp_single_pdf))
                v_doc.close()

                # Marker 解析
                with torch.no_grad():
                    rendered = converter(str(tmp_single_pdf))
                    page_text, _, marker_images = text_from_rendered(rendered)

                # 圖片語法修正
                def fix_image_syntax(match):
                    orig_img_name = match.group(1)
                    new_img_name = f"{pdf_name}_vector_p{page_idx}_{orig_img_name}"
                    return f"![{new_img_name}]({pdf_name}_image/{new_img_name})"

                page_text = re.sub(r"!\[\]\((.*?)\)", fix_image_syntax, page_text)

                # 儲存 Marker 圖片
                if marker_images:
                    for img_name, img_obj in marker_images.items():
                        # 過濾小碎圖
                        if hasattr(img_obj, 'size') and (img_obj.size[0] < 200 or img_obj.size[1] < 200):
                            continue
                        
                        v_full_name = f"{pdf_name}_vector_p{page_idx}_{img_name}"
                        if isinstance(img_obj, bytes):
                            with open(img_dir / v_full_name, "wb") as f: f.write(img_obj)
                        else:
                            img_obj.save(img_dir / v_full_name)

                # 讀取當前 MD 並精準替換該頁錨點
                with open(md_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                placeholder = f"[[MARKER_INSERT_PAGE_{page_idx}]]"
                insertion = f"> [!IMPORTANT] 向量圖深度解析：\n{page_text.strip()}\n"
                content = content.replace(placeholder, insertion)

                with open(md_file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # 每處理完一頁，清理一次顯存
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"第 {page_idx} 頁解析失敗: {e}")

        # 結束後徹底銷毀模型
        del converter
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    doc.close()
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
    print(f"🎊 單頁精準處理完成！")

if __name__ == "__main__":
    process_pdf_by_insertion("D:/璞真RAG資料夾/12.個案銷講資料/文林中正/1110119_璞真文林中正案_建築銷售簡報.pdf")