import json
import hashlib
import subprocess
import platform
import shutil
import torch
import gc
import fitz
import re
from pathlib import Path
from enum import Enum, auto

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

class ProcessStatus(Enum):
    INITIALIZING = auto()
    CONVERTING_FORMAT = auto()
    FITZ_SCANNING = auto()
    MARKER_LOADING = auto()
    MARKER_PROCESSING = auto()
    STITCHING = auto()
    COMPLETED = auto()
    FAILED = auto()

class RAGSmartPipeline:
    def __init__(self, input_path, output_base):
        self.state = ProcessStatus.INITIALIZING
        self.input_path = Path(input_path).absolute()
        self.original_stem = self.input_path.stem
        
        self.tmp_dir = Path("./_process_tmp").absolute()
        self.output_base = Path(output_base).absolute()
        self.img_dir = self.output_base / f"{self.original_stem}_image"
        self.md_file_path = self.output_base / f"{self.original_stem}.md"
        
        self.working_pdf_path = None
        self.md_fragments = []
        self.vector_indices = []
        self.document_main_title = "Unknown"
        
        self.blacklisted_regions = {}     
        self.banned_global_texts = set()  
        self.banned_image_hashes = set()  
        
        self.xref_to_hash_cache = {}
        self.saved_image_hashes = set()
        self.hash_to_filename = {}

    def _get_soffice_path(self):
        if platform.system() == "Windows":
            p = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
            return str(p) if p.exists() else "soffice.exe"
        return "libreoffice"

    def _get_file_hash(self):
        hasher = hashlib.md5()
        with open(self.input_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _pre_scan_for_templates(self, doc):
        total_pages = len(doc)
        if total_pages < 2: return
        prev_page_texts = {}
        image_occurrence = {}

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_w, page_h = page.rect.width, page.rect.height
            curr_page_texts = {}
            blocks = page.get_text("blocks")
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                norm_text = re.sub(r'\s+', '', text.strip())
                if not norm_text or len(norm_text) < 2: continue

                curr_bbox = [x0, y0, x1, y1]
                if norm_text in prev_page_texts:
                    prev_bbox = prev_page_texts[norm_text]
                    dx = abs(x0 - prev_bbox[0]) / page_w
                    dy = abs(y0 - prev_bbox[1]) / page_h
                    if dx <= 0.1 and dy <= 0.1:
                        if norm_text not in self.blacklisted_regions:
                            self.blacklisted_regions[norm_text] = []
                        self.blacklisted_regions[norm_text].append(curr_bbox)
                        self.banned_global_texts.add(norm_text)
                curr_page_texts[norm_text] = curr_bbox
            prev_page_texts = curr_page_texts

            for img in page.get_images(full=True):
                xref = img[0]
                if xref not in self.xref_to_hash_cache:
                    img_data = doc.extract_image(xref)
                    h = hashlib.md5(img_data["image"]).hexdigest()
                    self.xref_to_hash_cache[xref] = h
                h = self.xref_to_hash_cache[xref]
                image_occurrence[h] = image_occurrence.get(h, 0) + 1

        for h, count in image_occurrence.items():
            if count / total_pages > 0.5:
                self.banned_image_hashes.add(h)
        print(f"🚫 已偵測全域雜訊：文字 {len(self.blacklisted_regions)} 組, 圖片 {len(self.banned_image_hashes)} 組。")

    def _get_smart_title_for_page(self, page):
        """
        修正版標題抓取：如果最大字體有多個項目，則不建立標題。
        """
        try:
            blocks = page.get_text("dict")["blocks"]
            candidates = []
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            text = s["text"].strip()
                            if len(text) < 2 or text.isdigit(): continue
                            # 排除清單符號開頭的作為標題
                            if text.startswith(('*', '-', '•', '1.', '2.')): continue
                            
                            candidates.append({
                                "text": text,
                                "size": round(s["size"], 1),
                                "y0": s["bbox"][1]
                            })
            
            if not candidates: return None

            # 依字體大小降序排序
            candidates.sort(key=lambda x: (-x["size"], x["y0"]))
            max_size = candidates[0]["size"]
            
            # 找到所有與最大字體相同的候選者
            max_size_group = [c for c in candidates if abs(c["size"] - max_size) < 0.1]
            
            # --- 邏輯修正：如果最大字體包含多個獨立塊（例如目錄或並列標題），則回傳 None ---
            if len(max_size_group) > 1:
                return None  # 這會導致後續不建立 ### 標題

            if max_size < 13: 
                return "續前頁內容"

            return candidates[0]["text"]
        except Exception:
            return None

    def _clean_text_blocks(self, page, page_idx, page_title):
        blocks = page.get_text("blocks")
        page_w, page_h = page.rect.width, page.rect.height
        unique_texts = []
        seen_in_page = {} 

        # 預先處理標題比對字串
        norm_title = re.sub(r'\s+', '', page_title) if page_title else None

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            norm_text = re.sub(r'\s+', '', text.strip())
            if not norm_text: continue

            # 1. 黑名單排除
            if norm_text in self.blacklisted_regions:
                is_banned = False
                for b_box in self.blacklisted_regions[norm_text]:
                    if abs(x0 - b_box[0])/page_w <= 0.1 and abs(y0 - b_box[1])/page_h <= 0.1:
                        is_banned = True
                        break
                if is_banned: continue

            # 2. 重影過濾
            if norm_text in seen_in_page:
                prev_coords = seen_in_page[norm_text]
                if abs(x0 - prev_coords[0]) < 3 and abs(y0 - prev_coords[1]) < 3:
                    continue 
            seen_in_page[norm_text] = (x0, y0)

            # 3. 標題排除（僅當標題存在時，避免標題出現在內容中兩次）
            if norm_title and norm_text == norm_title:
                continue

            unique_texts.append(text.strip())

        return "\n".join(unique_texts)

    def _extract_fitz_images_dedup(self, page, page_idx, doc, page_content):
        actual_images = page.get_image_info(xrefs=True)
        visible_xrefs = {img['xref'] for img in actual_images if img['bbox'][2] - img['bbox'][0] > 5}
        for img_info in page.get_images(full=True):
            try:
                xref = img_info[0]
                if xref not in visible_xrefs: continue
                img_hash = self.xref_to_hash_cache.get(xref)
                if not img_hash:
                    base_img = doc.extract_image(xref)
                    img_hash = hashlib.md5(base_img["image"]).hexdigest()
                if img_hash in self.banned_image_hashes: continue
                if img_hash in self.saved_image_hashes:
                    name = self.hash_to_filename[img_hash]
                    page_content.append(f"![{name}]({self.img_dir.name}/{name})\n")
                    continue
                base_img = doc.extract_image(xref)
                if base_img["width"] < 60 or base_img["height"] < 60: continue
                img_name = f"{self.original_stem}_h{img_hash[:8]}.{base_img['ext']}"
                with open(self.img_dir / img_name, "wb") as f: f.write(base_img["image"])
                self.saved_image_hashes.add(img_hash)
                self.hash_to_filename[img_hash] = img_name
                page_content.append(f"![{img_name}]({self.img_dir.name}/{img_name})\n")
            except: continue

    def _marker_stage(self, converter):
        from marker.output import text_from_rendered
        doc = fitz.open(self.working_pdf_path)
        for page_idx in self.vector_indices:
            v_doc = fitz.open()
            v_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            tmp_pdf = self.tmp_dir / f"p{page_idx}.pdf"
            v_doc.save(str(tmp_pdf)); v_doc.close()
            with torch.no_grad():
                rendered = converter(str(tmp_pdf))
                page_text, _, images = text_from_rendered(rendered)
            if not page_text:
                continue

            for banned_txt in self.banned_global_texts:
                if banned_txt in page_text:
                    page_text = page_text.replace(banned_txt, "")

            # === 1) 儲存 Marker 產出的圖片並改寫 md 引用 ===
            # Marker 對單頁子 PDF 永遠回 `_page_0_*` 命名，跨頁會撞名 → 加 stem+page prefix。
            if images:
                for orig_name, img_obj in images.items():
                    safe_name = f"{self.original_stem}_p{page_idx + 1}_{orig_name}"
                    target = self.img_dir / safe_name
                    try:
                        if isinstance(img_obj, bytes):
                            target.write_bytes(img_obj)
                        elif hasattr(img_obj, "save"):
                            img_obj.save(target)
                        else:
                            continue
                    except Exception as e:
                        print(f"⚠️ Marker 圖片儲存失敗 P{page_idx+1} {orig_name}: {e}")
                        continue
                    # 改寫所有指向 orig_name 的 ![...](orig_name) 為 ![safe](folder/safe)
                    new_ref = f"![{safe_name}]({self.img_dir.name}/{safe_name})"
                    page_text = re.sub(
                        r"!\[[^\]]*\]\(" + re.escape(orig_name) + r"\)",
                        lambda _m, ref=new_ref: ref,
                        page_text,
                    )

            # === 2) 補抓 h3：fitz 沒給標題時，從 Marker 找 `## **xxx**` 或 `## xxx` ===
            no_h3_marker = f"## 第 {page_idx + 1} 頁\n\n"
            if no_h3_marker in self.md_fragments[page_idx]:
                title_m = re.search(r"^##\s+\*\*(.+?)\*\*\s*$", page_text, re.MULTILINE)
                if not title_m:
                    title_m = re.search(r"^##\s+(.+?)\s*$", page_text, re.MULTILINE)
                if title_m:
                    title = title_m.group(1).strip().strip("*").strip()
                    if 2 <= len(title) <= 80:
                        self.md_fragments[page_idx] = self.md_fragments[page_idx].replace(
                            no_h3_marker,
                            f"## 第 {page_idx + 1} 頁\n### {title}\n\n",
                            1,
                        )
                        # 把那行 ## 從 Marker 內容移掉避免雙標題
                        page_text = page_text.replace(title_m.group(0), "", 1)
                        page_text = re.sub(r"\n{3,}", "\n\n", page_text)

            anchor = f"[[MARKER_REPLACE_P{page_idx}]]"
            self.md_fragments[page_idx] = self.md_fragments[page_idx].replace(anchor, f"\n{page_text}\n")
        doc.close()

    def run(self, converter_instance):
        try:
            self._prepare_env()
            ext = self.input_path.suffix.lower()
            if ext in [".ppt", ".pptx"]:
                self.state = ProcessStatus.CONVERTING_FORMAT
                self._convert_ppt_to_pdf()
            else:
                self.working_pdf_path = self.input_path
            self.state = ProcessStatus.FITZ_SCANNING
            self._fitz_stage()
            if self.vector_indices:
                self.state = ProcessStatus.MARKER_PROCESSING
                self._marker_stage(converter_instance)
            self.state = ProcessStatus.STITCHING
            self._stitching_stage()
            return True
        except Exception as e:
            print(f"❌ 失敗: {self.input_path.name} -> {e}")
            return False
        finally: self._cleanup()

    def _prepare_env(self):
        if self.tmp_dir.exists(): shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        if not self.output_base.exists(): self.output_base.mkdir(parents=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def _convert_ppt_to_pdf(self):
        soffice = self._get_soffice_path()
        subprocess.run([soffice, '--headless', '--convert-to', 'pdf', '--outdir', str(self.tmp_dir), str(self.input_path)], check=True, capture_output=True)
        self.working_pdf_path = self.tmp_dir / f"{self.original_stem}.pdf"

    def _fitz_stage(self):
        doc = fitz.open(self.working_pdf_path)
        self._pre_scan_for_templates(doc)
        
        # 抓取第一頁作為主標題（允許群組標題回傳 None 的處理）
        first_title = self._get_smart_title_for_page(doc[0])
        self.document_main_title = first_title if first_title else "未命名簡報"
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            current_page_title = self._get_smart_title_for_page(page)
            
            # --- 邏輯修正：如果 current_page_title 是 None，就不顯示三級標題列 ---
            page_header = f"## 第 {page_idx + 1} 頁\n"
            if current_page_title:
                page_header += f"### {current_page_title}\n\n"
            else:
                page_header += "\n" # 保持間距但無標題
            
            page_content = [page_header]
            if len(page.get_drawings()) > 200:
                self.vector_indices.append(page_idx)
                page_content.append(f"[[MARKER_REPLACE_P{page_idx}]]\n")
            else:
                cleaned_text = self._clean_text_blocks(page, page_idx, current_page_title)
                page_content.append(f"{cleaned_text}\n\n")
                self._extract_fitz_images_dedup(page, page_idx, doc, page_content)
            
            page_content.append("\n---\n")
            self.md_fragments.append("".join(page_content))
        doc.close()

    def _stitching_stage(self):
        with open(self.md_file_path, "w", encoding="utf-8") as f:
            f.write(f"# {self.input_path.stem}\n\n---\nextracted_main_title: \"{self.document_main_title}\"\nfile_hash: \"{self._get_file_hash()}\"\n---\n\n")
            f.write("".join(self.md_fragments))

    def _cleanup(self):
        if self.tmp_dir.exists(): shutil.rmtree(self.tmp_dir)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

class BatchProcessor:
    def __init__(self, data_root, output_root):
        self.data_root = Path(data_root).absolute()
        self.output_root = Path(output_root).absolute()
        self.tracker_path = self.output_root / "process_tracker.json"
        self.processed_files = self._load_tracker()

    def _load_tracker(self):
        if self.tracker_path.exists():
            with open(self.tracker_path, "r", encoding="utf-8") as f: return json.load(f)
        return {}

    def _save_tracker(self):
        with open(self.tracker_path, "w", encoding="utf-8") as f: json.dump(self.processed_files, f, ensure_ascii=False, indent=4)

    def process_all(self):
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        print("🚀 初始化 Marker 模型環境...")
        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)
        files = []
        for ext in ["*.pdf", "*.ppt", "*.pptx"]: files.extend(list(self.data_root.rglob(ext)))
        for file_path in files:
            file_key = str(file_path.relative_to(self.data_root))
            if self.processed_files.get(file_key) == "SUCCESS": continue
            pipeline = RAGSmartPipeline(file_path, self.output_root)
            if pipeline.run(converter):
                self.processed_files[file_key] = "SUCCESS"
                print(f"✅ 完成: {file_key}")
            else: self.processed_files[file_key] = "FAILED"
            self._save_tracker()

if __name__ == "__main__":
    DATA_PATH = r"C:/Users/PC/Desktop/test_RAG"
    OUTPUT_PATH = r"./mkdata"
    manager = BatchProcessor(DATA_PATH, OUTPUT_PATH)
    manager.process_all()