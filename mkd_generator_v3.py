import json
import hashlib
import subprocess
import platform
import shutil
import torch
import gc
import fitz
from pathlib import Path
from enum import Enum, auto

# --- 狀態機狀態 ---
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
        
        # 圖片去重容器
        self.saved_image_hashes = set()
        self.hash_to_filename = {}

    def _get_soffice_path(self):
        if platform.system() == "Windows":
            p = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
            return str(p) if p.exists() else "soffice.exe"
        return "libreoffice"

    def run(self, converter_instance):
        """執行單一檔案處理，傳入已載入的 converter 以節省時間"""
        try:
            self._prepare_env()
            
            # 1. 格式轉檔
            ext = self.input_path.suffix.lower()
            if ext in [".ppt", ".pptx"]:
                self._convert_ppt_to_pdf()
            else:
                self.working_pdf_path = self.input_path

            # 2. Fitz 掃描與去重提取
            self._fitz_stage()

            # 3. Marker 深度解析 (僅針對向量頁)
            if self.vector_indices:
                self._marker_stage(converter_instance)
            
            # 4. 縫合
            self._stitching_stage()
            return True
        except Exception as e:
            print(f"❌ 處理 {self.input_path.name} 失敗: {e}")
            return False
        finally:
            self._cleanup()

    def _prepare_env(self):
        if self.tmp_dir.exists(): shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def _convert_ppt_to_pdf(self):
        soffice = self._get_soffice_path()
        subprocess.run([
            soffice, '--headless', '--convert-to', 'pdf',
            '--outdir', str(self.tmp_dir), str(self.input_path)
        ], check=True, capture_output=True, timeout=120)
        self.working_pdf_path = self.tmp_dir / f"{self.original_stem}.pdf"

    def _fitz_stage(self):
        doc = fitz.open(self.working_pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_content = [f"## 第[{page_idx}]頁\n\n"]
            
            if len(page.get_drawings()) > 200:
                self.vector_indices.append(page_idx)
                page_content.append(f"[[MARKER_REPLACE_P{page_idx}]]\n")
            else:
                page_content.append(f"{page.get_text().strip()}\n\n")
                # 呼叫去重提取功能
                self._extract_fitz_images_dedup(page, page_idx, doc, page_content)
            
            page_content.append("\n---\n")
            self.md_fragments.append("".join(page_content))
        doc.close()

    def _extract_fitz_images_dedup(self, page, page_idx, doc, page_content):
        # 取得「這張頁面上真正有顯示」的圖片資訊與座標
        actual_images = page.get_image_info(xrefs=True)

        # 建立一個集合，記錄這頁真正有畫出來的 xref
        visible_xrefs = {img['xref'] for img in actual_images if img['bbox'][2] - img['bbox'][0] > 5}

        for img_info in page.get_images(full=True):
            try:
                xref = img_info[0]

                # --- 關鍵修正：如果這張圖在這一頁沒有實際顯示座標，就跳過 ---
                if xref not in visible_xrefs:
                    continue

                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                img_hash = hashlib.md5(img_bytes).hexdigest()

                if img_hash in self.saved_image_hashes:
                    name = self.hash_to_filename[img_hash]
                    # 檢查是否已經在「這一頁」的 page_content 寫過了 (防止一頁內重複引用同圖)
                    link_str = f"![{name}]({self.original_stem}_image/{name})\n"
                    if link_str not in page_content:
                        page_content.append(link_str)
                    continue

                if base_img["width"] < 60 or base_img["height"] < 60: continue

                img_name = f"{self.original_stem}_h{img_hash[:8]}.{base_img['ext']}"
                with open(self.img_dir / img_name, "wb") as f: f.write(img_bytes)

                self.saved_image_hashes.add(img_hash)
                self.hash_to_filename[img_hash] = img_name
                page_content.append(f"![{img_name}]({self.original_stem}_image/{img_name})\n")
            except: continue

    def _marker_stage(self, converter):
        from marker.output import text_from_rendered
        doc = fitz.open(self.working_pdf_path)
        for page_idx in self.vector_indices:
            v_doc = fitz.open()
            v_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            tmp_pdf = self.tmp_dir / f"p{page_idx}.pdf"
            v_doc.save(str(tmp_pdf))
            v_doc.close()

            with torch.no_grad():
                rendered = converter(str(tmp_pdf))
                page_text, _, _ = text_from_rendered(rendered)
            
            anchor = f"[[MARKER_REPLACE_P{page_idx}]]"
            self.md_fragments[page_idx] = self.md_fragments[page_idx].replace(anchor, f"\n{page_text}\n")
        doc.close()

    def _stitching_stage(self):
        with open(self.md_file_path, "w", encoding="utf-8") as f:
            f.write("".join(self.md_fragments))

    def _cleanup(self):
        if self.tmp_dir.exists(): shutil.rmtree(self.tmp_dir)
        torch.cuda.empty_cache()

# --- 檔案管理與狀態追蹤層 ---
class BatchProcessor:
    def __init__(self, data_root, output_root):
        self.data_root = Path(data_root).absolute()
        self.output_root = Path(output_root).absolute()
        self.tracker_path = self.output_root / "process_tracker.json"
        self.processed_files = self._load_tracker()

    def _load_tracker(self):
        if self.tracker_path.exists():
            with open(self.tracker_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_tracker(self):
        with open(self.tracker_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=4)

    def process_all(self):
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        
        print("🚀 正在初始化 Marker 模型...")
        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)

        # 遍歷所有 ppt, pptx, pdf
        extensions = ["*.pdf", "*.ppt", "*.pptx"]
        files_to_process = []
        for ext in extensions:
            files_to_process.extend(list(self.data_root.rglob(ext)))

        print(f"🔍 找到 {len(files_to_process)} 個目標檔案，開始檢查進度...")

        for file_path in files_to_process:
            file_key = str(file_path.relative_to(self.data_root))
            
            # 檢查檔案是否已處理過 (比對名稱即可，或進階比對檔案修改時間)
            if file_key in self.processed_files and self.processed_files[file_key] == "SUCCESS":
                print(f"⏩ 跳過已完成檔案: {file_key}")
                continue

            print(f"\n📂 正在處理: {file_key}")
            pipeline = RAGSmartPipeline(file_path, output_base=self.output_root)
            success = pipeline.run(converter)

            if success:
                self.processed_files[file_key] = "SUCCESS"
                self._save_tracker() # 處理完一個存一次，防止當機損失進度
            else:
                self.processed_files[file_key] = "FAILED"
                self._save_tracker()

        print("\n✨ 所有任務執行完畢！")

if __name__ == "__main__":
    # 設定輸入與輸出路徑
    DATA_PATH = r"D:/璞真RAG資料夾/12.個案銷講資料"
    # DATA_PATH = r"D:/璞真RAG資料夾/20.個案結案報告資料夾"
    OUTPUT_PATH = r"./mkdata"

    manager = BatchProcessor(DATA_PATH, OUTPUT_PATH)
    manager.process_all()