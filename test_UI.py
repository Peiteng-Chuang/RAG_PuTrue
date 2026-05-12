import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
import torch
import sys
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

import gc

# =========================
# stdout redirect
# =========================
class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)  # 自動滾動

    def flush(self):
        pass


def hello_world():
    print("Hello World")


def check_gpu_status():
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        check_GPU.config(bg="green", activebackground="green")
    else:
        check_GPU.config(bg="red", activebackground="red")

    print("="*30)
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {cuda_available}")
    print("="*30)


def run_stable_conversion():
    IMAGE_DIR = "./img_tmp"
    OUTPUT_FILE = "./output_result.txt"
    PDF_PATH = "C:/project_file/RAG_PuTrue/page_69.pdf"
    os.makedirs(IMAGE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 當前使用的設備: {device}")

    artifact_dict = None
    converter = None
    rendered = None
    images = None

    try:
        print("🚀 正在載入模型至設備...")
        artifact_dict = create_model_dict()

        converter = PdfConverter(
            artifact_dict=artifact_dict,
        )

        print(f"📄 開始解析 PDF: {os.path.basename(PDF_PATH)}")

        with torch.no_grad():  # 👉 避免建立計算圖
            rendered = converter(PDF_PATH)
            text, _, images = text_from_rendered(rendered)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ 文本成功存至: {OUTPUT_FILE}")

        if images:
            for img_name, img_obj in images.items():
                img_path = os.path.join(IMAGE_DIR, img_name)

                if isinstance(img_obj, bytes):
                    with open(img_path, "wb") as f:
                        f.write(img_obj)
                else:
                    img_obj.save(img_path)

            print(f"✅ 成功提取 {len(images)} 張圖片")
        else:
            print("❓ images 為空")

    except Exception as e:
        print(f"❌ 錯誤: {e}")

    finally:
        print("🧹 開始釋放 VRAM...")

        # =========================
        # 🔥 關鍵：刪除所有 reference
        # =========================
        del rendered
        del images
        del converter
        del artifact_dict

        # =========================
        # 🔥 強制 GC
        # =========================
        gc.collect()

        # =========================
        # 🔥 清空 CUDA cache
        # =========================
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✅ VRAM 已釋放")


# =========================
# UI
# =========================
root = tk.Tk()
root.title("PDF Tool")

# 按鈕區
btn = tk.Button(root, text="點我", command=hello_world)
btn.pack()

check_GPU = tk.Button(root, text="檢測 GPU 狀態", bg="lightgray", command=check_gpu_status)
check_GPU.pack()


# 👉 用 thread 包起來避免卡 UI
def run_thread():
    threading.Thread(target=run_stable_conversion).start()

run_chunk = tk.Button(root, text="執行穩定轉換", command=run_thread)
run_chunk.pack()


# =========================
# Log 視窗
# =========================
log_box = ScrolledText(root, height=20, width=100)
log_box.pack()

# stdout redirect
sys.stdout = TextRedirector(log_box)
sys.stderr = TextRedirector(log_box)


root.mainloop()