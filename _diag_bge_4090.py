"""4090 上 bge-m3 無聲崩潰診斷。

在 4090 跑 md_review_ui 的『同一個 venv』裡執行（不要在 streamlit 內，純 Python）：
    python _diag_bge_4090.py cuda      # 重現 GPU 載入崩潰
    python _diag_bge_4090.py cpu       # 看 CPU 會不會也崩

faulthandler 會在 segfault 時印出 C 層堆疊（Python 預設不印 → 你才會看到「無聲消失」）。
請把【完整輸出 + 最後 EXIT 行】貼回來。
"""
import faulthandler
import os
import sys
import traceback

faulthandler.enable()  # segfault 時印 native 堆疊

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
print(f"=== DEVICE={DEVICE} | KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')} ===", flush=True)
print("PYTHON:", sys.version, flush=True)

try:
    import torch
    print("TORCH:", torch.__version__, "| cuda?", torch.cuda.is_available(),
          "| cudaver:", getattr(torch.version, "cuda", None), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free/total: {free/1e9:.2f}/{total/1e9:.2f} GB", flush=True)
except Exception:
    traceback.print_exc()

print("--- import FlagEmbedding ---", flush=True)
from FlagEmbedding import BGEM3FlagModel

print("--- loading bge-m3 ---", flush=True)
if DEVICE == "cpu":
    m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")
else:
    m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("--- loaded; encoding ---", flush=True)

o = m.encode(["測試一句話"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
print("ENCODE OK, dense shape:", o["dense_vecs"].shape, flush=True)
print("=== REACHED END (no crash) ===", flush=True)
