"""launch_review_ui.py — md_review_ui 的正式啟動入口（取代 `streamlit run md_review_ui.py`）

用法:
    python launch_review_ui.py            # 等同 streamlit run md_review_ui.py，但已修無聲崩潰
    python launch_review_ui.py --server.port 8502   # 其餘參數會原樣轉給 streamlit

為什麼需要這支 launcher（治本，勿刪）:
    md_review_ui 用 @st.cache_resource 懶載 bge-m3（FlagEmbedding）。該 import 鏈
    FlagEmbedding → datasets → pyarrow.dataset(C 擴充) 的「首次初始化」會被延到
    Streamlit 的 ScriptRunner *worker 執行緒* 才發生；此時主執行緒已在開機時載過
    pyarrow 基礎模組，子模組 C 擴充在 ScriptRunner 執行緒初始化就 access violation
    → 整個 Python 行程直接 abort、終端無 traceback（使用者會看到「按下 embedding 就閃退」）。

    2026-07-16 實測驗證：純 Python(CPU/CUDA)✅、普通 thread✅、ScriptRunner❌(EXIT=139)、
    ScriptRunner + 本檔主執行緒預載✅。非 OpenMP 重複 DLL、非 CUDA OOM。

    修法：在「真正的主執行緒、且 Streamlit 尚未啟動 ScriptRunner 之前」把這些原生庫
    import 完，C 擴充就已常駐；之後 worker 執行緒再 import 只是 no-op → 不再崩潰。
    ⚠️ 把 import 塞進 md_review_ui.py 頂端沒用 —— 目標腳本 top-level 也在 ScriptRunner
    執行緒跑，只會把崩潰提早，必須用這支獨立 launcher 在 CLI 主執行緒預載。

    伺服器設定（fileWatcherType/headless/static 等）仍由 .streamlit/config.toml 掌管，
    本檔不重複帶那些旗標。
"""
import faulthandler
import os
import sys

# 未來若仍有任何原生崩潰，至少印出 C 層堆疊，不再「無聲消失」
faulthandler.enable()

# --- 關鍵：主執行緒預載原生 C 擴充，必須在 import streamlit CLI 之前 ---
print("[launch] 主執行緒預載原生庫 (torch / pyarrow.dataset / datasets / FlagEmbedding)...", flush=True)
import torch  # noqa: F401  對齊既有「入口 top-level import torch」慣例
import pyarrow  # noqa: F401
import pyarrow.dataset  # noqa: F401  ← 真正會在 ScriptRunner 執行緒爆掉的 C 擴充
import datasets  # noqa: F401
from FlagEmbedding import BGEM3FlagModel  # noqa: F401  一併預載更保險（不建模型，只 import）
print("[launch] 原生庫預載完成，啟動 Streamlit...", flush=True)

# 以本檔所在目錄為工作目錄，確保 .streamlit/config.toml、./static、相對路徑都正常解析
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
_APP = os.path.join(_HERE, "md_review_ui.py")

from streamlit.web import cli as stcli

# 把使用者附加的參數（如 --server.port）原樣轉給 streamlit
_extra = sys.argv[1:]
sys.argv = ["streamlit", "run", _APP, *_extra]
sys.exit(stcli.main())
