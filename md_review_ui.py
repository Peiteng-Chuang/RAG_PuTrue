"""
md_review_ui.py — PDF ↔ Markdown 即時比對 + 圖片管理 + 向量資料庫 chunk/metadata 預覽

用法:
    pip install streamlit
    streamlit run md_review_ui.py

依賴: streamlit, PyMuPDF (fitz)。PPT/PPTX 比對需要 LibreOffice。

圖片刪除採「軟刪除＋垃圾桶」策略：
- 刪除 = 從 md 移除引用 + 將檔案移至 ./_md_trash/{stem}/
- 復原 = 從垃圾桶搬回 + 在原位置重新插入 md 引用
"""
# ⚠️ torch / torchvision 必須「第一優先級」import，早於 fitz / numpy / streamlit 等任何
# 原生擴充。否則其他函式庫會先載入衝突的 DLL 依賴，導致 torch 的 c10.dll 初始化失敗：
#   [WinError 1114] 動態連結程式庫 (DLL) 初始化例行程序失敗。
#   Error loading "...\torch\lib\c10.dll" or one of its dependencies.
# 讓 torch 先乾淨載好自己的 DLL，後續 FlagEmbedding/bge-m3 才能正常載入。勿調整此順序。
import torch  # noqa: F401  DLL 載入順序，勿移動
import torchvision  # noqa: F401  DLL 載入順序，勿移動

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

import fitz
import numpy as np
import streamlit as st

# 對齊 pipeline.ipynb / LLM_prompt_test.py：用 .env 統一 ollama 設定
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 選用：VSCode 級編輯器（缺失時 fallback 至 st.text_area）
try:
    from streamlit_ace import st_ace
    HAS_ACE = True
except ImportError:
    st_ace = None
    HAS_ACE = False

# Qdrant client（P3 上傳用；缺失時 Tab 5 會 disable）
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    HAS_QDRANT = True
except ImportError:
    QdrantClient = None
    qm = None
    HAS_QDRANT = False

# vLLM（OpenAI 相容 API）+ ChatBot（對話模式用；缺失時切到對話模式會 disable）
try:
    from openai import OpenAI
    from llm_chat import (
        ChatBot,
        DEFAULT_SYSTEM_PROMPT as LLM_DEFAULT_SYSTEM_PROMPT,
        RAG_SYSTEM_PROMPT as LLM_RAG_SYSTEM_PROMPT,
        COMPARE_SYSTEM_PROMPT as LLM_COMPARE_SYSTEM_PROMPT,
        STRUCTURED_SYSTEM_PROMPT as LLM_STRUCTURED_SYSTEM_PROMPT,
        format_chunks as llm_format_chunks,
        generate_hyde as llm_generate_hyde,
        generate_title as llm_generate_title,
        analyze_query as llm_analyze_query,
        QueryPlan,
    )
    HAS_LLM = True
except ImportError:
    OpenAI = None
    ChatBot = None
    LLM_DEFAULT_SYSTEM_PROMPT = ""
    LLM_RAG_SYSTEM_PROMPT = ""
    LLM_COMPARE_SYSTEM_PROMPT = ""
    LLM_STRUCTURED_SYSTEM_PROMPT = ""
    llm_format_chunks = None
    llm_generate_hyde = None
    llm_generate_title = None
    llm_analyze_query = None
    QueryPlan = None
    HAS_LLM = False

# 第 2 層結構化屬性表（人工維護、熱載入；downstream-only）
try:
    import structured_table as _structured_table
    HAS_STRUCTURED = True
except ImportError:
    _structured_table = None
    HAS_STRUCTURED = False
# 預設路徑：專案目錄下 structured_data/建案屬性表.csv（可在 sidebar 覆寫）
STRUCTURED_TABLE_PATH = str(Path(__file__).resolve().parent / "structured_data" / "建案屬性表.csv")

# 對話紀錄「瀏覽器 localStorage」持久化（退化版：不再寫 server 磁碟，多人共用 server 時各自獨立）
try:
    from streamlit_local_storage import LocalStorage
    HAS_LOCALSTORAGE = True
except Exception:  # noqa: BLE001 — 缺元件時退回純記憶體（session_state），不擋啟動
    LocalStorage = None
    HAS_LOCALSTORAGE = False


# === 路徑設定 ===
# ⚠️ DATA_ROOT（= file_key 相對基底），必須是「文件種類資料夾的父層」，不可含 doc_type 那層。
# file_key = 類別/建案/檔（含類別），故開檔 data_root/file_key 才對得上；若這裡多包一層類別，
# 12 會雙重巢狀、20 會被塞進 12 底下 → 開檔全部「本機未找到」（檢索不受影響）。
DEFAULT_DATA_PATH = r"D:/璞真RAG資料夾"
# .env：DATA_ROOT = 原始資料根目錄（= file_key 相對基底）；DATA_DOC_TYPE = 其下的文件種類
# 資料夾（逗號分隔）。完整路徑 = DATA_ROOT / <選定的 DATA_DOC_TYPE> / …。審閱模式用下拉選
# DATA_DOC_TYPE，清單與 fast pipeline 只含「該資料夾底下、且實體存在」的檔。
DATA_ROOT_ENV = (os.getenv("DATA_ROOT", "") or "").strip()
DATA_DOC_TYPES = [s.strip() for s in (os.getenv("DATA_DOC_TYPE", "") or "").split(",") if s.strip()]
MKDATA_PATH = Path("./mkdata")
TRACKER = MKDATA_PATH / "process_tracker.json"
CHAT_LS_KEY = "putrue_chat_sessions"  # 對話紀錄存放於瀏覽器 localStorage 的 item key（取代舊的 server 磁碟檔）
REVIEW_PASSWORD = "123456"  # 進入審閱模式（ETL 標記工具）的密碼鎖
PDF_CACHE_DIR = Path("./_compare_cache")
TRASH_DIR = Path("./_md_trash")
PDF_CACHE_DIR.mkdir(exist_ok=True)
TRASH_DIR.mkdir(exist_ok=True)


# === 系統常數（對齊 qdrant格式.md v2.1.0） ===
PIPELINE_VERSION = "2.1.1"   # 2.1.1：project_name=file_key[1]（建案）、doc_type fallback=file_key[0]（類別）
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_VERSION = "v1"
EMBEDDING_DIM_DENSE = 1024
SIDECAR_VERSION = "1.0"
CHUNKING_STRATEGY = "heading+token-v2"
# 文件種類（doc_type）：以「資料夾 → doc_type」對照表為主、sidecar 逐檔覆寫為輔，皆無則用預設。
# 對照表只是一檔小 JSON（一個資料夾一筆），避免逐檔 bulk 寫 sidecar。
DOC_TYPE_MAP_PATH = MKDATA_PATH / "doc_type_map.json"
DOC_TYPE_DEFAULT = "未分類"
REVIEW_STATUSES = ["unprocessed", "processing", "encoded", "ingested"]
STATUS_BADGE = {
    "unprocessed": {"emoji": "🔴", "label": "未處理",   "bg": "#fdecec", "fg": "#c0392b"},
    "processing":  {"emoji": "🟡", "label": "處理中",   "bg": "#fff5d6", "fg": "#a86b00"},
    "encoded":     {"emoji": "🟢", "label": "處理完",   "bg": "#e3f7e3", "fg": "#27713a"},
    "ingested":    {"emoji": "🔵", "label": "已寫入庫", "bg": "#d6ecf7", "fg": "#1f5fa8"},
}
# 舊版 sidecar 欄位值的相容遷移
LEGACY_STATUS_MIGRATION = {
    "unreviewed":   "unprocessed",
    "in_progress":  "processing",
    "approved":     "ingested",
    "needs_rework": "processing",
    "done":         "ingested",   # 舊 3 態的 done 對齊新 4 態的 ingested
}
AVAILABLE_EMBEDDERS = ["BAAI/bge-m3"]  # 之後可加 Qwen3-Embedding-4B 等

# === Qdrant 常數（對齊 qdrant格式.md §2）===
QDRANT_TEXT_COLLECTION = "putrue_rag_text_v1"
QDRANT_DEFAULT_URL = "http://localhost:6333"
# vLLM（OpenAI 相容）設定優先序：.env > 內建 fallback
LLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.201.66:8000/v1")
LLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM 不驗 key，但 OpenAI client 需非空字串
LLM_DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "")  # 實際 model id 由 /v1/models 動態抓
# .env 內建 model 候選（抓不到 /v1/models 時的 fallback）
_env_model_candidates = [
    os.getenv("LLM_MODEL_NAME"),
    os.getenv("VLM_MODEL_NAME"),
]
LLM_MODEL_CHOICES = [m for m in _env_model_candidates if m]
LLM_DEFAULT_TOP_K = 8
# 三路意圖 router（第 1 層，見 memory: rag_comparison_routing_design）
LLM_COMPARE_TOP_K_PER_ENTITY = 4   # compare/rank fan-out：每個建案各撈幾條
LLM_MAX_FANOUT_ENTITIES = 12       # fan-out 建案數上限（超過截斷 + UI 提示，不靜默）
LLM_DEFAULT_TEMPERATURE = 0.7   # 對齊 pipeline.ipynb + llm_chat.ChatBot 預設
LLM_DEFAULT_TOP_P = 0.9
LLM_DEFAULT_NUM_PREDICT = 4096
LLM_DEFAULT_NUM_CTX = 32768
QDRANT_BATCH_SIZE = 100


# === LibreOffice 偵測 ===
import os

LIBREOFFICE_CANDIDATES_WIN = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    r"D:\Program Files\LibreOffice\program\soffice.exe",
    r"D:\LibreOffice\program\soffice.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\LibreOffice\program\soffice.exe"),
]


def find_soffice(override: str | None = None) -> str | None:
    """回傳第一個可用的 LibreOffice 執行檔路徑；找不到回傳 None。"""
    if override:
        op = Path(override)
        if op.exists():
            return str(op)
        return None
    if platform.system() == "Windows":
        for cand in LIBREOFFICE_CANDIDATES_WIN:
            if cand and Path(cand).exists():
                return cand
        # PATH 兜底
        for name in ("soffice.exe", "soffice"):
            found = shutil.which(name)
            if found:
                return found
        return None
    found = shutil.which("libreoffice") or shutil.which("soffice")
    return found


def _lo_profiles_root() -> Path:
    """所有「每次轉檔用完即丟」的臨時 profile 的母目錄。"""
    root = (PDF_CACHE_DIR / "_lo_profiles").absolute()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_soffice_convert(soffice_path: str, source_path: Path, profile_dir: Path):
    """用一個指定的、獨立的 user profile 跑一次 headless 轉檔。回 CompletedProcess。

    profile 獨立（每次呼叫換新）是可靠性關鍵：LibreOffice headless 在 Windows 上
    常因『共用 profile 殘留鎖檔 / 髒狀態』或『多實例並發共用同一 profile』而以 heap
    corruption（exit=0xC0000374 / 3221226356）崩潰。不同 UserInstallation 會被當成
    不同實例 → 可並發、且永不被前一次的殘留鎖檔毒化。另加數個硬化旗標。
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_uri = "file:///" + str(profile_dir.absolute()).replace("\\", "/")
    cmd = [
        soffice_path,
        f"-env:UserInstallation={profile_uri}",
        "--headless", "--norestore", "--nolockcheck",
        "--nologo", "--nofirststartwizard",
        "--convert-to", "pdf",
        "--outdir", str(PDF_CACHE_DIR),
        str(source_path),
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=180,
        encoding="utf-8", errors="replace",
    )


def ensure_pdf(source_path: Path, soffice_path: str | None = None) -> Path:
    ext = source_path.suffix.lower()
    if ext == ".pdf":
        return source_path
    if ext not in [".ppt", ".pptx"]:
        raise ValueError(f"不支援的副檔名: {ext}")

    cached = PDF_CACHE_DIR / f"{source_path.stem}.pdf"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    if not soffice_path:
        raise RuntimeError(
            "找不到 LibreOffice。請至 https://www.libreoffice.org/download/ 安裝，"
            "或在左側 sidebar『LibreOffice 路徑』手動指定 soffice.exe。"
        )

    # 每次轉檔給一個全新、獨立的臨時 profile；失敗（含 heap-corruption 崩潰）就換另一個
    # 全新 profile 再試一次。每次用完即刪，不留鎖檔毒化後續轉檔。
    last_err = ""
    result = None
    for _attempt in range(2):
        profile_dir = Path(tempfile.mkdtemp(prefix="lo_", dir=str(_lo_profiles_root())))
        try:
            result = _run_soffice_convert(soffice_path, source_path, profile_dir)
        except FileNotFoundError:
            raise RuntimeError(f"LibreOffice 路徑無效：{soffice_path}")
        except subprocess.TimeoutExpired:
            last_err = "LibreOffice 轉檔逾時 (>180s)，可能有 LO 實例卡住。"
            result = None
            continue
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)

        if result.returncode == 0:
            break
        last_err = (
            f"LibreOffice 轉檔失敗 (exit={result.returncode})\n"
            f"stderr: {(result.stderr or '').strip()[:500]}\n"
            f"stdout: {(result.stdout or '').strip()[:200]}"
        )

    if result is None or result.returncode != 0:
        hint = ""
        if result is not None and (result.returncode & 0xFFFFFFFF) == 0xC0000374:
            hint = (
                "\n\n提示：exit=0xC0000374 是 Windows heap corruption，soffice 程序崩潰。"
                "常見原因：(1) 同機另有 LibreOffice 視窗/程序在跑 → 請關閉；"
                "(2) LibreOffice 版本過舊或安裝損毀 → 重裝最新版；"
                "(3) 該 .pptx 本身含 LO 無法處理的內容。本程式已每次用獨立 profile + 自動重試一次。"
            )
        raise RuntimeError((last_err or "LibreOffice 轉檔失敗") + hint)

    if cached.exists() and cached.stat().st_size > 0:
        return cached

    # 寬鬆比對：LO 可能因為檔名特殊字元而輸出不同名字
    siblings = sorted(
        (p for p in PDF_CACHE_DIR.glob("*.pdf") if p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if siblings:
        latest = siblings[0]
        if latest != cached:
            try:
                latest.rename(cached)
                return cached
            except Exception:
                return latest
        return latest

    raise RuntimeError(
        f"LibreOffice 回報成功但找不到輸出 PDF。\n"
        f"預期: {cached.name}\n"
        f"stderr: {(result.stderr or '').strip()[:500]}"
    )


@st.cache_data
def load_tracker():
    with open(TRACKER, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=30)
def list_processed_files_in_dir(doc_dir_str: str, data_root_str: str, mkdata_str: str) -> list[str]:
    """掃實體目錄 doc_dir 下的 PDF/PPT/PPTX，只回「mkdata 已有對應 {stem}.md（已處理）」的檔。

    回傳 file_key = 相對 data_root 的路徑（= 文件種類/建案/檔，與 rebuild_tracker/v5 一致）。
    以「實體目錄」為清單來源 → 每個實體檔只出現一次，天然去重，不受 tracker 殘留/重複 key 影響。
    set() 再排序去重。cache 30s 避免每次 rerun 重掃整個目錄樹。
    """
    doc_dir, data_root, mkdata = Path(doc_dir_str), Path(data_root_str), Path(mkdata_str)
    out: list[str] = []
    if doc_dir.exists():
        for ext in ("*.pdf", "*.ppt", "*.pptx"):
            for p in doc_dir.rglob(ext):
                if p.is_file() and (mkdata / f"{p.stem}.md").exists():
                    out.append(str(p.relative_to(data_root)))
    return sorted(set(out))


def split_key_parts(file_key: str) -> list[str]:
    return [p for p in re.split(r"[\\/]+", file_key) if p]


def derive_md_path(file_key: str) -> Path:
    file_name = split_key_parts(file_key)[-1]
    stem = Path(file_name).stem
    return MKDATA_PATH / f"{stem}.md"


# === Sidecar 持久化（review 狀態 / 自訂標籤 / 刪除歷史） ===
def derive_sidecar_path(file_key: str) -> Path:
    file_name = split_key_parts(file_key)[-1]
    stem = Path(file_name).stem
    return MKDATA_PATH / f"{stem}.review.json"


def compute_file_hash(source_path: Path) -> str:
    """來源檔 MD5；找不到或讀不到回空字串。1MB 一塊串流避免大檔吃記憶體。"""
    if not source_path.exists():
        return ""
    try:
        h = hashlib.md5()
        with open(source_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


def sidecar_default(file_key: str) -> dict:
    return {
        "version": SIDECAR_VERSION,
        "file_key": file_key,
        "file_hash": "",
        "review_status": "unprocessed",
        "reviewer": "",
        "reviewed_at": None,
        "custom_labels": {"document": [], "pages": {}},
        "split_settings": {"mode": "delimiter", "delim": "\\n"},
        "delete_history": [],
    }


def load_sidecar(file_key: str) -> dict:
    """讀 sidecar；無檔或解析失敗回空白骨架（不寫盤）。"""
    path = derive_sidecar_path(file_key)
    if not path.exists():
        return sidecar_default(file_key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        st.warning(f"sidecar {path.name} 解析失敗，改用空白狀態：{e}")
        return sidecar_default(file_key)
    # 補欄位（前向相容）
    defaults = sidecar_default(file_key)
    for k, v in defaults.items():
        data.setdefault(k, v)
    # 舊版狀態值遷移
    raw_status = data.get("review_status", "unprocessed")
    if raw_status in LEGACY_STATUS_MIGRATION:
        data["review_status"] = LEGACY_STATUS_MIGRATION[raw_status]
    elif raw_status not in REVIEW_STATUSES:
        data["review_status"] = "unprocessed"
    return data


def save_sidecar(file_key: str, sidecar: dict) -> None:
    """原子寫入：tmp + rename。"""
    path = derive_sidecar_path(file_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_doc_type_map() -> dict:
    """讀「資料夾 → doc_type」對照表；無檔或壞檔回空 dict。"""
    if not DOC_TYPE_MAP_PATH.exists():
        return {}
    try:
        data = json.loads(DOC_TYPE_MAP_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_doc_type_map(mapping: dict) -> None:
    """原子寫入「資料夾 → doc_type」對照表（tmp + rename）。"""
    DOC_TYPE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DOC_TYPE_MAP_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DOC_TYPE_MAP_PATH)


def resolve_doc_type(file_key: str, sidecar: dict, doc_type_map: dict | None = None) -> str:
    """決定某檔的文件種類，優先序：sidecar 逐檔覆寫 > 資料夾對照表 > **最上層資料夾名** > 未分類。

    「資料夾名即類型」：data_root 設在最上層時 file_key[0] = 類別資料夾名，預設直接拿它當
    doc_type（零維護）。對照表只在「資料夾名 ≠ 想要的 doc_type」時用來翻譯（覆寫）。
    傳入 doc_type_map 可避免在批次 chunk 迴圈裡反覆讀檔；未傳則自行載入。
    """
    override = (sidecar or {}).get("doc_type")
    if override:
        return str(override)
    parts = split_key_parts(file_key)
    folder = parts[0] if len(parts) > 1 else ""
    dtm = doc_type_map if doc_type_map is not None else load_doc_type_map()
    return str(dtm.get(folder) or folder or DOC_TYPE_DEFAULT)


def derive_project_name(file_key: str) -> str:
    """建案名 = file_key 第二段（結構為 類別/建案/…/檔 → 取 建案）。

    類別 = file_key[0]（給 doc_type），建案 = file_key[1]，兩者正交。
    無建案層（類別/檔）退回第一段；只有檔名退回未分類。
    """
    parts = split_key_parts(file_key)
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    return "未分類"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_file_status_quick(file_key: str) -> str:
    """直接從 sidecar JSON 撈 review_status，不經 session_state、不污染。
    sidebar 列表用：每 rerun 對所有檔案掃一次 disk。"""
    sp = derive_sidecar_path(file_key)
    if not sp.exists():
        return "unprocessed"
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unprocessed"
    raw = data.get("review_status", "unprocessed")
    raw = LEGACY_STATUS_MIGRATION.get(raw, raw)
    return raw if raw in REVIEW_STATUSES else "unprocessed"


def mark_processing(file_key: str) -> None:
    """編輯動作觸發：unprocessed/encoded/ingested → processing。同時 persist sidecar。
    已是 processing 則 no-op，避免多餘磁碟寫入。
    從 encoded/ingested 進入時記下 prior_status，供 捨棄編輯 還原。"""
    if not file_key:
        return
    sidecar = st.session_state.sidecars.get(file_key)
    if sidecar is None:
        return
    if sidecar.get("review_status") == "processing":
        persist_review_state(file_key)
        return
    prev = sidecar.get("review_status")
    if prev in ("encoded", "ingested"):
        sidecar["prior_status"] = prev
        sidecar["disk_dirty"] = False
    sidecar["review_status"] = "processing"
    persist_review_state(file_key)


def mark_encoded(file_key: str) -> None:
    """Tab 4 全檔批次 encode 成功觸發：→ encoded（綠）。
    代表本地向量已建好、reviewer 至少切到末頁；下一步應該 Tab 5 上傳。"""
    if not file_key:
        return
    sidecar = st.session_state.sidecars.get(file_key)
    if sidecar is None:
        return
    sidecar["review_status"] = "encoded"
    sidecar.pop("prior_status", None)
    sidecar.pop("disk_dirty", None)
    persist_review_state(file_key)


def mark_ingested(file_key: str) -> None:
    """Tab 5 Qdrant 上傳成功觸發：→ ingested（淡藍），並蓋時間戳。
    Qdrant 收下整檔 vectors + payload 後才會走到這。"""
    if not file_key:
        return
    sidecar = st.session_state.sidecars.get(file_key)
    if sidecar is None:
        return
    sidecar["review_status"] = "ingested"
    sidecar["reviewed_at"] = now_iso()
    sidecar.pop("prior_status", None)
    sidecar.pop("disk_dirty", None)
    persist_review_state(file_key)


def set_status_on_disk(file_key: str, sidecar: dict, status: str) -> None:
    """批次（fast pipeline）用：直接改 sidecar dict 的 review_status 並寫盤。

    與 mark_encoded/mark_ingested 不同 —— 那些依賴 st.session_state.sidecars（只有
    『目前開啟的檔』才在裡面），對未開啟的檔會 no-op。批次處理的是磁碟上的檔，
    必須走這條 disk-direct 路徑。傳入的 sidecar 應為 load_sidecar() 取得的同一份。
    """
    sidecar["review_status"] = status
    if status == "ingested":
        sidecar["reviewed_at"] = now_iso()
    sidecar.pop("prior_status", None)
    sidecar.pop("disk_dirty", None)
    save_sidecar(file_key, sidecar)
    # 若該檔剛好也開在前景，同步 session_state 副本避免狀態列顯示過時
    mem = st.session_state.get("sidecars", {}).get(file_key)
    if mem is not None:
        mem["review_status"] = status


def persist_review_state(file_key: str) -> None:
    """把目前 session_state 內的編輯狀態同步寫回 sidecar。"""
    if not file_key:
        return
    sidecar = st.session_state.sidecars.get(file_key)
    if sidecar is None:
        return
    labels = st.session_state.custom_labels.get(file_key, {"document": [], "pages": {}})
    sidecar["custom_labels"] = {
        "document": list(labels.get("document", [])),
        # JSON key 必須是 str
        "pages": {
            str(k): list(v)
            for k, v in labels.get("pages", {}).items()
            if v
        },
    }
    sidecar["split_settings"] = dict(
        st.session_state.split_settings.get(
            file_key, {"mode": "delimiter", "delim": "\\n"}
        )
    )
    sidecar["delete_history"] = list(
        st.session_state.delete_history.get(file_key, [])
    )
    save_sidecar(file_key, sidecar)


# === Qdrant point id / section id ===
def make_point_id(file_hash: str, page_num: int, section_idx: int, chunk_idx: int) -> str:
    seed = f"{file_hash or 'NOHASH'}|{page_num}|{section_idx}|{chunk_idx}"
    return str(uuid5(NAMESPACE_DNS, seed))


def make_section_id(file_hash: str, page_num: int, section_idx: int) -> str:
    seed = f"{file_hash or 'NOHASH'}|{page_num}|section|{section_idx}"
    return str(uuid5(NAMESPACE_DNS, seed))


def parse_md(md_text: str):
    fm = {}
    fm_match = re.search(r"^---\n(.*?)\n---\n", md_text, re.DOTALL | re.MULTILINE)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            kv = line.split(":", 1)
            if len(kv) == 2:
                fm[kv[0].strip()] = kv[1].strip().strip('"')

    pattern = re.compile(r"^## 第 (\d+) 頁\n", re.MULTILINE)
    matches = list(pattern.finditer(md_text))
    if not matches:
        return fm, md_text, []

    pages = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        pages.append((page_num, md_text[start:end]))
    header = md_text[: matches[0].start()]
    return fm, header, pages


@st.cache_data(max_entries=256, show_spinner=False)
def _render_pdf_page_png_cached(pdf_path_str: str, mtime: float, page_idx: int, dpi: int) -> bytes:
    doc = fitz.open(pdf_path_str)
    try:
        if page_idx < 0 or page_idx >= len(doc):
            raise IndexError(f"頁碼超出範圍 (0-{len(doc)-1})")
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page_idx].get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_pdf_page_png(pdf_path: Path, page_idx: int, dpi: int = 110) -> bytes:
    mtime = pdf_path.stat().st_mtime
    return _render_pdf_page_png_cached(str(pdf_path), mtime, page_idx, dpi)


# 「開啟原圖」用：把整頁高解析 PNG 落到 Streamlit 靜態服務目錄，回傳可用 <a target=_blank>
# 開新分頁的真實 URL。這是 server 端按鈕唯一能可靠開新分頁的方式——真實 http URL 不會被
# 瀏覽器擋（不像 data:/blob:）。需 .streamlit/config.toml 開 enableStaticServing。
# 目錄專用，只放/刪本功能自產的 png，絕不碰 ./static 下其他內容（如 fonts/）。
#
# 錨定到「本腳本所在目錄/static」，而非 CWD：Streamlit 以 os.path.abspath(script)
# 的 parent/static 作為靜態服務根（與行程 CWD 無關），用 __file__ 對齊才不會把檔案
# 寫到「服務不到」的目錄（正是先前 /app/static 404 的元兇之一）。
PAGE_PREVIEW_DIR = Path(__file__).resolve().parent / "static" / "_page_preview"
_PAGE_PREVIEW_KEEP = 60  # 只保留最新 N 張，避免無限長大


def _prune_page_preview_dir(keep: int = _PAGE_PREVIEW_KEEP) -> None:
    try:
        pngs = sorted(
            PAGE_PREVIEW_DIR.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in pngs[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def high_dpi_preview_url(pdf_path: Path, page_idx: int, dpi: int) -> str:
    """渲染整頁 PNG 落到 ./static/_page_preview，回傳 `app/static/...` 相對 URL。

    檔名＝(絕對路徑, mtime, 頁碼, dpi) 的 hash → 同頁同 DPI 重用同檔、不重覆寫；PDF 內容
    變動（mtime 變）會自然換新檔。回傳相對 URL（非絕對 /app/...）以相容部屬 baseUrlPath。
    """
    mtime = pdf_path.stat().st_mtime
    token = f"{pdf_path.resolve()}|{mtime}|{page_idx}|{dpi}"
    name = hashlib.md5(token.encode("utf-8")).hexdigest()[:16] + ".png"
    out = PAGE_PREVIEW_DIR / name
    if not out.exists():
        PAGE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out.write_bytes(render_pdf_page_png(pdf_path, page_idx, dpi=dpi))
        _prune_page_preview_dir()
    return f"app/static/{PAGE_PREVIEW_DIR.name}/{name}"


# 原檔下載用的 MIME 對照（用於「下載原檔」鈕）。
_DOWNLOAD_MIME = {
    ".pdf": "application/pdf",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def strip_page_header(page_md: str) -> str:
    body = re.sub(r"^## 第 \d+ 頁\n", "", page_md)
    body = re.sub(r"\n---\s*\n?$", "", body)
    return body.strip()


# === 圖片解析／編輯 ===
# 規格：v4 一定每行一個 ![alt](path)，alt/path 可包含 ()，但不會跨行也不含 ]。
# 用 greedy `.+` + 行尾錨點，確保路徑中的 () 不會把 match 切斷。
IMG_REF_RE = re.compile(
    r"!\[(?P<alt>[^\]\n]*)\]\((?P<path>.+)\)[ \t]*$",
    re.MULTILINE,
)


def parse_images_in_page(page_md: str) -> list[dict]:
    """回傳本頁所有 ![]() 的位置與內容。"""
    return [
        {
            "alt": m.group("alt"),
            "md_path": m.group("path"),
            "match_start": m.start(),
            "match_end": m.end(),
            "full_match": m.group(0),
        }
        for m in IMG_REF_RE.finditer(page_md)
    ]


def image_folder_for(file_key: str, image_root: Path) -> Path:
    """每個文件對應的 `{stem}_image` 資料夾。"""
    stem = Path(split_key_parts(file_key)[-1]).stem
    return image_root / f"{stem}_image"


def resolve_image_path(file_key: str, md_path_in_md: str, image_root: Path) -> Path:
    """
    自動對應規則：
        image_root / "{file_stem}_image" / basename(md_path_in_md)

    只取 md 中 ![](...) 的檔名部分，忽略前綴目錄 — 圖片實際位置由
    「圖片根目錄 + 檔名衍生子資料夾」決定。
    """
    basename = Path(md_path_in_md.replace("\\", "/")).name
    return (image_folder_for(file_key, image_root) / basename).resolve()


def remove_image_line(page_md: str, ref: dict) -> tuple[str, str, int]:
    """從 page_md 移除整行（含換行）。回傳 (new_md, removed_line, line_start_offset)。"""
    line_start = page_md.rfind("\n", 0, ref["match_start"]) + 1
    line_end_idx = page_md.find("\n", ref["match_end"])
    line_end = len(page_md) if line_end_idx == -1 else line_end_idx + 1
    removed = page_md[line_start:line_end]
    new_md = page_md[:line_start] + page_md[line_end:]
    return new_md, removed, line_start


def insert_at(page_md: str, text: str, offset: int) -> str:
    offset = min(offset, len(page_md))
    return page_md[:offset] + text + page_md[offset:]


def is_ref_used_elsewhere(md_text: str, current_page_idx: int, img_md_path: str) -> bool:
    _, _, all_pages = parse_md(md_text)
    needle = re.compile(r"!\[[^\]]*\]\(" + re.escape(img_md_path) + r"\)")
    for i, (_, p_md) in enumerate(all_pages):
        if i == current_page_idx:
            continue
        if needle.search(p_md):
            return True
    return False


def trash_subdir_for(file_key: str) -> Path:
    stem = Path(split_key_parts(file_key)[-1]).stem
    d = TRASH_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def move_to_trash(abs_path: Path, file_key: str) -> Path:
    sub = trash_subdir_for(file_key)
    target = sub / abs_path.name
    counter = 1
    while target.exists():
        target = sub / f"{abs_path.stem}__{counter}{abs_path.suffix}"
        counter += 1
    shutil.move(str(abs_path), str(target))
    return target


def restore_from_trash(trash_path: Path, original_path: Path):
    original_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(trash_path), str(original_path))


def replace_page(md_text: str, page_idx: int, new_page_md: str) -> str:
    fm, header, pages = parse_md(md_text)
    if page_idx >= len(pages):
        return md_text
    if not new_page_md.endswith("\n"):
        new_page_md += "\n"
    new_pages = [(p[0], new_page_md) if i == page_idx else p for i, p in enumerate(pages)]
    return header + "".join(p[1] for p in new_pages)


# === Heading 與切分 ===
# `(?!#)` 防止把 `##`、`####` 也吃進來；`\s*` 容許 `###foo` 這種無空格寫法
H1_RE = re.compile(r"^#(?!#)\s*(.+?)\s*$", re.MULTILINE)
# h3-h6 通用 regex：捕捉 `#` 數量決定階級。h1/h2 由文件結構固定，不參與 section 切分。
HEADING_RE = re.compile(r"^(#{3,6})(?!#)\s*(.+?)\s*$", re.MULTILINE)


def extract_h1(md_text: str, fallback: str) -> str:
    """整份 md 第一個 `# ...`，找不到回 fallback。"""
    m = H1_RE.search(md_text)
    return m.group(1).strip() if m else fallback


def strip_image_refs(text: str) -> str:
    """
    移除 markdown 圖片引用，避免進入 chunk 文字（路徑字串對向量檢索無幫助；
    圖片資訊已透過 metadata.image_paths 保留）。
    - 整行就是 ![]() → 連同換行去掉
    - 行內 inline ![]() → 只去引用本身
    - 連續空行壓回 1 個
    """
    line_only = re.compile(
        r"^[ \t]*!\[[^\]\n]*\]\(.+\)[ \t]*\n?",
        re.MULTILINE,
    )
    cleaned = line_only.sub("", text)
    cleaned = IMG_REF_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# 怪字 bullet 符號集（黑方塊/圓圈/原點/菱形等，markdown_report A2）。
# 不含 markdown 標準 `-*+`（那是合法 list marker，僅在與 glyph 同行時順帶處理）。
BULLET_GLYPHS = "■□●○◎◯◆◇▪▫♦◊•‣⁃‧·∙・◦►▶▷▸▹❖✦◾◽∎"
# 行首 bullet marker：可選 markdown list（- * +）、可選 bold-open（**）、一個以上 glyph、
# 可選分隔符（. 、 ,）。group(1)=縮排、group(2)=被吃掉的 bold-open（用來決定是否去尾 **）。
_LEADING_BULLET_RE = re.compile(
    rf"^([ \t]*)(?:[-*+][ \t]+)?(\*\*)?[ \t]*[{BULLET_GLYPHS}]+[ \t]*[.、,]?[ \t]*"
)
# 「整行只有符號/星號/list marker/分隔符」的垃圾行偵測用字元集
_BULLET_NOISE_CHARS = re.compile(rf"[\s*+.、,\-{BULLET_GLYPHS}]")


def _strip_bullet_line(line: str) -> str | None:
    """正規化單行的怪字 bullet（A2）。回 None 表示整行是純符號垃圾、應丟棄。

    - 只剝「行首」符號（行內符號保留，依使用者選擇不拆黏行）
    - 連同包住整行的 bold（`**...**`）一起去掉
    - `- ****`、`•`、`****` 這類只有符號的空行 → 丟棄
    """
    if line.strip() and not _BULLET_NOISE_CHARS.sub("", line):
        return None
    m = _LEADING_BULLET_RE.match(line)
    if not m:
        return line
    rest = line[m.end():]
    if m.group(2):  # 行首吃掉了 bold-open → 去對應的 bold-close
        rest = re.sub(r"[ \t]*\*\*[ \t]*$", "", rest)
    return m.group(1) + rest


def _normalize_bullet_glyphs(text: str) -> str:
    """逐行剝除怪字 bullet 符號 + 丟棄純符號垃圾行，回正規化後文字（A2）。"""
    kept = [s for ln in text.split("\n") if (s := _strip_bullet_line(ln)) is not None]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


def derive_image_label_key(headings: dict[int, str], section_body: str) -> str:
    """
    決定該 section 圖片標籤的 key：
    1. 有 heading（h3+）→ 用最深一階的標題
    2. 完全無 heading，但 body 內僅有「唯一一條非標題且非空白」的文字 → 用該文字
    3. 其餘情況 → 'image'
    """
    if headings:
        deepest = max(headings.keys())
        return headings[deepest]
    candidates = [
        ln.strip() for ln in section_body.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return "image"


def _demote_orphan_leaf_headings(sections: list[dict]) -> list[dict]:
    """把「孤兒葉節點標題」降級為內容文字（markdown_report A1/B 修正）。

    chunking 以 heading 階層作 breadcrumb metadata。若某 section 的最深層 heading
    底下沒有任何非標題內文（只剩更深標題或圖片連結），它在切 chunk 時要嘛被當成
    純路徑節點、要嘛因 body 空整段被 skip，導致該標題攜帶的有意義文字
    （如「系統管材概要說明」）從未進向量庫、檢索不到。

    規則（逐 section 判定）：一個 section 的最深層 heading 若同時滿足
      (1) 底下無文字 body（圖片連結不算內文），且
      (2) 不被任何更深的 section 接續（即它是葉節點，而非結構容器）
    則把該最深標題從 headings 路徑移除、改寫成 body 一行文字。**降級層不進路徑**，
    更上層 heading 仍保留為 breadcrumb（只降最深一層，對齊「最小的子標題轉為內容」）。
    最後合併路徑相同的相鄰 section，讓同一父標題下多條降級文字併為一塊。
    """
    def is_extended(idx: int) -> bool:
        """sections[idx] 的最深 heading 是否被某個更深 section 沿用（= 有子節點）。"""
        path = sections[idx]["headings"]
        if not path:
            return False
        lmax = max(path)
        for j, other in enumerate(sections):
            if j == idx:
                continue
            opath = other["headings"]
            if max(opath, default=-1) <= lmax:
                continue
            if all(opath.get(lvl) == title for lvl, title in path.items()):
                return True
        return False

    demoted: list[dict] = []
    for i, sec in enumerate(sections):
        headings = dict(sec["headings"])
        body = sec["body"]
        if headings and not body and not is_extended(i):
            body = headings.pop(max(headings))
        demoted.append({
            "headings": headings,
            "body": body,
            "image_paths": list(sec["image_paths"]),
        })

    merged: list[dict] = []
    for sec in demoted:
        if merged and merged[-1]["headings"] == sec["headings"]:
            prev = merged[-1]
            prev["body"] = "\n".join(b for b in (prev["body"], sec["body"]) if b)
            prev["image_paths"].extend(sec["image_paths"])
        else:
            merged.append(sec)
    return merged


def split_page_by_headings(page_md: str) -> list[dict]:
    """
    把一頁切成 sections，支援 h3-h6 任意層級巢狀。
    回傳 list[{
        'headings': dict[int, str],  # {3: "...", 4: "...", ...} 從最淺到最深
        'body':     str,             # 已 strip 圖片引用 + trim
        'image_paths': list[str],    # 該 section 內所有 ![]() 路徑
    }]。

    巢狀規則：遇到 level=L 的標題時，清掉 current 中所有 level >= L 的舊條目，
    再寫入 current[L] = title。這代表新標題會取代同層或更深層的歷史脈絡。
    第一個標題之前的內容歸到 headings={} 的 section（若有 body 或圖片才出現）。
    """
    body = re.sub(r"^## 第 \d+ 頁\s*\n", "", page_md)
    body = re.sub(r"\n---\s*\n?$", "", body)
    body = _normalize_bullet_glyphs(body)  # A2：先清怪字 bullet，再做 heading 偵測/A1 降級

    def make_section(headings: dict[int, str], raw: str) -> dict:
        imgs = [r["md_path"] for r in parse_images_in_page(raw)]
        cleaned = strip_image_refs(raw).strip()
        return {"headings": dict(headings), "body": cleaned, "image_paths": imgs}

    matches = list(HEADING_RE.finditer(body))
    sections: list[dict] = []
    current: dict[int, str] = {}

    if not matches:
        sec = make_section(current, body)
        if sec["body"] or sec["image_paths"]:
            sections.append(sec)
        return sections

    pre_sec = make_section(current, body[: matches[0].start()])
    if pre_sec["body"] or pre_sec["image_paths"]:
        sections.append(pre_sec)

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        for lvl in [k for k in current.keys() if k >= level]:
            del current[lvl]
        current[level] = title

        nl = body.find("\n", m.end())
        sec_start = nl + 1 if nl != -1 else len(body)
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(make_section(current, body[sec_start:sec_end]))

    return _demote_orphan_leaf_headings(sections)


def decode_delim(s: str) -> str:
    """支援 \\n、\\t 等 escape sequence 輸入。"""
    if not s:
        return ""
    try:
        return s.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return s


def split_content(content: str, mode: str, delimiter: str) -> list[str]:
    """mode = 'page' (整頁一塊) | 'delimiter' (依分隔字元)。"""
    if not content.strip():
        return []
    if mode == "page":
        return [content.strip()]
    delim = decode_delim(delimiter)
    if not delim:
        return [content.strip()]
    return [seg.strip() for seg in content.split(delim) if seg.strip()]


def build_text_with_prefix(
    project_name: str, file_stem: str, page_num: int,
    headings: dict[int, str], seg_text: str,
) -> str:
    """產生注入 prefix 後的 chunk 文字（embedding 用）。
    格式：[project | file_stem | P{page} | h3 > h4 > ...] {seg_text}"""
    parts = [project_name, file_stem, f"P{page_num}"]
    if headings:
        heading_path = " > ".join(headings[lvl] for lvl in sorted(headings))
        parts.append(heading_path)
    return f"[{' | '.join(parts)}] {seg_text}"


def build_image_struct(md_path: str, file_key: str, image_root: Path) -> dict:
    """從 md 中 ![alt](path) 的 path 解析出 Qdrant payload 圖片結構。"""
    basename = Path(md_path.replace("\\", "/")).name
    abs_img = resolve_image_path(file_key, md_path, image_root)
    return {
        "file_name": basename,
        "local_path": str(abs_img),
        "md_ref": f"![{basename}]({md_path})",
        "alt_text": basename,
    }


def build_chunk_payload(
    *,
    file_key: str,
    sidecar: dict,
    page_num: int,
    section_idx: int,
    section: dict,
    seg_text: str,
    chunk_idx: int,
    chunk_idx_global: int,
    h1: str,
    doc_labels: list,
    page_labels: list,
    image_root: Path,
    prev_chunk_id: str | None,
    next_chunk_id: str | None,
    doc_type: str | None = None,
) -> dict:
    """產出 qdrant格式.md v2.1.0 完整 payload（含 point id）。
    single source of truth：UI 預覽、JSON 下載、Qdrant upsert 全部用同一個輸出。

    doc_type 由呼叫端（每份文件）算好一次傳入，避免在 chunk 迴圈反覆讀對照表；未傳則自行解析。"""
    parts = split_key_parts(file_key)
    project_name = derive_project_name(file_key)  # 建案 = file_key[1]（類別在 [0]）
    doc_type = doc_type if doc_type is not None else resolve_doc_type(file_key, sidecar)
    file_name = parts[-1]
    file_stem = Path(file_name).stem
    file_hash = sidecar.get("file_hash", "")
    headings = section.get("headings", {})
    image_paths = section.get("image_paths", [])

    point_id = make_point_id(file_hash, page_num, section_idx, chunk_idx)
    parent_section_id = make_section_id(file_hash, page_num, section_idx)
    text_with_prefix = build_text_with_prefix(
        project_name, file_stem, page_num, headings, seg_text
    )
    img_label = (
        derive_image_label_key(headings, section.get("body", ""))
        if image_paths else ""
    )
    images_struct = [build_image_struct(p, file_key, image_root) for p in image_paths]
    headings_sorted = sorted(headings.keys())
    label_keys = sorted(
        {lab["key"] for lab in doc_labels} | {lab["key"] for lab in page_labels}
    )
    breadcrumb = [h1, f"第 {page_num} 頁"] + [headings[lvl] for lvl in headings_sorted]
    current_header = headings[headings_sorted[-1]] if headings_sorted else f"第 {page_num} 頁"

    return {
        "id": point_id,
        "payload": {
            "metadata": {
                "source": {
                    "project_name": project_name,
                    "doc_type": doc_type,
                    "file_key": file_key,
                    "file_name": file_name,
                    "file_path": file_key,
                    "file_hash": file_hash,
                    "doc_title": h1,
                },
                "location": {
                    "page": page_num,
                    "page_label": f"第 {page_num} 頁",
                    "headings": {str(k): v for k, v in headings.items()},
                    "headings_flat": [headings[lvl] for lvl in headings_sorted],
                    "breadcrumb": breadcrumb,
                    "current_header": current_header,
                    "section_idx": section_idx,
                    "chunk_idx": chunk_idx,
                    "chunk_idx_global": chunk_idx_global,
                },
            },
            "content": {
                "text": seg_text,
                "text_with_prefix": text_with_prefix,
                "md_content": seg_text,
                "token_count": len(seg_text),
                "char_count": len(seg_text),
            },
            "visuals": {
                "has_image": bool(image_paths),
                "image_label": img_label,
                "image_count": len(image_paths),
                "images": images_struct,
            },
            "labels": {
                "document": list(doc_labels),
                "page": list(page_labels),
                "label_keys": label_keys,
            },
            "chunking": {
                "strategy": CHUNKING_STRATEGY,
                "parent_section_id": parent_section_id,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
            },
            "sys_info": {
                "ingestion_time": None,
                "pipeline_version": PIPELINE_VERSION,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_version": EMBEDDING_VERSION,
                "review_status": sidecar.get("review_status", "unreviewed"),
                "reviewer": sidecar.get("reviewer", ""),
                "reviewed_at": sidecar.get("reviewed_at"),
            },
        },
    }


def build_all_chunks_for_doc(
    *,
    file_key: str,
    sidecar: dict,
    pages: list[tuple[int, str]],
    page_md_override: dict[int, str] | None,
    h1: str,
    doc_labels: list,
    file_labels: dict,
    split_cfg: dict,
    image_root: Path,
) -> list[dict]:
    """走訪所有頁面 → sections → chunks，回傳全部 chunk payloads（含 prev/next 連結）。
    每筆額外帶 _page_pos（page index in pages list）供 UI 篩選用，
    寫入 Qdrant 前需 pop 掉這個底線開頭欄位。"""
    file_hash = sidecar.get("file_hash", "")
    doc_type = resolve_doc_type(file_key, sidecar)  # 每份文件算一次，傳給每個 chunk

    flat: list[dict] = []
    for page_pos, (page_num, page_md_default) in enumerate(pages):
        page_md_eff = (page_md_override or {}).get(page_pos, page_md_default)
        page_labels = file_labels["pages"].get(page_pos, [])
        sections = split_page_by_headings(page_md_eff)
        for section_idx, section in enumerate(sections):
            if not section.get("body"):
                continue
            seg_list = split_content(
                section["body"], split_cfg["mode"], split_cfg["delim"]
            )
            for chunk_idx, seg in enumerate(seg_list):
                flat.append({
                    "page_pos": page_pos,
                    "page_num": page_num,
                    "section_idx": section_idx,
                    "section": section,
                    "chunk_idx": chunk_idx,
                    "seg": seg,
                    "page_labels": page_labels,
                })

    point_ids = [
        make_point_id(file_hash, it["page_num"], it["section_idx"], it["chunk_idx"])
        for it in flat
    ]

    out: list[dict] = []
    for i, it in enumerate(flat):
        prev_id = point_ids[i - 1] if i > 0 else None
        next_id = point_ids[i + 1] if i + 1 < len(point_ids) else None
        pkg = build_chunk_payload(
            file_key=file_key,
            sidecar=sidecar,
            page_num=it["page_num"],
            section_idx=it["section_idx"],
            section=it["section"],
            seg_text=it["seg"],
            chunk_idx=it["chunk_idx"],
            chunk_idx_global=i,
            h1=h1,
            doc_labels=doc_labels,
            page_labels=it["page_labels"],
            image_root=image_root,
            prev_chunk_id=prev_id,
            next_chunk_id=next_id,
            doc_type=doc_type,
        )
        pkg["_page_pos"] = it["page_pos"]
        out.append(pkg)
    return out


def build_chunks_from_disk(
    file_key: str, image_root: Path, data_path: Path,
) -> tuple[list[dict], dict]:
    """純從磁碟（{stem}.md + sidecar）重建某檔全部 chunks，**不依賴 session_state**
    的編輯緩衝，供 fast pipeline 批次處理未開啟的檔使用。

    與互動流程一致地 lazy 計算缺失的 file_hash（從原始 PDF/PPT），並把標籤／切分設定
    從 sidecar 還原（sidecar 用 str key 存 pages，這裡轉回 int key）。
    回 (all_chunks, sidecar)；sidecar 為這次用到的同一份 dict（可接續寫狀態）。
    """
    md_path = derive_md_path(file_key)
    md_text = md_path.read_text(encoding="utf-8")
    _, _, pages = parse_md(md_text)
    sidecar = load_sidecar(file_key)
    if not sidecar.get("file_hash"):
        sidecar["file_hash"] = compute_file_hash(data_path / file_key)
        save_sidecar(file_key, sidecar)
    stem = Path(split_key_parts(file_key)[-1]).stem
    h1 = extract_h1(md_text, stem)
    cl = sidecar.get("custom_labels", {"document": [], "pages": {}})
    file_labels = {
        "document": list(cl.get("document", [])),
        "pages": {int(k): list(v) for k, v in cl.get("pages", {}).items()},
    }
    split_cfg = dict(sidecar.get("split_settings", {"mode": "delimiter", "delim": "\\n"}))
    chunks = build_all_chunks_for_doc(
        file_key=file_key,
        sidecar=sidecar,
        pages=pages,
        page_md_override=None,
        h1=h1,
        doc_labels=file_labels["document"],
        file_labels=file_labels,
        split_cfg=split_cfg,
        image_root=image_root,
    )
    return chunks, sidecar


# === Embedding 快取（mkdata/{stem}.vectors.* 三件套）===
def derive_vector_paths(file_key: str) -> dict:
    file_name = split_key_parts(file_key)[-1]
    stem = Path(file_name).stem
    return {
        "dense":    MKDATA_PATH / f"{stem}.vectors.dense.npy",
        "sparse":   MKDATA_PATH / f"{stem}.vectors.sparse.json",
        "manifest": MKDATA_PATH / f"{stem}.vectors.manifest.json",
    }


def load_vector_manifest(file_key: str) -> dict | None:
    p = derive_vector_paths(file_key)["manifest"]
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def vector_cache_status(
    file_key: str, sidecar: dict, expected_chunk_ids: list[str]
) -> dict:
    """判斷快取狀態。回 {exists, valid, reason, manifest?}."""
    manifest = load_vector_manifest(file_key)
    paths = derive_vector_paths(file_key)
    if (
        manifest is None
        or not paths["dense"].exists()
        or not paths["sparse"].exists()
    ):
        return {"exists": False, "valid": False, "reason": "無快取"}

    reasons = []
    if manifest.get("file_hash") != sidecar.get("file_hash"):
        reasons.append("file_hash 變動")
    if manifest.get("chunking_strategy") != CHUNKING_STRATEGY:
        reasons.append("chunking_strategy 變動")
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        reasons.append("embedding_model 變動")
    if manifest.get("embedding_version") != EMBEDDING_VERSION:
        reasons.append("embedding_version 變動")
    if manifest.get("chunk_ids") != expected_chunk_ids:
        reasons.append("chunks 結構變動")

    if reasons:
        return {
            "exists": True,
            "valid": False,
            "reason": "、".join(reasons),
            "manifest": manifest,
        }
    return {"exists": True, "valid": True, "reason": "", "manifest": manifest}


def save_vectors(
    file_key: str, sidecar: dict,
    chunk_ids: list[str], dense: np.ndarray, sparse: list[dict],
    fp16: bool,
) -> None:
    """原子寫入：dense.npy + sparse.json + manifest.json。"""
    paths = derive_vector_paths(file_key)
    paths["dense"].parent.mkdir(parents=True, exist_ok=True)

    # dense 存 fp16 省一半磁碟（檢索時轉 fp32）；非 fp16 流程仍存 fp32
    arr = dense.astype(np.float16 if fp16 else np.float32)
    tmp_dense = paths["dense"].with_suffix(".npy.tmp")
    # 用 file object 餵 np.save，避免 numpy 對非 .npy 結尾的路徑自動補 .npy
    # （否則會寫到 xxx.npy.tmp.npy，後續 replace 找不到 tmp 檔）
    with open(tmp_dense, "wb") as f:
        np.save(f, arr)
    tmp_dense.replace(paths["dense"])

    tmp_sparse = paths["sparse"].with_suffix(".json.tmp")
    tmp_sparse.write_text(json.dumps(sparse, ensure_ascii=False), encoding="utf-8")
    tmp_sparse.replace(paths["sparse"])

    manifest = {
        "file_key": file_key,
        "file_hash": sidecar.get("file_hash", ""),
        "chunking_strategy": CHUNKING_STRATEGY,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_version": EMBEDDING_VERSION,
        "dense_dim": int(dense.shape[1]),
        "dense_dtype": str(arr.dtype),
        "count": len(chunk_ids),
        "chunk_ids": list(chunk_ids),
        "encoded_at": now_iso(),
        "fp16": fp16,
    }
    tmp_manifest = paths["manifest"].with_suffix(".json.tmp")
    tmp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_manifest.replace(paths["manifest"])


def clear_vector_cache(file_key: str) -> int:
    """刪除三件套；回傳實際刪掉的檔數。"""
    n = 0
    for p in derive_vector_paths(file_key).values():
        if p.exists():
            p.unlink()
            n += 1
    return n


@st.cache_resource(show_spinner="載入 bge-m3 中（CPU 約 2-3s）...")
def load_embedder(model_name: str, use_fp16: bool, device: str = "cpu"):
    """快取整個 BGEM3FlagModel 物件（cache_resource = 行程內單例），避免每次 rerun 重載。

    device="cpu"（預設）：互動端 RTX 2050 等小顯存卡的唯一安全選擇。CPU 載入 bge-m3
    約 2-3s、單句 encode <1s，互動查詢綽綽有餘。
    device="cuda"：需 ≳6GB 空閒顯存。4GB 卡（2050）在 Streamlit 行程內載入 bge-m3 會
    CUDA OOM 讓**整個行程直接 abort**（終端無 traceback 死掉），故非預設、勿在小卡上開。

    另：必須主執行緒（ScriptRunner）載入並使用 —— GPU 模型跨執行緒建立/推理會崩潰。"""
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as e:
        raise RuntimeError(
            "FlagEmbedding 未安裝。請執行：\n"
            "    pip install -U FlagEmbedding\n"
            "（依賴 peft / accelerate，第一次裝約 100MB）"
        ) from e
    if device == "cpu":
        # fp16 在 CPU 無意義且部分算子不支援 → 強制 fp32
        return BGEM3FlagModel(model_name, use_fp16=False, devices="cpu")
    return BGEM3FlagModel(model_name, use_fp16=use_fp16)


def encode_chunks(
    model, texts: list[str], batch_size: int = 32,
) -> tuple[np.ndarray, list[dict]]:
    """bge-m3 一次推理同時拿 dense + sparse。
    回 (dense: ndarray (N, 1024), sparse: list[dict[token_id_str → weight]])。"""
    out = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    sparse = [
        {str(k): float(v) for k, v in row.items()}
        for row in out["lexical_weights"]
    ]
    return dense, sparse


# === Qdrant 寫入 helpers ===
@st.cache_resource(show_spinner="連線 Qdrant...")
def get_qdrant_client(url: str, api_key: str):
    """Cached client；URL / API key 改變會自動建新 cache entry。"""
    if not HAS_QDRANT:
        raise RuntimeError("qdrant-client 未安裝：pip install qdrant-client")
    return QdrantClient(
        url=url,
        api_key=api_key or None,
        timeout=30,
    )


@st.cache_resource(show_spinner=False)
def get_llm_client(base_url: str):
    """Cached OpenAI 相容 client，指向 vLLM 的 /v1。base_url 改變自動建新 entry。

    有界逾時：串流模式下 read timeout 是「相鄰兩 token 之間」的上限，不是整段生成上限，
    300s 對慢的首 token 也夠寬；connect 10s 避免遠端不通時整個 rerun 卡死。
    """
    import httpx
    return OpenAI(
        base_url=base_url,
        api_key=LLM_API_KEY,
        timeout=httpx.Timeout(300.0, connect=10.0),
        max_retries=1,
    )


@st.cache_data(show_spinner=False, ttl=60)
def list_llm_models(base_url: str) -> list[str]:
    """動態抓 vLLM 服務的模型清單（GET /v1/models）。

    給對話模式 model 下拉用。連線／查詢失敗回空 list，讓 UI 退回 .env 內建清單、不報錯。
    cache 60s（base_url 變動時 cache key 變 → 自動重抓）。
    """
    if not HAS_LLM:
        return []
    try:
        resp = get_llm_client(base_url).models.list()
        return sorted(m.id for m in resp.data if getattr(m, "id", None))
    except Exception:  # noqa: BLE001 — 取不到清單不可擋住對話 UI
        return []


def _llm_stream_tokens(stream):
    """從 OpenAI 相容串流（vLLM）逐塊取出 delta.content，餵給 st.write_stream。

    串流是底層斷線的治本解：生成期間 websocket 持續有資料流動，連線不會被判定為 idle
    而遭部屬層（proxy / 瀏覽器）剪斷，UI 也即時更新不再全凍。
    """
    for chunk in stream:
        try:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            tok = getattr(choices[0].delta, "content", None)
            if tok:
                yield tok
        except Exception:  # noqa: BLE001 — 單塊解析失敗不該中斷整串流
            continue


PAYLOAD_INDEX_FIELDS = [
    # (field path, schema type) — 對齊 qdrant格式.md §2.2
    ("metadata.source.project_name", "keyword"),
    ("metadata.source.doc_type",     "keyword"),
    ("metadata.source.file_hash",    "keyword"),
    ("metadata.source.file_key",     "keyword"),
    ("metadata.location.page",       "integer"),
    ("metadata.location.headings_flat", "keyword"),
    ("chunking.strategy",            "keyword"),
    ("visuals.has_image",            "bool"),
    ("visuals.image_label",          "keyword"),
    ("labels.label_keys",            "keyword"),
    ("sys_info.review_status",       "keyword"),
    ("sys_info.embedding_model",     "keyword"),
]


def ensure_payload_indexes(client, name: str) -> list[str]:
    """對既有 collection 補建所有 PAYLOAD_INDEX_FIELDS 索引（容錯、idempotent）。

    create_payload_index 對「已存在的索引」會丟例外，這裡逐一吞掉 → 既有欄位不受影響，
    新增欄位（如 doc_type）會在第一次呼叫時補建。回成功/已存在的 field 清單。"""
    schema_map = {
        "keyword": qm.PayloadSchemaType.KEYWORD,
        "integer": qm.PayloadSchemaType.INTEGER,
        "bool":    qm.PayloadSchemaType.BOOL,
    }
    done = []
    for field, kind in PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema_map[kind],
            )
            done.append(field)
        except Exception:
            pass  # 多半是「索引已存在」，忽略
    return done


def ensure_text_collection(client, name: str, recreate: bool = False) -> dict:
    """確保 collection 存在；recreate=True 會先刪除再建。
    回 {created: bool, recreated: bool, indexed: list[str]}。"""
    info = {"created": False, "recreated": False, "indexed": []}
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
        info["recreated"] = True
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "text_dense": qm.VectorParams(
                    size=EMBEDDING_DIM_DENSE,
                    distance=qm.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "text_sparse": qm.SparseVectorParams(),
            },
        )
        info["created"] = True
    # 不論新建或既有，都確保索引齊全（既有 collection 也能補上新增的 doc_type 索引）
    info["indexed"] = ensure_payload_indexes(client, name)
    return info


def count_existing_for_file(client, name: str, file_hash: str) -> int:
    """查 collection 內 file_hash 對應的 points 數。"""
    if not file_hash:
        return 0
    if not client.collection_exists(name):
        return 0
    try:
        res = client.count(
            collection_name=name,
            count_filter=qm.Filter(must=[
                qm.FieldCondition(
                    key="metadata.source.file_hash",
                    match=qm.MatchValue(value=file_hash),
                ),
            ]),
            exact=True,
        )
        return int(res.count)
    except Exception:
        return 0


def get_collection_total(client, name: str) -> int:
    try:
        if not client.collection_exists(name):
            return 0
        return int(client.count(collection_name=name, exact=True).count)
    except Exception:
        return 0


@st.cache_data(show_spinner=False, ttl=300)
def list_facet_values(
    url: str, api_key: str, coll: str, field: str,
    filter_field: str | None = None,
    filter_values: tuple[str, ...] | None = None,
) -> list[str]:
    """用 Qdrant facet API 列出某 keyword 欄位的 distinct 值（給篩選下拉用）。

    filter_field/filter_values（皆 hashable，供 cache key）：對 facet 加條件——只統計
    `filter_field IN filter_values` 的 points。用來做「連動下拉」（例如只列某文件種類底下
    實際存在的專案），避免使用者選出不存在的組合。
    連線／查詢失敗回空 list，讓 UI 退回「不篩」、不報錯。cache 5 分鐘避免每 rerun 打 Qdrant。
    """
    try:
        cli = get_qdrant_client(url, api_key)
        if not cli.collection_exists(coll):
            return []
        ffilter = None
        if filter_field and filter_values:
            ffilter = qm.Filter(must=[qm.FieldCondition(
                key=filter_field, match=qm.MatchAny(any=list(filter_values)),
            )])
        resp = cli.facet(collection_name=coll, key=field, facet_filter=ffilter, limit=1000)
        return sorted(
            h.value for h in resp.hits
            if isinstance(getattr(h, "value", None), str) and h.value
        )
    except Exception:  # noqa: BLE001 — 取不到清單不可擋住 UI
        return []


def list_projects_in_collection(
    url: str, api_key: str, coll: str, doc_types: list[str] | None = None,
) -> list[str]:
    """distinct 建案名（metadata.source.project_name）。

    傳 doc_types → 只列「這些文件種類底下實際存在」的建案（連動下拉，防止選出空組合）。
    對話模式取不到審閱模式的 file_keys，改從實際 ingest 進去的資料反查更準。"""
    return list_facet_values(
        url, api_key, coll, "metadata.source.project_name",
        filter_field="metadata.source.doc_type" if doc_types else None,
        filter_values=tuple(sorted(doc_types)) if doc_types else None,
    )


def list_doc_types_in_collection(url: str, api_key: str, coll: str) -> list[str]:
    """distinct 文件種類（metadata.source.doc_type）。"""
    return list_facet_values(url, api_key, coll, "metadata.source.doc_type")


def load_cached_vectors(file_key: str) -> tuple[np.ndarray, list[dict], dict]:
    """讀 mkdata/{stem}.vectors.* 三件套；dense 自動還原為 fp32。"""
    paths = derive_vector_paths(file_key)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    dense = np.load(paths["dense"])
    if dense.dtype != np.float32:
        dense = dense.astype(np.float32)
    sparse = json.loads(paths["sparse"].read_text(encoding="utf-8"))
    return dense, sparse, manifest


def build_points_for_upsert(
    chunk_payloads: list[dict],
    dense: np.ndarray,
    sparse: list[dict],
    ingestion_ts: str,
) -> list:
    """組裝 PointStruct list；對齊 cache 順序，注入 ingestion_time。"""
    points = []
    for i, pkg in enumerate(chunk_payloads):
        payload = pkg["payload"]
        payload["sys_info"]["ingestion_time"] = ingestion_ts
        sparse_dict = sparse[i] if i < len(sparse) else {}
        if sparse_dict:
            indices = [int(k) for k in sparse_dict.keys()]
            values = [float(v) for v in sparse_dict.values()]
        else:
            indices, values = [], []
        points.append(qm.PointStruct(
            id=pkg["id"],
            vector={
                "text_dense": dense[i].tolist(),
                "text_sparse": qm.SparseVector(indices=indices, values=values),
            },
            payload=payload,
        ))
    return points


def upsert_in_batches(
    client, name: str, points: list, batch_size: int = QDRANT_BATCH_SIZE,
    progress_cb=None,
) -> None:
    """分批 upsert；wait=True 確保每批寫完才回來，失敗會 raise。"""
    total = len(points)
    for start in range(0, total, batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=name, points=batch, wait=True)
        if progress_cb:
            progress_cb(min(start + len(batch), total), total)


# === Qdrant 檢索 helpers（P5 檢索測試 Tab 用） ===

def encode_query(model, query_text: str) -> tuple[np.ndarray, dict]:
    """bge-m3 對單條 query 拿 dense + sparse。
    不套 build_text_with_prefix —— bge-m3 是 symmetric encoder，
    query 跟 passage 走同一個 encoder、不需要不同前綴。"""
    out = model.encode(
        [query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)[0]
    sparse_raw = out["lexical_weights"][0]
    sparse = {str(k): float(v) for k, v in sparse_raw.items()}
    return dense, sparse


def _sparse_to_qdrant(sparse: dict) -> "qm.SparseVector":
    if not sparse:
        return qm.SparseVector(indices=[], values=[])
    indices = [int(k) for k in sparse.keys()]
    values = [float(v) for v in sparse.values()]
    return qm.SparseVector(indices=indices, values=values)


def search_dense(
    client, name: str, dense: np.ndarray, top_k: int,
    qfilter: "qm.Filter | None" = None,
) -> list:
    res = client.query_points(
        collection_name=name,
        query=dense.tolist(),
        using="text_dense",
        limit=top_k,
        with_payload=True,
        query_filter=qfilter,
    )
    return res.points


def search_sparse(
    client, name: str, sparse: dict, top_k: int,
    qfilter: "qm.Filter | None" = None,
) -> list:
    res = client.query_points(
        collection_name=name,
        query=_sparse_to_qdrant(sparse),
        using="text_sparse",
        limit=top_k,
        with_payload=True,
        query_filter=qfilter,
    )
    return res.points


def search_hybrid(
    client, name: str, dense: np.ndarray, sparse: dict, top_k: int,
    qfilter: "qm.Filter | None" = None,
    prefetch_k: int = 40,
) -> list:
    """RRF 融合：dense + sparse 各撈 prefetch_k，伺服器端做 Reciprocal Rank Fusion。"""
    res = client.query_points(
        collection_name=name,
        prefetch=[
            qm.Prefetch(
                query=dense.tolist(),
                using="text_dense",
                limit=prefetch_k,
                filter=qfilter,
            ),
            qm.Prefetch(
                query=_sparse_to_qdrant(sparse),
                using="text_sparse",
                limit=prefetch_k,
                filter=qfilter,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        query_filter=qfilter,
    )
    return res.points


# === 三路意圖 router 檢索 helpers（第 1 層；見 rag_comparison_routing_design） ===

def encode_queries(model, texts: list[str]) -> list[tuple[np.ndarray, dict]]:
    """一次 batch encode 多條 query（bge-m3 收 list）→ [(dense, sparse), ...]，順序對齊 texts。

    fan-out 的每建案改寫共用一次 encode，省去逐條呼叫開銷。bge-m3 symmetric，
    query/passage 同 encoder，不套前綴（同 encode_query）。"""
    if not texts:
        return []
    out = model.encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    dense_all = np.asarray(out["dense_vecs"], dtype=np.float32)
    results: list[tuple[np.ndarray, dict]] = []
    for i in range(len(texts)):
        sparse = {str(k): float(v) for k, v in out["lexical_weights"][i].items()}
        results.append((dense_all[i], sparse))
    return results


def resolve_entities(
    mentions: list[str], known_projects: list[str],
) -> tuple[list[tuple[int, str]], list[str]]:
    """把 router 的 entity_mentions 對映到 collection 內實際存在的 project_name。

    回 (pairs, unresolved)：
      - pairs = [(mention_index, project_name), ...]，project_name 保證是實際 payload key
        （metadata.source.project_name＝file_key[1]，見 doctype_project_path_model），
        且 mention_index 用來把 plan.rewrites[i] 對齊回正確的建案（避免中途對不到時錯序）。
      - unresolved = 對不到的原始 mention 字串（UI 需明講未納入，不靜默）。
    比對用正規化（去空白 + 小寫）子字串雙向包含；一個 project 只配一次（先到先得）。"""
    def _norm(s: str) -> str:
        return "".join(str(s).split()).lower()

    norm_known = [(p, _norm(p)) for p in known_projects]
    pairs: list[tuple[int, str]] = []
    unresolved: list[str] = []
    used: set[str] = set()
    for idx, m in enumerate(mentions):
        nm = _norm(m)
        hit = None
        for proj, npj in norm_known:
            if nm and (nm in npj or npj in nm) and proj not in used:
                hit = proj
                break
        if hit:
            pairs.append((idx, hit))
            used.add(hit)
        else:
            unresolved.append(m)
    return pairs, unresolved


def fanout_retrieve(
    embedder, cli, coll: str, search_mode: str,
    entities: list[str], rewrites: list[str],
    base_must: list, per_entity_k: int,
) -> list[dict]:
    """compare/rank 用：每個建案綁一段改寫、各做一次「加該建案 filter」的檢索，chunk 標
    `_entity` 後合併。rewrites 與 entities 順序 1:1 對齊（空字串 → 退用建案名當 query）。
    base_must＝sidebar 硬約束（專案/類別），與每建案的 project_name filter 以 AND 疊加。"""
    if not entities:
        return []
    texts = [
        (rewrites[i] if i < len(rewrites) and rewrites[i] else entities[i])
        for i in range(len(entities))
    ]
    encoded = encode_queries(embedder, texts)
    all_chunks: list[dict] = []
    for ent, (q_dense, q_sparse) in zip(entities, encoded):
        must = list(base_must) + [qm.FieldCondition(
            key="metadata.source.project_name",
            match=qm.MatchAny(any=[ent]),
        )]
        ent_filter = qm.Filter(must=must)
        if search_mode == "dense":
            hits = search_dense(cli, coll, q_dense, per_entity_k, ent_filter)
        elif search_mode == "sparse":
            hits = search_sparse(cli, coll, q_sparse, per_entity_k, ent_filter)
        else:
            hits = search_hybrid(cli, coll, q_dense, q_sparse, per_entity_k, ent_filter)
        for h in hits:
            all_chunks.append({"score": h.score, "payload": h.payload or {}, "_entity": ent})
    return all_chunks


@st.cache_data(show_spinner=False)
def _load_structured_cached(path: str, mtime: float):
    """讀屬性表；**mtime 當快取 key** → 你在 Excel 改完存檔（mtime 跳）就自動熱載入，
    app 不重啟。回 structured_table.TableData。"""
    return _structured_table.load_table(path)


def _get_structured_table():
    """取屬性表（熱載入 + last-good fallback）。回 (TableData|None, path, notices)。

    讀檔失敗（例如你正在存檔的半途）→ 沿用 session 內上次成功載入的版本 + 掛 notice，
    不讓 chat 因表壞掉而整個掛。"""
    notices: list[str] = []
    if not HAS_STRUCTURED:
        return None, "", notices
    path = st.session_state.get("_structured_path", STRUCTURED_TABLE_PATH)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    td = _load_structured_cached(path, mtime)
    if td is not None and getattr(td, "ok", False):
        st.session_state["_structured_last_good"] = td
        return td, path, notices
    # 只有「真正讀檔/格式失敗」（有 error）才用 last-good；純粹「尚未填」（空表）不觸發
    if td is not None and getattr(td, "error", None):
        last = st.session_state.get("_structured_last_good")
        if last is not None:
            notices.append(f"⚠️ 屬性表讀取異常（{td.error}），沿用上次成功載入的版本。")
            return last, path, notices
    return td, path, notices  # 空表或無 last-good：回 td，上游自行判斷


def structured_lookup(plan, restrict_projects):
    """第 2 層屬性表查詢。回 {markdown, notices, covered} 或 None。

    None 代表「表未收錄此查詢相關屬性」→ 上游明講、**禁止 fall through 回向量猜數字**
    （見 rag_comparison_routing_design 定案）。restrict_projects＝sidebar 硬約束。"""
    if not HAS_STRUCTURED or plan is None:
        return None
    td, _path, load_notices = _get_structured_table()
    if td is None or not getattr(td, "ok", False):
        return None
    res = _structured_table.lookup(
        td,
        entities=plan.entity_mentions,
        attributes=plan.attributes,
        restrict_projects=restrict_projects or None,
    )
    if res is None:
        return None
    notices = list(load_notices) + list(td.warnings) + list(res.notices)
    return {"markdown": res.markdown, "notices": notices, "covered": res.covered_projects}


def route_and_retrieve(
    *, embedder, cli, coll: str, search_mode: str, query: str, top_k: int,
    plan, base_must: list, sidebar_projects: list[str], sidebar_doc_types: list[str],
    qdrant_url: str, qdrant_key: str,
    llm_client=None, model: str = "", do_hyde: bool = False,
    per_entity_k: int = LLM_COMPARE_TOP_K_PER_ENTITY,
    max_entities: int = LLM_MAX_FANOUT_ENTITIES,
) -> dict:
    """依 plan.intent 三路檢索。回 {chunks, intent, grouped, notices}。

    - plan=None（router 關閉／失敗／LLM 關閉）→ 一律單發 fact，等同現行行為。
    - compare → 每建案 fan-out（需 ≥2 個對得上的建案，否則退 fact）。
    - structured → 查第 2 層屬性表（權威數值來源），回 structured_md；另撈參考片段當質性補充。
      表未收錄 → structured_md=None + 明講 notice；STRUCTURED prompt 禁止 LLM 從片段取數字排序。
    sidebar_projects 是硬約束：router 不得覆蓋（有指定就取交集 / 或直接用指定清單）。"""
    notices: list[str] = []
    intent = plan.intent if plan else "fact"

    def _single_shot(k: int) -> list[dict]:
        text = query
        if do_hyde and llm_client is not None and llm_generate_hyde is not None:
            try:
                text = llm_generate_hyde(llm_client, model, query) or query
            except Exception as e:  # noqa: BLE001 — HyDE 失敗不擋檢索
                notices.append(f"HyDE 失敗，改用原 query：{e}")
                text = query
        d, s = encode_query(embedder, text)
        flt = qm.Filter(must=base_must) if base_must else None
        if search_mode == "dense":
            hits = search_dense(cli, coll, d, k, flt)
        elif search_mode == "sparse":
            hits = search_sparse(cli, coll, s, k, flt)
        else:
            hits = search_hybrid(cli, coll, d, s, k, flt)
        return [{"score": h.score, "payload": h.payload or {}} for h in hits]

    def _resolve() -> tuple[list[str], list[str], list[str]]:
        """回 (entities, rewrites_aligned, unresolved)。含 sidebar 硬約束 + 上限截斷。"""
        known = list_projects_in_collection(
            qdrant_url, qdrant_key, coll, doc_types=sidebar_doc_types or None,
        )
        pairs, unresolved = resolve_entities(plan.entity_mentions if plan else [], known)
        if sidebar_projects:  # 硬約束：router 對到的取交集；都對不到就直接用 sidebar 清單
            inter = [(i, p) for (i, p) in pairs if p in sidebar_projects]
            if inter:
                pairs = inter
            else:
                pairs = [(-1, p) for p in sidebar_projects]
                unresolved = []
        if len(pairs) > max_entities:
            notices.append(
                f"比較／排序實體超過上限 {max_entities}，只取前 {max_entities} 個（原 {len(pairs)} 個）。"
            )
            pairs = pairs[:max_entities]
        ents = [p for _, p in pairs]
        rws = [
            (plan.rewrites[i] if (plan and 0 <= i < len(plan.rewrites)) else "")
            for i, _ in pairs
        ]
        return ents, rws, unresolved

    if intent == "compare":
        ents, rws, unresolved = _resolve()
        if len(ents) >= 2:
            chunks = fanout_retrieve(
                embedder, cli, coll, search_mode, ents, rws, base_must, per_entity_k,
            )
            if unresolved:
                notices.append("下列指涉未對應到資料庫建案、未納入比較：" + "、".join(unresolved))
            return {"chunks": chunks, "intent": "compare", "grouped": True, "notices": notices}
        # 對不到 ≥2 建案 → 退回單發 fact，明講原因
        if plan and plan.entity_mentions:
            notices.append("未能在資料庫對應到 2 個以上的具名建案，已退回一般檢索。")
        return {"chunks": _single_shot(top_k), "intent": "fact", "grouped": False, "notices": notices}

    if intent == "structured":
        # 第 2 層：屬性表撈子表（權威數值來源）。撈到 → 掛表；撈不到 → 明講未收錄、不猜。
        sres = structured_lookup(plan, sidebar_projects)
        structured_md = sres["markdown"] if sres else None
        if sres:
            notices.extend(sres["notices"])
        else:
            notices.append(
                "ℹ️ 結構化屬性表未收錄此查詢相關屬性；以下為向量檢索參考片段，"
                "系統不會據此自行排序或編造數值。"
            )
        # 無論表有無命中，都撈參考片段當「質性補充」（STRUCTURED prompt 禁止從中取數字排序）
        ents, rws, _unresolved = _resolve()
        if len(ents) >= 2:
            chunks = fanout_retrieve(
                embedder, cli, coll, search_mode, ents, rws, base_must, per_entity_k,
            )
            grouped = True
        else:
            chunks = _single_shot(top_k)
            grouped = False
        return {"chunks": chunks, "intent": "structured", "grouped": grouped,
                "notices": notices, "structured_md": structured_md}

    # fact（含所有 fallback 路徑）
    return {"chunks": _single_shot(top_k), "intent": "fact", "grouped": False, "notices": notices}


# === Streamlit Callback ===
def editor_key_for(file_key: str, page_idx: int, version: int) -> str:
    return f"editor_{file_key}_{page_idx}_v{version}"


def bump_widget_version():
    st.session_state.widget_version = st.session_state.get("widget_version", 0) + 1


def commit_editor_if_dirty(file_key: str, page_idx: int) -> None:
    """讀 ace/text_area 在 session_state 中的最新值，若與 current_md 對應頁不同就 commit。
    取代舊版 text_area 的 on_change=commit_edit 流程，
    讓 ace（無 on_change callback）跟 text_area 走同一條 commit 路徑。

    編輯器只顯示去頁標頭（`## 第 N 頁`）與結尾分隔線（`---`）後的 body，
    這裡再對稱地把這兩個結構錨點黏回去，確保頁錨點/分隔線不被使用者誤改。
    比對在 stripped body 層級進行，避免空白差異造成每次 rerun 都誤判 dirty。"""
    if not file_key:
        return
    version = st.session_state.get("widget_version", 0)
    key = editor_key_for(file_key, page_idx, version)
    edited = st.session_state.get(key)
    if not isinstance(edited, str):
        return
    _, _, pages_curr = parse_md(st.session_state.current_md)
    if page_idx >= len(pages_curr):
        return
    page_num, current_page_md = pages_curr[page_idx]
    if edited.strip() != strip_page_header(current_page_md):
        has_separator = re.search(r"\n-{3,}\s*\n?$", current_page_md) is not None
        full_edited = f"## 第 {page_num} 頁\n{edited.strip()}\n"
        if has_separator:
            full_edited += "\n---\n"
        st.session_state.current_md = replace_page(
            st.session_state.current_md, page_idx, full_edited
        )
        mark_processing(file_key)


# === 圖片刪除／復原邏輯 ===
def delete_image_action(file_key: str, page_idx: int, ref: dict, image_root: Path):
    fm, header, pages = parse_md(st.session_state.current_md)
    page_num, page_md = pages[page_idx]

    new_page_md, removed_line, line_offset = remove_image_line(page_md, ref)

    abs_img = resolve_image_path(file_key, ref["md_path"], image_root)
    trashed_to = None
    used_elsewhere = is_ref_used_elsewhere(st.session_state.current_md, page_idx, ref["md_path"])

    if abs_img.exists() and not used_elsewhere:
        try:
            trashed_to = move_to_trash(abs_img, file_key)
        except Exception as e:
            st.warning(f"圖檔移至垃圾桶失敗：{e}（仍會移除 md 引用）")

    st.session_state.current_md = replace_page(
        st.session_state.current_md, page_idx, new_page_md
    )
    history = st.session_state.delete_history.setdefault(file_key, [])
    history.append({
        "page_idx": page_idx,
        "page_num": page_num,
        "md_path": ref["md_path"],
        "removed_line": removed_line,
        "line_offset": line_offset,
        "abs_path": str(abs_img),
        "trash_path": str(trashed_to) if trashed_to else None,
        "used_elsewhere": used_elsewhere,
        "deleted_at": now_iso(),
    })
    mark_processing(file_key)
    bump_widget_version()


def restore_image_action(file_key: str, history_idx: int):
    history = st.session_state.delete_history.get(file_key, [])
    if history_idx < 0 or history_idx >= len(history):
        return
    entry = history[history_idx]

    fm, header, pages = parse_md(st.session_state.current_md)
    page_idx = entry["page_idx"]
    if page_idx >= len(pages):
        st.warning("頁面結構改變，無法復原。")
        return
    page_num, page_md = pages[page_idx]
    new_page_md = insert_at(page_md, entry["removed_line"], entry["line_offset"])
    st.session_state.current_md = replace_page(
        st.session_state.current_md, page_idx, new_page_md
    )

    if entry["trash_path"]:
        trash_path = Path(entry["trash_path"])
        original = Path(entry["abs_path"])
        if trash_path.exists():
            try:
                restore_from_trash(trash_path, original)
            except Exception as e:
                st.warning(f"圖檔還原失敗：{e}（md 引用已復原）")

    history.pop(history_idx)
    mark_processing(file_key)
    bump_widget_version()


def restore_all_images_action(file_key: str) -> int:
    """一鍵復原本檔『近期刪除』的所有圖片（md 引用 + 垃圾桶檔案）。

    以 LIFO（後刪先復原）逐筆呼叫 restore_image_action，確保同頁多刪時
    line_offset 對位正確。回傳實際復原筆數；若某筆因頁面結構改變無法復原
    （restore_image_action 提早 return 而未 pop），偵測到數量未減少即跳出，
    避免死迴圈。
    """
    restored = 0
    while st.session_state.delete_history.get(file_key):
        before = len(st.session_state.delete_history[file_key])
        restore_image_action(file_key, before - 1)
        after = len(st.session_state.delete_history.get(file_key, []))
        if after >= before:
            break
        restored += 1
    return restored


def restore_trashed_files_for(file_key: str) -> None:
    """把本檔已軟刪除的圖檔從垃圾桶搬回原位（僅動檔案，不碰 md）。

    供「捨棄編輯」使用：捨棄會從磁碟 reload md（引用本就還在），但實體檔仍在
    垃圾桶，若不搬回會留下指向不存在檔案的孤兒引用。
    """
    for entry in st.session_state.delete_history.get(file_key, []):
        tp = entry.get("trash_path")
        if tp and Path(tp).exists():
            try:
                restore_from_trash(Path(tp), Path(entry["abs_path"]))
            except Exception as e:
                st.warning(f"圖檔還原失敗：{e}")


# === UI ===
st.set_page_config(page_title="RAG MD 比對工具", layout="wide")

# 全域 session_state 初始化
if "delete_history" not in st.session_state:
    st.session_state.delete_history = {}
if "widget_version" not in st.session_state:
    st.session_state.widget_version = 0
if "custom_labels" not in st.session_state:
    # custom_labels[file_key] = {"document": [...], "pages": {idx: [...]}}
    st.session_state.custom_labels = {}
if "split_settings" not in st.session_state:
    # split_settings[file_key] = {"mode": "page"|"delimiter", "delim": "\\n"}
    st.session_state.split_settings = {}
if "sidecars" not in st.session_state:
    # sidecars[file_key] = sidecar dict（mkdata/{stem}.review.json 鏡像）
    st.session_state.sidecars = {}

# === View state（審閱模式 ↔ 對話模式）+ chat sessions 容器 ===
def _get_local_storage():
    """取得 LocalStorage 元件實例（無元件回 None）。

    元件 __init__ 首跑會對瀏覽器做一次 getAll round-trip 並快取進 st.session_state，
    之後同一 session 重建走快取分支、不再連線瀏覽器 → 每個 rerun 呼叫都很便宜。
    """
    if not HAS_LOCALSTORAGE:
        return None
    return LocalStorage(key="_ls_chat")


def _slim_sessions_for_storage(sessions: dict) -> dict:
    """序列化到 localStorage 前剝掉每則 chunk 的 payload.content（正文，體積最大）。

    送出 LLM 後，chunks 只剩「來源清單」與「右欄圖片」會用到，兩者都只讀 metadata
    （source / location / visuals），用不到正文。剝除可大幅降低 localStorage 體積，
    避開瀏覽器 ~5MB 配額上限。assistant 回答本身存在 message.content，不受影響。
    """
    slim: dict = {}
    for sid, sess in (sessions or {}).items():
        msgs = []
        for m in sess.get("messages", []):
            mm = {"role": m.get("role"), "content": m.get("content", "")}
            chs = m.get("chunks")
            if chs:
                mm["chunks"] = [
                    {
                        "score": c.get("score"),
                        "payload": {"metadata": (c.get("payload", {}) or {}).get("metadata", {})},
                    }
                    for c in chs
                ]
            msgs.append(mm)
        slim[sid] = {
            "title": sess.get("title", "新對話"),
            "messages": msgs,
            "last_chunks": [],
            "created": sess.get("created"),
        }
    return slim


def _save_chat_sessions() -> None:
    """標記對話紀錄為「待寫入瀏覽器」（mutation callback 用）。

    ⚠️ 不在這裡直接 setItem：callback 內 setItem 後緊接 st.rerun()，會在元件 delta
    flush 到前端「之前」就中止本次 script run → localStorage 從未真正寫入 → F5 後讀回空白。
    這是先前 F5 清空的根因。改為僅清掉 dirty 快取，真正寫入延到
    _render_chat_view() 尾端（正常完成的 render）由 _persist_chat_to_browser() 執行。
    """
    st.session_state.pop("_ls_last_blob", None)


def _persist_chat_to_browser() -> None:
    """把 chat_sessions + active id 實際寫入『瀏覽器 localStorage』。

    只在 _render_chat_view() 尾端、render 正常完成（沒有緊接 st.rerun）時呼叫，setItem
    元件 delta 才會 flush 到前端、真正落地 localStorage。dirty 比對（_ls_last_blob）讓
    內容沒變就不重寫，避免每個 rerun 都打 setItem（兼省潛在 rerun 抖動）。

    退化版：不再寫 server 磁碟（舊版單一 chat_sessions.json 會被同 server 全團隊共用、
    互看互蓋）。改存瀏覽器後每人各自獨立。無元件時退回純記憶體（F5 即失），不中斷對話。
    """
    if not HAS_LOCALSTORAGE:
        return
    try:
        payload = {
            "version": 2,
            "active": st.session_state.get("active_chat_session"),
            "sessions": _slim_sessions_for_storage(st.session_state.get("chat_sessions", {})),
        }
        blob = json.dumps(payload, ensure_ascii=False)
        if st.session_state.get("_ls_last_blob") == blob:
            return  # 內容無變動 → 不重寫
        ls = _get_local_storage()
        ls.setItem(CHAT_LS_KEY, blob, key="_ls_set_chat")
        st.session_state["_ls_last_blob"] = blob
        # 一旦自己寫過，記憶體即為真相來源 → 別再讓啟動對帳用瀏覽器舊值覆蓋
        st.session_state["_ls_chat_loaded"] = True
    except Exception as e:  # noqa: BLE001 — 持久化失敗不可中斷對話
        st.warning(f"聊天紀錄存入瀏覽器失敗（本輪仍在記憶體中）：{e}")


def _load_chat_sessions() -> tuple[dict, str | None]:
    """從瀏覽器 localStorage 載回 (sessions, active_id)；無資料或壞檔回 ({}, None)。"""
    if not HAS_LOCALSTORAGE:
        return {}, None
    try:
        ls = _get_local_storage()
        raw = ls.getItem(CHAT_LS_KEY)
        if not raw:
            return {}, None
        data = json.loads(raw) if isinstance(raw, str) else raw
        sessions = data.get("sessions", {}) or {}
        active = data.get("active")
        if not isinstance(sessions, dict):
            return {}, None
        if active not in sessions:
            active = None
        return sessions, active
    except Exception as e:  # noqa: BLE001 — 壞資料不可中斷啟動
        st.warning(f"瀏覽器聊天紀錄讀取失敗，將以空白開始：{e}")
        return {}, None


if "app_view" not in st.session_state:
    st.session_state.app_view = "chat"  # 預設進對話模式（審閱模式需密碼解鎖）

# chat_sessions[session_id] = {"title": str, "messages": [{"role","content"}], "last_chunks": [], "created": iso}
# 從瀏覽器 localStorage 對帳載回。元件首跑回 default（空）、真資料於下一個 rerun 才到，
# 因此用 _ls_chat_loaded flag 對帳：在真資料抵達（或自己存過）前不鎖定，避免把「首跑的空」
# 當成最終值而永遠載不到舊紀錄。
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_chat_session" not in st.session_state:
    st.session_state.active_chat_session = None
if HAS_LOCALSTORAGE and not st.session_state.get("_ls_chat_loaded"):
    _loaded_sessions, _loaded_active = _load_chat_sessions()
    if _loaded_sessions:  # 瀏覽器真資料已抵達 → 還原並鎖定
        st.session_state.chat_sessions = _loaded_sessions
        st.session_state.active_chat_session = _loaded_active
        st.session_state["_ls_chat_loaded"] = True
    # 否則（首跑 default 或瀏覽器本就空）：不鎖定，下個 rerun 再試；一旦使用者開始對話
    # 觸發 _save_chat_sessions 即把 flag 設 True、改以記憶體為真相來源。


def _chat_fail(active_sess: dict, text: str, chunks: list | None = None) -> None:
    """送出流程失敗時：append 一則 assistant 錯誤泡泡 + 存檔 + rerun。

    取代原本散落的 st.error + st.stop()。好處：(1) 失敗也有可見且持久化的回覆，
    (2) 絕不留「結尾是 user、無 assistant」的孤兒訊息毒化下一輪。rerun 後 chat_input
    已清空 → 不會重跑送出區，無迴圈風險。
    """
    active_sess["messages"].append({
        "role": "assistant",
        "content": text,
        "chunks": chunks or [],
    })
    _save_chat_sessions()
    st.rerun()


def _new_chat_session(title: str = "新對話") -> str:
    sid = hashlib.md5(f"{datetime.now(timezone.utc).isoformat()}|{title}".encode()).hexdigest()[:8]
    st.session_state.chat_sessions[sid] = {
        "title": title,
        "messages": [],
        "last_chunks": [],
        "created": now_iso(),
    }
    st.session_state.active_chat_session = sid
    _save_chat_sessions()
    return sid


def _open_local_file(path: Path) -> tuple[bool, str]:
    """Cross-platform 開檔。Win 用 os.startfile，Mac/Linux 用 subprocess。"""
    if not path.exists():
        return False, f"找不到檔案：{path}"
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            os.startfile(str(path))  # noqa: SIM115 — Win-only API
        elif sys_name == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True, f"已開啟：{path.name}"
    except Exception as e:
        return False, f"開啟失敗：{e}"


def _group_chunks_by_file_page(chunks: list[dict]) -> dict[tuple[str, int], dict]:
    """把 retrieved chunks 依 (file_key, page) 分組。
    回 {(file_key, page): {'max_score', 'project', 'chunks': [...]}}，按 max_score 降序排序。
    """
    groups: dict[tuple[str, int], dict] = {}
    for c in chunks:
        payload = c.get("payload", {}) or {}
        meta = payload.get("metadata", {}) or {}
        src = meta.get("source", {}) or {}
        loc = meta.get("location", {}) or {}
        fk = src.get("file_key", "")
        pg = loc.get("page")
        if not fk or not pg:
            continue
        key = (fk, pg)
        g = groups.setdefault(key, {
            "max_score": 0.0,
            "project": src.get("project_name", "—"),
            "chunks": [],
        })
        g["chunks"].append(c)
        g["max_score"] = max(g["max_score"], c.get("score", 0.0))
    return dict(sorted(groups.items(), key=lambda kv: -kv[1]["max_score"]))


def _collect_images_for_page(file_key: str, page: int, image_root: Path) -> list[Path]:
    """單個 (file_key, page) 的圖檔 abs paths，dedup + 過濾不存在檔。"""
    try:
        md_path = derive_md_path(file_key)
    except Exception:
        return []
    if not md_path.exists():
        return []
    try:
        _, _, pages = parse_md(md_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    page_md = next((pmd for pn, pmd in pages if pn == page), None)
    if not page_md:
        return []
    try:
        imgs = parse_images_in_page(page_md)
    except Exception:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for img in imgs:
        md_ref = img.get("md_path", "")
        if not md_ref:
            continue
        abs_p = image_root / md_ref
        k = str(abs_p)
        if k in seen or not abs_p.exists():
            continue
        seen.add(k)
        out.append(abs_p)
    return out


def _img_to_data_uri(path: Path) -> str:
    """讀檔轉 base64 data URI（給 inline HTML <img src> 用，繞 file:// 瀏覽器限制）。"""
    ext = path.suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
    mime = mime_map.get(ext, "png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _stacked_images_html(
    img_paths: list[Path],
    cap: int = 6,
    card_w: int = 220,
    card_h: int = 160,
) -> str:
    """疊圖卡牌風預覽：每張稍微旋轉 + 位移，最多顯示 cap 張，多的給 +N 標籤。"""
    if not img_paths:
        return "<div style='color:#888; padding:20px;'>（無圖片）</div>"
    cards_html = []
    n_show = min(len(img_paths), cap)
    for i, ip in enumerate(img_paths[:cap]):
        try:
            uri = _img_to_data_uri(ip)
        except Exception:
            continue
        # 中央往兩側散開：rot 由 -8°→+8°
        rot = (i - (n_show - 1) / 2) * (16 / max(n_show - 1, 1))
        offset_x = (i - (n_show - 1) / 2) * 14
        offset_y = i * 4
        cards_html.append(
            f'<img src="{uri}" '
            f'style="position:absolute; '
            f'top:{20 + offset_y}px; '
            f'left:calc(50% + {offset_x}px - {card_w // 2}px); '
            f'width:{card_w}px; height:{card_h}px; object-fit:cover; '
            f'border:3px solid white; border-radius:8px; '
            f'box-shadow:0 4px 12px rgba(0,0,0,0.25); '
            f'transform: rotate({rot:.1f}deg); '
            f'transition: transform 0.2s; '
            f'z-index:{i};" '
            f'onmouseover="this.style.zIndex=100;this.style.transform=\'rotate(0deg) scale(1.05)\';" '
            f'onmouseout="this.style.zIndex={i};this.style.transform=\'rotate({rot:.1f}deg)\';" '
            f'/>'
        )
    extra_badge = ""
    if len(img_paths) > cap:
        extra_badge = (
            f'<div style="position:absolute; bottom:6px; right:14px; '
            f'background:rgba(0,0,0,0.75); color:white; padding:3px 10px; '
            f'border-radius:12px; font-size:12px; font-weight:bold; z-index:101;">'
            f'+{len(img_paths) - cap}'
            f'</div>'
        )
    container_h = card_h + 60
    return (
        f'<div style="position:relative; width:100%; height:{container_h}px; '
        f'background:#f4f4f6; border-radius:10px; overflow:hidden; '
        f'margin:6px 0;">'
        f'{"".join(cards_html)}'
        f'{extra_badge}'
        f'</div>'
    )


@st.dialog("📑 圖片瀏覽器", width="large")
def _chat_image_gallery_dialog():
    """全螢幕 modal（自帶半透明背景）：顯示所有 retrieved 圖片的細節。"""
    imgs = st.session_state.get("_chat_dialog_imgs", [])
    metas = st.session_state.get("_chat_dialog_metas", [])
    if not imgs:
        st.caption("無圖片可顯示")
        return
    st.caption(f"共 {len(imgs)} 張 · 點擊外部關閉")
    for ip, meta in zip(imgs, metas):
        with st.container(border=True):
            st.markdown(f"**{meta}**")
            st.image(str(ip), use_container_width=True)


def _render_source_page_inline(source_path: Path, page: int, key_suffix: str,
                                dpi: int = 144) -> None:
    """伺服器端把來源檔的指定頁渲染成 PNG，內嵌顯示在網頁（同審閱模式做法）。

    PDF 直接渲染；PPT/PPTX 先用 LibreOffice 轉 PDF（需 soffice，自動偵測）。bytes
    經 HTTP 傳到瀏覽器 → 不論瀏覽器在哪台都看得到，不會像 os.startfile 開在 server 桌面。
    另附「下載原始檔」鈕，讓使用者把 server 上的原檔抓到本機。
    """
    ext = source_path.suffix.lower()
    try:
        if ext == ".pdf":
            pdf_path = source_path
        elif ext in (".ppt", ".pptx"):
            soffice = find_soffice()
            if not soffice:
                st.warning(
                    "此檔為 PPT/PPTX，需 LibreOffice 才能在網頁預覽。"
                    "請在 server 安裝 LibreOffice（Markdown／圖片預覽不受影響）。"
                )
                return
            with st.spinner(f"轉換 {source_path.name} → PDF 中..."):
                pdf_path = ensure_pdf(source_path, soffice)
        else:
            st.warning(f"不支援在網頁預覽的格式：`{ext}`")
            return
        with st.spinner(f"渲染 {source_path.name} 第 {page} 頁中..."):
            png_bytes = render_pdf_page_png(pdf_path, page - 1, dpi=dpi)
        st.image(png_bytes, use_container_width=True,
                 caption=f"{source_path.name} · 第 {page} 頁")
        try:
            st.download_button(
                "⬇ 下載原始檔",
                data=source_path.read_bytes(),
                file_name=source_path.name,
                key=f"_dl_{key_suffix}",
                use_container_width=True,
            )
        except Exception as e:  # noqa: BLE001 — 下載鈕失敗不影響預覽
            st.caption(f"（下載鈕無法載入：{e}）")
    except Exception as e:  # noqa: BLE001
        st.error(f"預覽失敗：\n```\n{e}\n```")


@st.cache_data(show_spinner=False, ttl=300)
def _locate_source_by_name(data_root_str: str, doc_type: str, filename: str) -> str:
    """改名後備援：chunk 存的 file_key 失準（如建案資料夾被改名）時，用「實際檔名」在
    `data_root/<類別>` 底下找實體檔，回絕對路徑字串；找不到回 ''。

    對話歷史的 chunk payload 是檢索當下的快照、不會隨 Qdrant 修正更新；此後備讓舊對話的
    「📂 開啟」仍能定位檔案（類別層通常沒改，只改建案層）。同檔名多筆 → 取第一個（罕見）。
    cache 5 分鐘避免每 rerun 重掃。"""
    base = Path(data_root_str) / doc_type if doc_type else Path(data_root_str)
    if not base.exists():
        base = Path(data_root_str)
    try:
        for p in base.rglob(filename):
            if p.is_file():
                return str(p)
    except OSError:
        pass
    return ""


def _resolve_chunk_source(data_root: Path, fk: str) -> Path:
    """先試 data_root/file_key；不存在則用檔名在類別資料夾底下回退搜尋（對改名免疫）。"""
    direct = data_root / fk
    if direct.exists():
        return direct
    parts = split_key_parts(fk)
    filename = parts[-1] if parts else ""
    doc_type = parts[0] if len(parts) > 1 else ""
    if filename:
        found = _locate_source_by_name(str(data_root), doc_type, filename)
        if found:
            return Path(found)
    return direct  # 仍找不到 → 回原路徑（不存在）→ UI 顯示「本機未找到」


def _render_chunk_sources(
    chunks: list[dict],
    data_root: Path,
    btn_key_prefix: str,
) -> None:
    """渲染來源文件清單（per-assistant-message expander 用）。
    每個 (file, page) 一行：檔名 + score + 📂 開啟鈕。按下「開啟」會在 server 渲染
    該引用頁成 PNG 內嵌於網頁顯示（再按一次收合）。不含圖片（圖片已在右側預覽顯示）。
    """
    groups = _group_chunks_by_file_page(chunks)
    if not groups:
        st.caption("（無有效來源）")
        return
    preview_state_key = f"{btn_key_prefix}_preview"  # 目前展開預覽的 (fk,pg) token
    for (fk, pg), info in groups.items():
        pdf_path = _resolve_chunk_source(data_root, fk)
        safe_fk = fk.replace("/", "_").replace("\\", "_")
        token = f"{safe_fk}|{pg}"
        row = st.columns([4, 2, 1], gap="small")
        with row[0]:
            if pdf_path.exists():
                st.markdown(f"📄 {Path(fk).name} · P{pg}", help=str(pdf_path))
            else:
                st.markdown(f"📄 {Path(fk).name} · P{pg}  ⚠ 本機未找到")
        with row[1]:
            st.caption(f"`{info['project']}` · score `{info['max_score']:.3f}`")
        with row[2]:
            open_key = f"{btn_key_prefix}_open_{safe_fk}_{pg}"
            is_open = st.session_state.get(preview_state_key) == token
            if st.button(
                "📂 收合" if is_open else "📂 開啟",
                key=open_key,
                use_container_width=True,
                disabled=not pdf_path.exists(),
                help="在網頁顯示此頁（再按一次收合）",
            ):
                # toggle：按下開啟 → 記住此 token；已開著再按 → 收合
                st.session_state[preview_state_key] = None if is_open else token
                st.rerun()
        # 展開中 → 在整列下方渲染頁面預覽
        if st.session_state.get(preview_state_key) == token and pdf_path.exists():
            _render_source_page_inline(pdf_path, pg, key_suffix=f"{btn_key_prefix}_{token}")


def _render_chat_view() -> None:
    """全螢幕 GPT 風格對話介面：
    - 左 sidebar：model / HyDE / LLM 參數 / session 管理 / 折疊的 Qdrant 設定
    - 主區左：聊天歷史 + chat input
    - 主區右：最近一輪 retrieved chunks 的源頁圖片
    """
    if not HAS_LLM:
        st.error("`openai` 套件或 `llm_chat.py` 未載入：`pip install openai`")
        st.stop()

    # ---- 左 sidebar ----
    sessions = st.session_state.chat_sessions
    active_sid = st.session_state.active_chat_session
    if active_sid not in sessions:
        active_sid = _new_chat_session()

    with st.sidebar:
        # === Session 管理（最上方）===
        st.subheader("📂 對話 Session")
        if st.button("➕ 新對話", use_container_width=True, type="primary",
                     key="_chat_new"):
            _new_chat_session()
            st.rerun()

        # session 清單（按建立時間反序）+ 每筆自帶垃圾桶
        for sid, sess in sorted(sessions.items(),
                                key=lambda kv: kv[1].get("created", ""), reverse=True):
            is_active = (sid == active_sid)
            label = sess.get("title", "新對話")
            n_msg = len(sess.get("messages", []))
            row = st.columns([4, 1, 1], gap="small")
            with row[0]:
                btn_label = f"{'📌 ' if is_active else '   '}{label}  ({n_msg})"
                if st.button(
                    btn_label,
                    key=f"_chat_pick_{sid}",
                    use_container_width=True,
                    type="secondary",
                    disabled=is_active,
                ):
                    st.session_state.active_chat_session = sid
                    _save_chat_sessions()
                    st.rerun()
            with row[1]:
                if st.button("✏️", key=f"_chat_rename_{sid}",
                             help="重新命名此對話", use_container_width=True):
                    # toggle 編輯狀態（再按一次收起）
                    cur = st.session_state.get("_chat_editing")
                    st.session_state["_chat_editing"] = None if cur == sid else sid
                    st.rerun()
            with row[2]:
                if st.button("🗑", key=f"_chat_del_{sid}",
                             help="刪除此對話", use_container_width=True):
                    sessions.pop(sid, None)
                    if st.session_state.get("_chat_editing") == sid:
                        st.session_state["_chat_editing"] = None
                    # 若刪掉的是 active，切到第一個剩下的；都沒了就開新的
                    if sid == active_sid:
                        next_sid = next(iter(sessions), None)
                        if next_sid is None:
                            next_sid = _new_chat_session()  # 內含存檔
                        st.session_state.active_chat_session = next_sid
                    _save_chat_sessions()
                    st.rerun()
            # 點鉛筆 → 行下方展開改名輸入
            if st.session_state.get("_chat_editing") == sid:
                ed = st.columns([4, 1, 1], gap="small")
                with ed[0]:
                    new_name = st.text_input(
                        "新名稱", value=label,
                        key=f"_chat_rename_input_{sid}",
                        label_visibility="collapsed",
                        placeholder="輸入對話名稱",
                    )
                with ed[1]:
                    if st.button("✓", key=f"_chat_rename_save_{sid}",
                                 help="儲存", use_container_width=True):
                        sess["title"] = new_name.strip() or "新對話"
                        st.session_state["_chat_editing"] = None
                        _save_chat_sessions()
                        st.rerun()
                with ed[2]:
                    if st.button("✕", key=f"_chat_rename_cancel_{sid}",
                                 help="取消", use_container_width=True):
                        st.session_state["_chat_editing"] = None
                        st.rerun()

        st.divider()

        # === 對話設定 ===
        st.subheader("💬 對話設定")

        # vLLM（OpenAI 相容）服務位址：base_url 需含 /v1
        sb_llm_url = st.text_input(
            "vLLM URL (OpenAI 相容，含 /v1)",
            value=st.session_state.get("_chat_llm_url", LLM_BASE_URL),
            key="_chat_llm_url",
        )
        # 動態抓服務的模型清單（GET /v1/models）；抓不到（服務不通）退回 .env 內建清單
        _remote_models = list_llm_models(sb_llm_url)
        sb_model_options = _remote_models or list(LLM_MODEL_CHOICES)
        # vLLM 通常只服務一個模型 → 有抓到遠端清單就以它為預設（.env 的 tag 多半對不上 vLLM id）
        _default_model = _remote_models[0] if _remote_models else (LLM_DEFAULT_MODEL or "")
        sb_current = st.session_state.get("_chat_model", _default_model)
        if sb_current and sb_current not in sb_model_options:
            sb_model_options.insert(0, sb_current)
        if not sb_model_options:
            sb_model_options = [""]  # 避免空 options 讓 selectbox 崩
        _sel_idx = sb_model_options.index(sb_current) if sb_current in sb_model_options else 0
        sb_model = st.selectbox(
            "Model", options=sb_model_options,
            index=_sel_idx,
            key="_chat_model",
            help="清單動態抓自 vLLM（/v1/models）；抓不到時退回 .env 內建。"
                 "vLLM 通常只服務一個模型；若清單無想要的，在 Custom 自填",
        )
        if not _remote_models:
            st.caption("（未連上 vLLM 或無模型 → 顯示 .env 內建清單）")
        else:
            st.caption(f"🟢 vLLM {len(_remote_models)} 個模型可用")
        sb_custom = st.text_input(
            "Custom model tag（覆寫上方）", value="",
            key="_chat_model_custom",
            placeholder="例如 qwen3:32b-instruct-q4_K_M",
        )
        active_model = sb_custom.strip() or sb_model

        # HyDE
        sb_hyde = st.toggle(
            "🔮 啟用 HyDE",
            value=st.session_state.get("_chat_hyde", False),
            key="_chat_hyde",
            help="用 LLM 把 query 改寫成「像是節錄段落」再 embed，"
                 "對口語 ↔ 文件術語落差大的查詢有幫助。"
                 "代價：每輪多一次 LLM 呼叫",
        )

        # RAG 設定
        with st.expander("📚 RAG 設定", expanded=True):
            sb_search_mode = st.selectbox(
                "檢索模式", options=["hybrid", "dense", "sparse"],
                index=0, key="_chat_search_mode",
            )
            sb_top_k = st.number_input(
                "Top-K", min_value=1, max_value=30,
                value=st.session_state.get("_chat_top_k", LLM_DEFAULT_TOP_K),
                step=1, key="_chat_top_k",
            )
            sb_rag_on = st.toggle(
                "🔗 注入檢索結果到 system context", value=True,
                key="_chat_rag_on",
                help="關掉 = 純 LLM 對話（不檢索）",
            )
            sb_smart_route = st.toggle(
                "🧭 智慧路由（比較／排序題）",
                value=st.session_state.get("_chat_smart_route", True),
                key="_chat_smart_route",
                help="開＝先用 LLM 判斷意圖再分派檢索：\n"
                     "・比較題 → 對每個建案各撈一次（涵蓋率不再靠 top-k 運氣）\n"
                     "・排序／極值／篩選／查屬性題 → 查人工維護的結構化屬性表（第 2 層），數字以表為準\n"
                     "・一般題 → 現行單發檢索\n"
                     "需 LLM 開啟；LLM 關閉或路由失敗時自動退回單發。",
            )
            # 第 2 層結構化屬性表狀態 + 熱載入（改 CSV 存檔即生效，此鈕僅供強制刷新）
            if HAS_STRUCTURED and sb_smart_route:
                _stpath = st.session_state.get("_structured_path", STRUCTURED_TABLE_PATH)
                if os.path.exists(_stpath):
                    _td = _load_structured_cached(_stpath, os.path.getmtime(_stpath))
                    if _td is not None and _td.ok:
                        st.caption(
                            f"📊 屬性表：{len(_td.projects)} 建案 × {len(_td.attributes)} 屬性"
                            f"（{len(_td.rows)} 列）"
                        )
                    elif _td is not None and _td.error:
                        st.caption(f"📊 屬性表讀取異常：{_td.error}")
                    else:
                        st.caption("📊 屬性表：模板就緒，尚未填入資料")
                else:
                    st.caption(f"📊 屬性表未找到：{_stpath}")
                if st.button("🔄 重新載入屬性表", key="_structured_reload",
                             help="改完 CSV 存檔通常會自動生效；此鈕清快取強制刷新"):
                    _load_structured_cached.clear()
                    st.rerun()
            # 「指定專案」多選框不放這裡，改置於聊天室正上方的 sticky bar（見主區）

        # LLM 參數
        with st.expander("⚙️ LLM 參數", expanded=False):
            sb_temp = st.slider(
                "temperature", 0.0, 2.0,
                value=st.session_state.get("_chat_temp", LLM_DEFAULT_TEMPERATURE),
                step=0.05, key="_chat_temp",
            )
            sb_top_p = st.slider(
                "top_p", 0.0, 1.0,
                value=st.session_state.get("_chat_top_p", LLM_DEFAULT_TOP_P),
                step=0.05, key="_chat_top_p",
            )
            sb_num_predict = st.number_input(
                "max_tokens (max output tokens)", min_value=128, max_value=16384,
                value=st.session_state.get("_chat_num_predict", LLM_DEFAULT_NUM_PREDICT),
                step=128, key="_chat_num_predict",
            )
            st.caption(
                "ℹ️ context 長度（num_ctx）由 vLLM 服務端 `--max-model-len` 決定，"
                "非 per-request 參數，故此處不再提供。"
            )

        # 連線設定（Qdrant + 原始檔路徑）
        with st.expander("🗄 連線設定（Qdrant / 路徑）", expanded=False):
            sb_qdrant_url = st.text_input(
                "Qdrant URL", value=st.session_state.get("_chat_qdrant_url", QDRANT_DEFAULT_URL),
                key="_chat_qdrant_url",
            )
            sb_qdrant_key = st.text_input(
                "API key", value=st.session_state.get("_chat_qdrant_key", ""),
                type="password", key="_chat_qdrant_key",
            )
            sb_qdrant_coll = st.text_input(
                "Collection", value=st.session_state.get("_chat_qdrant_coll", QDRANT_TEXT_COLLECTION),
                key="_chat_qdrant_coll",
            )
            sb_image_root = st.text_input(
                "圖片根目錄",
                value=st.session_state.get("_chat_image_root", str(MKDATA_PATH)),
                key="_chat_image_root",
                help="右側資料塊圖片從這裡找（mkdata/ 預設）",
            )
            # 一次性遷移：舊版預設誤指到 doc_type 子層（…/12.個案銷講資料），開檔會全 fail。
            # 偵測到既有 session 還存著這個壞值就自動改回 DATA_ROOT，讓現有 session 立即生效
            # （預設只在 key 不存在時生效、不會覆蓋既有值，故需此遷移）。見 [[open_file_data_root_layer_bug]]。
            _good_data_root = DATA_ROOT_ENV or DEFAULT_DATA_PATH
            if st.session_state.get("_chat_data_root") == r"D:/璞真RAG資料夾/12.個案銷講資料":
                st.session_state["_chat_data_root"] = _good_data_root
            sb_data_root = st.text_input(
                "原始檔根目錄（PDF/PPT 用）",
                value=st.session_state.get("_chat_data_root", _good_data_root),
                key="_chat_data_root",
                help="開啟本機原始檔用：data_root / file_key（須為 DATA_ROOT＝類別資料夾的父層）。"
                     "預設與審閱模式 sidebar 同款（.env DATA_ROOT 優先）",
            )

    # ---- 主區 ----
    active_sid = st.session_state.active_chat_session
    active_sess = st.session_state.chat_sessions.get(active_sid)
    if active_sess is None:
        st.warning("尚無 active session。請點左側「➕ 新對話」")
        return

    # === 聊天室正上方「指定專案／文件種類」sticky bar ===
    # 釘在 Streamlit header 下方（position:sticky），捲動對話歷史時固定不被泡泡蓋住。
    # 背景隨主題自適應：用 st.context.theme.type（依實際背景推斷 light/dark）挑色，
    # 取代先前寫死的白底（深色主題下會出現白方塊）。
    try:
        _theme_type = st.context.theme.type or "light"
    except Exception:  # noqa: BLE001 — 取不到主題就當淺色
        _theme_type = "light"
    _bar_bg = "#0e1117" if _theme_type == "dark" else "#ffffff"
    _bar_border = "rgba(250,250,250,0.20)" if _theme_type == "dark" else "rgba(0,0,0,0.10)"
    st.markdown(
        f"""
        <style>
        .st-key-chat_project_bar {{
            position: sticky;
            top: 3.75rem;            /* ≈ Streamlit 固定 header 高度，貼其下方不被蓋住 */
            z-index: 100;
            background: {_bar_bg};
            padding: 0.45rem 0.2rem;
            margin-bottom: 0.3rem;
            border-bottom: 1px solid {_bar_border};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    sb_projects: list[str] = []
    sb_doc_types: list[str] = []
    with st.container(key="chat_project_bar"):
        if sb_rag_on:
            _qurl = st.session_state.get("_chat_qdrant_url", QDRANT_DEFAULT_URL)
            _qkey = st.session_state.get("_chat_qdrant_key", "")
            _qcoll = st.session_state.get("_chat_qdrant_coll", QDRANT_TEXT_COLLECTION)
            _dt_opts = list_doc_types_in_collection(_qurl, _qkey, _qcoll)
            # 顯示順序：指定專案（上）→ 文件種類（下）。連動仍以「文件種類」為父：
            # 先從 session_state 讀文件種類的當前選擇來篩專案選項——即使文件種類渲染在下方
            # 也成立（widget 變動會在 rerun 前寫回 session_state，下個 run 即反映）。
            _sel_dt = st.session_state.get("_chat_filter_doc_types", [])
            # 專案連動：只列「所選文件種類底下」的專案，杜絕選出不存在的（類別✕專案）空組合 → 查無結果
            _proj_opts = list_projects_in_collection(
                _qurl, _qkey, _qcoll, doc_types=_sel_dt or None,
            )
            # 防呆：類別變動使某些已選專案失效時，先剃掉再渲染（避免殘留無效值 → AND 後空交集）
            _cur_proj = st.session_state.get("_chat_filter_projects", [])
            _valid_proj = [p for p in _cur_proj if p in _proj_opts]
            if _valid_proj != _cur_proj:
                st.session_state["_chat_filter_projects"] = _valid_proj
            # 指定專案（上）
            sb_projects = st.multiselect(
                "🏗 指定專案（留空＝該類別全部）",
                options=_proj_opts,
                key="_chat_filter_projects",
                help="只列出下方所選文件種類底下的專案；未選文件種類時列全部（多選 OR）。",
            )
            # 文件種類（下）
            sb_doc_types = st.multiselect(
                "📁 文件種類（留空＝全部）",
                options=_dt_opts, default=[],
                key="_chat_filter_doc_types",
                help="選文件種類會連動上方專案清單，只列該類別底下實際存在的專案（多選 OR）。",
            )
            if not _dt_opts and not _proj_opts:
                st.caption("（未取得清單：Qdrant 未連線或 collection 為空 → 將不套用篩選）")
            elif _sel_dt and not _proj_opts:
                st.caption("（所選文件種類底下查無專案——可能該類別尚未 ingest）")
        else:
            st.caption("🔗 RAG 已關閉（純 LLM 對話）—— 開啟後可在此指定專案／文件種類")

    col_chat, col_imgs = st.columns([2, 1], gap="large")

    with col_chat:
        # === 標題（唯讀顯示）：改名改到左側「對話 Session」清單的 ✏️ 鈕 ===
        st.markdown(f"### 💬 {active_sess.get('title', '新對話')}")

        st.caption(
            f"Model: `{active_model}` · HyDE: {'on' if sb_hyde else 'off'} · "
            f"RAG: {'on' if sb_rag_on else 'off'} ({sb_search_mode} top-{sb_top_k})"
        )

        image_root_chat = Path(sb_image_root)
        data_root_chat = Path(sb_data_root)

        # 歷史：assistant 訊息下方掛「來源文件」expander
        for msg_idx, m in enumerate(active_sess["messages"]):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    for _note in (m.get("notices") or []):
                        st.caption(f"🧭 {_note}")
                    msg_chunks = m.get("chunks") or []
                    if msg_chunks:
                        with st.expander(
                            f"📑 來源文件（{len(_group_chunks_by_file_page(msg_chunks))} 檔／"
                            f"{len(msg_chunks)} chunks）",
                            expanded=False,
                        ):
                            _render_chunk_sources(
                                msg_chunks,
                                data_root_chat,
                                btn_key_prefix=f"_src_{active_sid}_{msg_idx}",
                            )

            # 本輪 in-flight 對話的串流容器：放在歷史下方、輸入框上方。
        # 送出處理區（在 col 之外）會 `with live_turn:` 把 user 泡泡 + 串流 assistant
        # 泡泡寫進這裡，確保串流內容渲染在正確的對話欄位、且 websocket 持續有流量。
        live_turn = st.container()

        # 輸入列：送出箭頭旁加「🤖 LLM」開關。關閉＝僅檢索模式（不呼叫 LLM，回覆「查詢到 N 個
        # 相關文件」並展示 RAG 來源／圖片）。
        in_cols = st.columns([6, 1], vertical_alignment="bottom")
        with in_cols[0]:
            user_q = st.chat_input("輸入問題（Enter 送出）...")
        with in_cols[1]:
            sb_llm_on = st.toggle(
                "🤖 LLM",
                value=st.session_state.get("_chat_llm_on", True),
                key="_chat_llm_on",
                help="開＝LLM 生成回答；關＝僅檢索，回覆「查詢到 N 個相關文件」並展示來源（完全不呼叫 LLM）",
            )

    with col_imgs:
        st.subheader("🖼 相關圖片")
        # 找最後一個 assistant 訊息的 chunks（= 本輪檢索結果）
        latest_asst = next(
            (m for m in reversed(active_sess["messages"]) if m["role"] == "assistant"),
            None,
        )
        latest_chunks = (latest_asst or {}).get("chunks") or []
        if not latest_chunks:
            st.info("送出 RAG 對話後，這裡會顯示疊圖預覽，點擊可開啟瀏覽器。")
        else:
            # 收集 + dedup
            all_imgs: list[Path] = []
            all_metas: list[str] = []
            seen: set[str] = set()
            groups = _group_chunks_by_file_page(latest_chunks)
            for (fk, pg), _info in groups.items():
                for ip in _collect_images_for_page(fk, pg, image_root_chat):
                    k = str(ip)
                    if k in seen:
                        continue
                    seen.add(k)
                    all_imgs.append(ip)
                    all_metas.append(f"{Path(fk).name} · P{pg}")

            if not all_imgs:
                st.caption("（本輪檢索到的 chunks 對應頁面無可顯示圖片）")
            else:
                st.caption(f"本輪 {len(all_imgs)} 張 · 點下方按鈕開啟瀏覽器看細節")
                st.markdown(
                    _stacked_images_html(all_imgs),
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"🔍 開啟圖片瀏覽器（{len(all_imgs)} 張）",
                    use_container_width=True,
                    type="primary",
                    key=f"_chat_open_gallery_{active_sid}",
                ):
                    st.session_state["_chat_dialog_imgs"] = all_imgs
                    st.session_state["_chat_dialog_metas"] = all_metas
                    _chat_image_gallery_dialog()

    # ---- 送出處理 ----
    if user_q and user_q.strip():
        q = user_q.strip()

        # 先把 user msg 加入 history
        active_sess["messages"].append({"role": "user", "content": q})

        # 立即在 in-flight 容器渲染 user 泡泡（不必等整輪生成完才看到自己的提問）
        with live_turn:
            with st.chat_message("user"):
                st.markdown(q)

        # 1. RAG 檢索：RAG 開啟，或 LLM 關閉（僅檢索模式必須有結果可展示）時都要跑
        rag_chunks: list[dict] = []
        route_intent = "fact"          # 三路 router 結果：fact / compare / structured
        route_grouped = False          # chunks 是否依建案分組（compare/structured fan-out）
        route_structured_md = None     # structured 路徑撈到的屬性表 markdown（權威數值來源）
        route_notices: list[str] = []  # 給使用者看的路由訊息（截斷/未對應/退回等，不靜默）
        do_retrieval = sb_rag_on or not sb_llm_on
        if do_retrieval:
            if not HAS_QDRANT:
                _chat_fail(active_sess, "⚠️ 需要檢索但 `qdrant-client` 未安裝。請安裝，或開啟 LLM 改用純對話。")
            try:
                cli = get_qdrant_client(sb_qdrant_url, sb_qdrant_key)
            except Exception as e:
                _chat_fail(active_sess, f"⚠️ Qdrant 連線失敗：\n```\n{e}\n```")
            if not cli.collection_exists(sb_qdrant_coll):
                _chat_fail(active_sess, f"⚠️ Collection `{sb_qdrant_coll}` 不存在")
            # Embedder：主執行緒前景載入，cache_resource 只載一次。預設 CPU（device 預設
            # cpu）—— 2050 的 4GB 顯存放不下 Streamlit 行程內的 bge-m3，GPU 路徑會 CUDA
            # OOM 讓整個行程 abort。CPU 載入約 2-3s、單句 encode <1s，互動查詢足夠。
            try:
                embedder = load_embedder(
                    st.session_state.get("embed_model", EMBEDDING_MODEL),
                    st.session_state.get("embed_fp16", True),
                    st.session_state.get("embed_device", "cpu"),
                )
            except Exception as e:
                _chat_fail(active_sess, f"⚠️ Embedder 載入失敗：\n```\n{e}\n```")

            llm_cli = get_llm_client(sb_llm_url)

            # 2. 意圖路由。base_must＝sidebar 硬約束（專案 + 文件種類；條內 OR、條間 AND），
            #    router 不得覆蓋。僅「LLM 開 + 智慧路由開」才呼叫 router；否則 plan=None → 單發 fact。
            _must = []
            if sb_projects:
                _must.append(qm.FieldCondition(
                    key="metadata.source.project_name",
                    match=qm.MatchAny(any=list(sb_projects)),
                ))
            if sb_doc_types:
                _must.append(qm.FieldCondition(
                    key="metadata.source.doc_type",
                    match=qm.MatchAny(any=list(sb_doc_types)),
                ))

            plan = None
            if sb_smart_route and sb_llm_on and llm_analyze_query is not None:
                try:
                    with st.spinner("分析查詢意圖..."):
                        plan = llm_analyze_query(llm_cli, active_model, q)
                except Exception as e:  # noqa: BLE001 — router 失敗 → 退回單發 fact，UI 明講
                    plan = None
                    route_notices.append(f"智慧路由失敗，已退回一般檢索：{e}")

            # 3. 依 plan 三路檢索（fact 單發 / compare 每建案 fan-out / rank 未排序參考）。
            #    失敗也走 _chat_fail，不留孤兒 user。
            _intent = plan.intent if plan else "fact"
            _label = {
                "compare": "逐建案比較檢索",
                "structured": "查結構化屬性表",
            }.get(_intent, f"檢索中（{sb_search_mode} top-{sb_top_k}）")
            try:
                with st.spinner(f"{_label}..."):
                    _route = route_and_retrieve(
                        embedder=embedder, cli=cli, coll=sb_qdrant_coll,
                        search_mode=sb_search_mode, query=q, top_k=int(sb_top_k),
                        plan=plan, base_must=_must,
                        sidebar_projects=list(sb_projects),
                        sidebar_doc_types=list(sb_doc_types),
                        qdrant_url=sb_qdrant_url, qdrant_key=sb_qdrant_key,
                        llm_client=llm_cli, model=active_model,
                        do_hyde=(sb_hyde and sb_llm_on),
                    )
            except Exception as e:
                _chat_fail(active_sess, f"⚠️ 檢索失敗：\n```\n{e}\n```")
            rag_chunks = _route["chunks"]
            route_intent = _route["intent"]
            route_grouped = _route["grouped"]
            route_structured_md = _route.get("structured_md")
            route_notices.extend(_route["notices"])
        else:
            llm_cli = get_llm_client(sb_llm_url)

        if sb_llm_on:
            # 4. 組 messages。system prompt 依意圖路由選：compare → 逐案比較 + 涵蓋率提醒；
            #    rank → 明講第 2 層未接入、禁止自行排序；fact → 現行 RAG / 純對話。
            if route_intent == "compare":
                sys_prompt = LLM_COMPARE_SYSTEM_PROMPT or LLM_RAG_SYSTEM_PROMPT
            elif route_intent == "structured":
                sys_prompt = LLM_STRUCTURED_SYSTEM_PROMPT or LLM_RAG_SYSTEM_PROMPT
            elif rag_chunks:
                sys_prompt = LLM_RAG_SYSTEM_PROMPT
            else:
                sys_prompt = LLM_DEFAULT_SYSTEM_PROMPT
            messages = [{"role": "system", "content": sys_prompt}]
            # 歷史（不含剛加的 user msg）
            prior = active_sess["messages"][:-1]
            # 防呆：剝除結尾孤兒 user（無對應 assistant），避免送出連續兩個 user role
            # 導致部分模型回空字串。正常交替歷史結尾為 assistant → 此迴圈不觸發、零 context 損失。
            while prior and prior[-1].get("role") == "user":
                prior = prior[:-1]
            messages.extend(prior[-10:])  # 保留近 5 輪（user+assistant）
            ctx_blocks: list[str] = []
            if route_structured_md:
                ctx_blocks.append(
                    "【結構化資料表】（權威數值來源；排序／計數／任何數字只能依這張表）\n"
                    + route_structured_md
                )
            if rag_chunks:
                ctx_text = llm_format_chunks(rag_chunks, group_by_entity=route_grouped)
                ref_label = (
                    "【參考資料】（僅供質性補充，勿從中取數字排序）"
                    if route_structured_md else "【參考資料】"
                )
                ctx_blocks.append(f"{ref_label}\n{ctx_text}")
            if ctx_blocks:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(ctx_blocks) + f"\n\n【問題】\n{q}",
                })
            else:
                messages.append({"role": "user", "content": q})

            # 5. LLM 呼叫（串流）—— 底層斷線治本：生成期間 token 持續流過 websocket，
            # 連線不會 idle、UI 即時逐字更新。answer = 串流全文（st.write_stream 回傳）。
            answer = ""
            try:
                with live_turn:
                    with st.chat_message("assistant"):
                        # vLLM（OpenAI 相容）：num_ctx 不是 per-request 參數（context 長度由
                        # 服務端 --max-model-len 決定），故不傳；num_predict → max_tokens。
                        stream = llm_cli.chat.completions.create(
                            model=active_model,
                            messages=messages,
                            temperature=float(sb_temp),
                            top_p=float(sb_top_p),
                            max_tokens=int(sb_num_predict),
                            stream=True,
                        )
                        answer = st.write_stream(_llm_stream_tokens(stream))
            except Exception as e:
                _chat_fail(active_sess, f"⚠️ LLM 呼叫失敗：`{e}`", chunks=rag_chunks)

            # 空回答防呆：模型偶爾回空字串（num_predict 太小／上下文異常）→ 別塞空白泡泡
            if not (answer or "").strip():
                answer = "⚠️ 模型回傳空內容（可能 num_predict 太小或上下文異常）。請重試，或調整 num_predict／檢查對話歷史。"
        else:
            # LLM 關閉 → 僅檢索模式：固定回覆 + 由下方來源 expander／右欄圖片展示 RAG 結果
            answer = f"查詢到 {len(rag_chunks)} 個相關文件"
            with live_turn:
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    for _note in route_notices:
                        st.caption(f"🧭 {_note}")

        active_sess["messages"].append({
            "role": "assistant",
            "content": answer,
            "chunks": rag_chunks,
            "notices": route_notices,
        })

        # auto-title：title 仍是預設「新對話」就用「第一則 user 問題」生成重點標題；
        # 用第一則 user（非本輪 q）→ 即使前面有錯誤turn，首次成功回答後仍能正確命名。
        if active_sess.get("title") == "新對話":
            first_user = next(
                (m["content"] for m in active_sess["messages"] if m.get("role") == "user"),
                q,
            )
            fallback_title = first_user[:20] + ("…" if len(first_user) > 20 else "")
            title = ""
            if sb_llm_on and llm_generate_title is not None:
                try:
                    with st.spinner("產生對話標題中..."):
                        title = llm_generate_title(llm_cli, active_model, first_user)
                except Exception:
                    title = ""  # LLM 失敗 → 用 fallback
            active_sess["title"] = title or fallback_title

        _save_chat_sessions()  # 標記 dirty；實際寫入由下方尾端 persist 執行
        st.rerun()

    # 持久化到瀏覽器：放在「正常完成的 render」尾端，setItem 元件 delta 才會 flush 到前端
    # 真正落地（mutation callback 內 setItem→st.rerun 會在 flush 前被中止 → F5 清空的根因）。
    _persist_chat_to_browser()


# === 頂部 toolbar：右上角 view 切換鈕 ===
_topbar_cols = st.columns([5, 1])
with _topbar_cols[0]:
    st.markdown(
        "🗂 **審閱模式**" if st.session_state.app_view == "review"
        else "💬 **對話模式**"
    )
with _topbar_cols[1]:
    if st.session_state.app_view == "review":
        if st.button("💬 對話模式", use_container_width=True, key="_btn_switch_chat"):
            st.session_state.app_view = "chat"
            st.rerun()
    else:
        if st.button("📋 審閱模式", use_container_width=True, key="_btn_switch_review"):
            st.session_state.app_view = "review"
            st.rerun()

if st.session_state.app_view == "chat":
    _render_chat_view()
    st.stop()

# === 審閱模式密碼鎖（解鎖後本 session 維持解鎖）===
if not st.session_state.get("review_unlocked", False):
    st.title("🔒 審閱模式")
    st.caption("審閱模式為 ETL 標記工具，需密碼進入。")
    with st.form("_review_unlock_form"):
        pw = st.text_input("請輸入密碼", type="password")
        submitted = st.form_submit_button("解鎖", type="primary")
    if submitted:
        if pw == REVIEW_PASSWORD:
            st.session_state.review_unlocked = True
            st.rerun()
        else:
            st.error("密碼錯誤")
    if st.button("← 返回對話模式", key="_review_lock_back"):
        st.session_state.app_view = "chat"
        st.rerun()
    st.stop()

# review 模式才顯示原標題
st.title("PDF ↔ Markdown 比對 + Chunk Preview")


def _pathfix_scan(
    data_root: Path, scan_dir: Path,
    url: str, api_key: str, collection: str, selected_doc_type: str,
) -> dict:
    """掃描磁碟現況 vs Qdrant payload，用 file_hash（內容 md5）配對，找出「改了路徑名」的檔。

    為何用 file_hash：改資料夾名後 stem/相對路徑都可能對不上，唯有內容雜湊是穩定身分 →
    對改名免疫、也繞過 stem 撞號。掃描限縮在選定的文件種類資料夾，避免整樹重算 md5。
    回 dict：diffs（待修）/ n_same（一致）/ not_in_db（磁碟有 DB 沒有，需跑 v5）/
    db_orphan（DB 有但磁碟內容找不到）/ 計數。純讀取，不寫任何資料。
    """
    if not scan_dir.exists():
        return {"error": f"掃描目錄不存在：{scan_dir}"}
    # 1) 掃磁碟（限縮 scan_dir）→ {file_hash: 新 file_key}
    disk_by_hash: dict[str, str] = {}
    disk_collision: set[str] = set()
    for ext in ("*.pdf", "*.ppt", "*.pptx"):
        for p in scan_dir.rglob(ext):
            if not p.is_file():
                continue
            h = compute_file_hash(p)
            if not h:
                continue
            new_fk = str(p.relative_to(data_root))
            if h in disk_by_hash and disk_by_hash[h] != new_fk:
                disk_collision.add(h)  # 同內容多檔（真重複），罕見
            disk_by_hash[h] = new_fk
    # 2) 掃 Qdrant：filter doc_type（用「解析後」的 doc_type，對齊 payload 實際值）→ 依 file_hash 分組
    try:
        cli = get_qdrant_client(url, api_key)
        if not cli.collection_exists(collection):
            return {"error": f"collection 不存在：{collection}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Qdrant 連線失敗：{e}"}
    dtm = load_doc_type_map()
    resolved_dt = str(dtm.get(selected_doc_type) or selected_doc_type)
    flt = qm.Filter(must=[qm.FieldCondition(
        key="metadata.source.doc_type", match=qm.MatchValue(value=resolved_dt),
    )])
    db_by_hash: dict[str, dict] = {}
    nohash = 0
    offset = None
    try:
        while True:
            pts, offset = cli.scroll(
                collection, scroll_filter=flt, limit=512, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for pt in pts:
                src = ((pt.payload or {}).get("metadata") or {}).get("source") or {}
                h = (src.get("file_hash") or "").strip()
                if not h:
                    nohash += 1
                    continue
                g = db_by_hash.setdefault(
                    h, {"fk": src.get("file_key") or "", "ids": [], "src": src})
                g["ids"].append(pt.id)
            if offset is None:
                break
    except Exception as e:  # noqa: BLE001
        return {"error": f"Qdrant 掃描失敗：{e}"}
    # 3) 比對（以磁碟內容雜湊為準）
    diffs: list[dict] = []
    n_same = 0
    not_in_db: list[str] = []
    for h, new_fk in disk_by_hash.items():
        g = db_by_hash.get(h)
        if g is None:
            not_in_db.append(new_fk)  # 磁碟有、DB 沒有 → 未入庫或 stem 撞號被覆蓋
            continue
        if str(g["fk"]).replace("\\", "/") == new_fk.replace("\\", "/"):
            n_same += 1
            continue
        src = g["src"]
        diffs.append({
            "old_fk": g["fk"], "new_fk": new_fk, "ids": g["ids"], "src": src,
            "new_dt": resolve_doc_type(new_fk, src, dtm),
            "new_pj": derive_project_name(new_fk),
            "old_pj": src.get("project_name", "—"),
        })
    disk_hashes = set(disk_by_hash)
    db_orphan = sorted(g["fk"] for hh, g in db_by_hash.items() if hh not in disk_hashes)
    return {
        "diffs": diffs, "n_same": n_same, "not_in_db": sorted(not_in_db),
        "db_orphan": db_orphan, "n_disk": len(disk_by_hash), "n_db": len(db_by_hash),
        "nohash": nohash, "collision": sorted(disk_collision), "resolved_dt": resolved_dt,
    }


def _pathfix_apply(url: str, api_key: str, collection: str, diffs: list[dict]) -> str:
    """把 diffs 的新路徑寫回 payload：set_payload(key='metadata.source') 整檔覆蓋。
    point_id 是內容雜湊衍生、未變 → 原地覆蓋、不動向量、不重 embed。[[feedback-data-dependency-audit-first]]"""
    cli = get_qdrant_client(url, api_key)
    done = fails = 0
    errs: list[str] = []
    for d in diffs:
        new_src = dict(d["src"])
        new_src.update(
            file_key=d["new_fk"], file_path=d["new_fk"],
            project_name=d["new_pj"], doc_type=d["new_dt"],
        )
        try:
            cli.set_payload(collection, payload=new_src, points=d["ids"], key="metadata.source")
            done += 1
        except Exception as e:  # noqa: BLE001
            fails += 1
            errs.append(f"{d['new_fk']}: {e}")
    msg = f"✅ 已修正 {done}/{len(diffs)} 檔的 file_key（原地覆蓋 payload，向量未動）。"
    if fails:
        msg += f"　⚠️ 失敗 {fails}：" + "；".join(errs[:5])
    return msg


def _render_path_correction(
    data_root: Path, selected_doc_type: str,
    qdrant_url: str, qdrant_api_key: str, collection: str,
) -> None:
    """🧭 路徑修正：改了資料夾名（如建案）後，把 Qdrant payload 的 file_key 同步成新路徑。

    症狀：RAG 仍查得到（用存好的向量），但「📂 開啟原始檔」會『本機未找到』、指定專案下拉
    少了新名——因為 payload 的 file_key/project_name 還是舊路徑。本區塊用檔案內容（file_hash）
    比對磁碟與 payload，安全地原地改 file_key/file_path/project_name/doc_type，不重 embed。
    """
    st.divider()
    st.subheader("🧭 路徑修正（改名後同步 file_key）")
    # 修正報告（apply 後 rerun 才顯示）
    _rep = st.session_state.pop("_pathfix_report", None)
    if _rep:
        (st.success if _rep.startswith("✅ 已修正") and "失敗" not in _rep else st.warning)(_rep)
    st.caption(
        "改了資料夾名後 RAG 仍查得到，但「📂 開啟原始檔」會『本機未找到』、指定專案下拉也少了"
        "新名——因 payload 的 file_key 還是舊路徑。這裡用檔案內容（file_hash）比對，安全改成新路徑。"
    )
    if not HAS_QDRANT:
        st.warning("qdrant-client 未安裝，無法修正。")
        return
    if not selected_doc_type:
        st.info("請先在上方選一個「文件種類資料夾」；掃描會限縮在該資料夾底下（避免整樹重算 md5）。")
        return

    scan_dir = data_root / selected_doc_type
    st.caption(f"掃描範圍：`{scan_dir}`　·　比對鍵＝檔案內容 md5（對改名免疫）")
    _scan_key = f"_pathfix_scan::{collection}::{selected_doc_type}"

    if st.button("🔍 掃描路徑差異", key=f"_pathfix_btn_{selected_doc_type}",
                 use_container_width=True):
        with st.spinner("掃描磁碟 + Qdrant，計算內容雜湊中…"):
            st.session_state[_scan_key] = _pathfix_scan(
                data_root, scan_dir, qdrant_url, qdrant_api_key, collection, selected_doc_type)

    result = st.session_state.get(_scan_key)
    if not result:
        return
    if result.get("error"):
        st.error(result["error"])
        return

    diffs = result["diffs"]
    st.markdown(
        f"磁碟檔 **{result['n_disk']}**／DB 此類別（`{result['resolved_dt']}`）檔 **{result['n_db']}**"
        f"　｜　✅ 一致 {result['n_same']}　🔧 待修 **{len(diffs)}**"
        f"　🆕 未入庫 {len(result['not_in_db'])}　❓DB 無磁碟對應 {len(result['db_orphan'])}"
    )
    if result["collision"]:
        st.warning(f"⚠️ 磁碟有 {len(result['collision'])} 組同內容多檔（真重複）→ 取最後掃到的路徑。")

    if diffs:
        with st.expander(f"🔧 待修正（{len(diffs)} 檔）", expanded=True):
            for d in diffs[:200]:
                st.markdown(
                    f"- `{d['old_fk']}`　→　`{d['new_fk']}`"
                    f"　（{len(d['ids'])} pts｜專案 `{d['old_pj']}`→`{d['new_pj']}`）")
            if len(diffs) > 200:
                st.caption(f"…另 {len(diffs) - 200} 檔（套用時全部處理）")
    if result["not_in_db"]:
        with st.expander(f"🆕 磁碟有、DB 沒有：{len(result['not_in_db'])}（未入庫或 stem 撞號被覆蓋 → 需跑 v5）"):
            for fk in result["not_in_db"][:100]:
                st.markdown(f"- `{fk}`")
    if result["db_orphan"]:
        with st.expander(f"❓ DB 有、磁碟內容找不到：{len(result['db_orphan'])}（檔已移走 / 改在別類別 / 撞號）"):
            for fk in result["db_orphan"][:100]:
                st.markdown(f"- `{fk}`")

    if not diffs:
        st.success("沒有需要修正的路徑差異 👍")
        return

    _confirm = st.checkbox(
        f"我確認把上列 {len(diffs)} 檔的 file_key 改成新路徑（原地覆蓋 payload、不動向量）",
        key=f"_pathfix_confirm_{selected_doc_type}")
    if st.button("✅ 套用路徑修正", key=f"_pathfix_apply_{selected_doc_type}",
                 disabled=not _confirm, use_container_width=True):
        st.session_state["_pathfix_report"] = _pathfix_apply(
            qdrant_url, qdrant_api_key, collection, diffs)
        st.session_state.pop(_scan_key, None)  # 清掉舊掃描結果，逼下次重掃
        try:
            list_facet_values.clear()  # 清 facet cache，讓指定專案下拉立即反映新名
        except Exception:  # noqa: BLE001
            pass
        st.rerun()


def _render_fast_pipeline(
    file_keys, _file_status_map, image_root, data_path,
    qdrant_url, qdrant_api_key, qdrant_collection_name,
):
    """fast pipeline 主體：批次 encode → Qdrant。在 sidebar 內、通過密碼鎖後才呼叫。

    模式切換：
      🛡 保護模式 → 略過已入庫(🔵)檔，只處理未入庫的（安全預設）。
      ♻️ 全部重來 → 整批（含已入庫）重新 encode + 重新上傳；point_id deterministic 會原地
                     覆蓋，適合改了 doc_type／chunk 規則後整資料夾重建。
    """
    # 上一次批次結果（rerun 後才顯示，避免被 st.rerun 清掉）
    _fp_report = st.session_state.pop("_fastpipe_report", None)
    if _fp_report:
        (st.error if "失敗" in _fp_report else st.success)(_fp_report)

    _fp_mode = st.segmented_control(
        "批次模式",
        options=["🛡 保護模式", "♻️ 全部重來"],
        default="🛡 保護模式",
        key="_fastpipe_mode",
        help="保護模式：鎖定已入庫(🔵)檔、只處理未入庫。"
             "全部重來：整批含已入庫重新 encode + 上傳（原地覆蓋）。",
    )
    _redo_all = (_fp_mode == "♻️ 全部重來")

    _fp_non_ingested = [fk for fk in file_keys if _file_status_map.get(fk) != "ingested"]
    _fp_encoded = [fk for fk in file_keys if _file_status_map.get(fk) == "encoded"]
    if _redo_all:
        _embed_targets = list(file_keys)     # 含已入庫，全部重新 encode
        _upsert_targets = list(file_keys)    # 全部重新上傳
    else:
        _embed_targets = _fp_non_ingested    # 保護：略過已入庫
        _upsert_targets = _fp_encoded

    st.caption(
        f"模式：**{'♻️ 全部重來' if _redo_all else '🛡 保護模式'}** ｜ "
        f"待 embedding：**{len(_embed_targets)}** 檔 ｜ 待上傳：**{len(_upsert_targets)}** 檔"
    )
    st.caption(
        "⚠️ 批次會**跳過逐頁審閱關卡**，並使用**磁碟上已存檔的 md**（非未存的編輯緩衝）。"
        + ("　🔴 全部重來會覆蓋已入庫資料！" if _redo_all else "")
    )
    _fp_confirm = st.checkbox(
        "我已確認上述 md 審閱完畢，可批次處理",
        key="_fastpipe_confirm",
    )

    # ---- 文件種類一鍵標記（folder → doc_type 對照表）----
    with st.expander("📁 文件種類一鍵標記（folder → doc_type）", expanded=False):
        _all_folders = sorted({
            split_key_parts(fk)[0] for fk in file_keys if len(split_key_parts(fk)) > 1
        })
        _dtmap_fp = load_doc_type_map()
        if _dtmap_fp:
            st.caption("目前對照：" + " · ".join(
                f"`{k}`→{v}" for k, v in sorted(_dtmap_fp.items())
            ))
        else:
            st.caption("目前對照表為空（皆為「未分類」）。")
        _mk_cols = st.columns([2, 2, 1], vertical_alignment="bottom")
        with _mk_cols[0]:
            _sel_folders = st.multiselect(
                "選資料夾（可多選）", options=_all_folders, default=[],
                key="_fp_doctype_folders",
                help="一次選多個資料夾，套用同一個 doc_type",
            )
        with _mk_cols[1]:
            _dt_val = st.text_input(
                "doc_type", value="", key="_fp_doctype_value",
                placeholder="專案報告 / 教育訓練 / 法規規範 / 部門規則（清空＝移除）",
            )
        with _mk_cols[2]:
            if st.button("🏷 標記", use_container_width=True, key="_fp_doctype_apply"):
                if not _sel_folders:
                    st.warning("請先選至少一個資料夾")
                else:
                    _v = _dt_val.strip()
                    _m = load_doc_type_map()
                    for f in _sel_folders:
                        if _v:
                            _m[f] = _v
                        else:
                            _m.pop(f, None)  # 清空＝移除對照，退回未分類
                    save_doc_type_map(_m)
                    st.success(
                        f"已標記 {len(_sel_folders)} 個資料夾 → {_v or DOC_TYPE_DEFAULT}"
                    )
                    st.rerun()
        st.caption(
            "ℹ️ 只改對照表、不動向量。標記後需到下方「🧬 一鍵 embedding」+「🗄 更新資料庫」"
            "重跑，新 doc_type 才會寫進 Qdrant。"
        )

    # 模型參數沿用 Tab 4 的設定（session_state），首次未開 Tab 4 時用預設
    _fp_model = st.session_state.get("embed_model", EMBEDDING_MODEL)
    _fp_bs = int(st.session_state.get("embed_batch", 32))
    _fp_fp16 = bool(st.session_state.get("embed_fp16", True))

    fp_cols = st.columns(2)

    # ---- [一鍵 embedding] ----
    with fp_cols[0]:
        if st.button(
            "🧬 一鍵 embedding",
            type="primary",
            use_container_width=True,
            disabled=(not _fp_confirm) or (len(_embed_targets) == 0),
            help=f"對 {len(_embed_targets)} 檔批次 encode（已有有效快取者自動跳過）",
            key="_fastpipe_embed_btn",
        ):
            try:
                _embedder = load_embedder(_fp_model, _fp_fp16, st.session_state.get("embed_device", "cpu"))
            except Exception as e:
                st.session_state["_fastpipe_report"] = f"⚡ 一鍵 embedding 失敗：模型載入錯誤 {e}"
                st.rerun()
            n_enc = n_cache = n_empty = n_nohash = 0
            fails: list[str] = []
            n = len(_embed_targets)
            prog = st.progress(0.0, text="準備中...")
            for fi, fk in enumerate(_embed_targets):
                prog.progress(fi / n, text=f"[{fi+1}/{n}] {Path(fk).name}")
                try:
                    chunks, sc = build_chunks_from_disk(fk, image_root, data_path)
                    ids = [c["id"] for c in chunks]
                    if not ids:
                        n_empty += 1
                        continue
                    if not sc.get("file_hash"):
                        n_nohash += 1  # 空 hash 會導致跨檔 point-id 碰撞，跳過
                        fails.append(f"{Path(fk).name}: file_hash 為空（原始檔找不到？）")
                        continue
                    if vector_cache_status(fk, sc, ids)["valid"]:
                        n_cache += 1
                        # 已有有效快取 = 已 embedding；非 encoded 就統一標成 encoded，
                        # 讓「更新資料庫」能接手上傳（processing/unprocessed/ingested 都收斂過來）
                        if _file_status_map.get(fk) != "encoded":
                            set_status_on_disk(fk, sc, "encoded")
                        continue
                    texts = [c["payload"]["content"]["text_with_prefix"] for c in chunks]
                    dparts, sall = [], []
                    for s in range(0, len(texts), _fp_bs):
                        d, sp = encode_chunks(_embedder, texts[s:s + _fp_bs], batch_size=_fp_bs)
                        dparts.append(d)
                        sall.extend(sp)
                    save_vectors(fk, sc, ids, np.concatenate(dparts, axis=0), sall, fp16=_fp_fp16)
                    set_status_on_disk(fk, sc, "encoded")
                    n_enc += 1
                except Exception as e:
                    fails.append(f"{Path(fk).name}: {e}")
            prog.empty()
            rep = (
                f"⚡ 一鍵 embedding 完成：新 encode **{n_enc}** · 已快取跳過 **{n_cache}** · "
                f"空檔 {n_empty} · 無 hash 跳過 {n_nohash}"
            )
            if fails:
                rep += f"\n\n失敗 {len(fails)} 檔：\n- " + "\n- ".join(fails[:10])
                if len(fails) > 10:
                    rep += f"\n- …另 {len(fails) - 10} 檔"
            st.session_state["_fastpipe_report"] = rep
            st.rerun()

    # ---- [更新資料庫] ----
    with fp_cols[1]:
        if st.button(
            "🗄 更新資料庫",
            use_container_width=True,
            disabled=(not _fp_confirm) or (len(_upsert_targets) == 0) or (not HAS_QDRANT),
            help=f"把 {len(_upsert_targets)} 檔上傳到 Qdrant collection `{qdrant_collection_name}`",
            key="_fastpipe_upsert_btn",
        ):
            try:
                _qcli = get_qdrant_client(qdrant_url, qdrant_api_key)
                ensure_text_collection(_qcli, qdrant_collection_name, recreate=False)
            except Exception as e:
                st.session_state["_fastpipe_report"] = f"⚡ 更新資料庫失敗：Qdrant 連線/建表錯誤 {e}"
                st.rerun()
            n_up = n_skip_invalid = n_mismatch = n_empty = 0
            n_pts = 0
            fails = []
            n = len(_upsert_targets)
            prog = st.progress(0.0, text="準備中...")
            for fi, fk in enumerate(_upsert_targets):
                prog.progress(fi / n, text=f"[{fi+1}/{n}] {Path(fk).name}")
                try:
                    chunks, sc = build_chunks_from_disk(fk, image_root, data_path)
                    ids = [c["id"] for c in chunks]
                    if not ids:
                        n_empty += 1
                        continue
                    if not sc.get("file_hash"):
                        n_mismatch += 1  # 空 hash → 跨檔 point-id 碰撞風險，拒絕上傳
                        fails.append(f"{Path(fk).name}: file_hash 為空，拒絕上傳（避免覆蓋他檔）")
                        continue
                    if not vector_cache_status(fk, sc, ids)["valid"]:
                        n_skip_invalid += 1
                        fails.append(f"{Path(fk).name}: 快取失效，需重新 encode")
                        continue
                    dense, sparse, manifest = load_cached_vectors(fk)
                    if manifest.get("chunk_ids") != ids:
                        n_mismatch += 1
                        fails.append(f"{Path(fk).name}: 快取 chunk_ids 與當前 chunks 不一致")
                        continue
                    points = build_points_for_upsert(chunks, dense, sparse, now_iso())
                    upsert_in_batches(
                        _qcli, qdrant_collection_name, points, batch_size=QDRANT_BATCH_SIZE,
                    )
                    set_status_on_disk(fk, sc, "ingested")
                    n_up += 1
                    n_pts += len(points)
                except Exception as e:
                    fails.append(f"{Path(fk).name}: {e}")
            prog.empty()
            rep = (
                f"⚡ 更新資料庫完成：上傳 **{n_up}** 檔 / {n_pts} points → 🔵 已入庫 · "
                f"快取失效跳過 {n_skip_invalid} · 不一致 {n_mismatch} · 空檔 {n_empty}"
            )
            if fails:
                rep += f"\n\n問題 {len(fails)} 檔：\n- " + "\n- ".join(fails[:10])
                if len(fails) > 10:
                    rep += f"\n- …另 {len(fails) - 10} 檔"
            st.session_state["_fastpipe_report"] = rep
            st.rerun()


with st.sidebar:
    st.header("設定")
    data_path_str = st.text_input(
        "原始資料根目錄（DATA_ROOT）", DATA_ROOT_ENV or DEFAULT_DATA_PATH,
        help="file_key 的相對基底；預設讀 .env 的 DATA_ROOT。完整路徑 = 此根 / 文件種類資料夾 / …"
    )
    data_path = Path(data_path_str)

    # .env DATA_DOC_TYPE 下拉：選文件種類資料夾 → 清單與 fast pipeline 只含「DATA_ROOT/<此資料夾>」
    # 底下、且實體存在的檔（file_key 首段 == 此資料夾）。對應不到目錄的檔不載入。
    selected_doc_type = ""
    if DATA_DOC_TYPES:
        selected_doc_type = st.selectbox(
            "文件種類資料夾（DATA_DOC_TYPE）",
            options=DATA_DOC_TYPES,
            key="_review_doc_type_folder",
            help="完整路徑 = 原始資料根目錄 / 此資料夾。只處理此資料夾底下、且實體存在的檔。",
        )
        st.caption(
            f"📂 作用文件種類（相對）：`{selected_doc_type}/…` "
            f"· 比對只看 file_key 首段，與絕對根無關 · 絕對根僅供 PDF 預覽拼路徑"
        )

    image_root_str = st.text_input(
        "圖片根目錄", str(MKDATA_PATH),
        help="包含全部 `{檔名}_image/` 子資料夾的目錄。"
             "工具會用『檔名 + md 中圖片檔名』自動對應實體位置，"
             "不依賴 md 內 ![]() 的相對路徑"
    )
    image_root = Path(image_root_str)

    auto_soffice = find_soffice()
    soffice_override = st.text_input(
        "LibreOffice 路徑（PPT/PPTX 預覽用）",
        value=auto_soffice or "",
        help="自動偵測；若未安裝可留空（PPT 將無法 PDF 預覽，但 md 編輯仍可用）。"
             "或手動指定 soffice.exe 完整路徑。",
    )
    soffice_path = find_soffice(soffice_override.strip() or None)

    if not TRACKER.exists():
        st.error(f"找不到 {TRACKER}")
        st.stop()
    tracker = load_tracker()
    file_keys = sorted(tracker.keys())

    # 清單來源：直接掃「原始資料根目錄 / <選定文件種類>」實體目錄，只留 mkdata 已處理（有 {stem}.md）的檔。
    # 以實體檔為準 → 每檔只出現一次，避免 tracker 殘留 / 跨資料夾重複 key 造成清單重複。
    # file_key = 相對原始資料根目錄的路徑（文件種類/建案/檔）；絕對根（D: 或 ./RAG_raw_data）只在此拼實體路徑用。
    if selected_doc_type:
        file_keys = list_processed_files_in_dir(
            str(data_path / selected_doc_type), str(data_path), str(MKDATA_PATH),
        )
        if not file_keys:
            st.caption(
                f"ℹ️ 「{selected_doc_type}」目錄底下找不到已處理（mkdata 有 .md）的檔。"
                f"確認原始資料根目錄（{data_path}）可存取、且該類別已產生 .md。"
            )

    # 清單為空（disk-scan 無結果 / tracker 空）→ 停在這，避免 selectbox 回 None 後續崩。
    # 上方「原始資料根目錄」「文件種類」下拉已渲染、仍可調整 → rerun 後可恢復。
    if not file_keys:
        st.warning("目前清單為空（無檔可審閱）。請調整上方原始資料根目錄／文件種類，或先產生 .md。")
        st.stop()

    # 每個檔案的 review_status（從 sidecar 直接讀 disk）
    _file_status_map: dict[str, str] = {
        fk: get_file_status_quick(fk) for fk in file_keys
    }
    _status_counts: dict[str, int] = {s: 0 for s in REVIEW_STATUSES}
    for s in _file_status_map.values():
        _status_counts[s] = _status_counts.get(s, 0) + 1

    # 治本：selectbox 的選擇只靠 keyed widget 狀態記憶；在高頻 rerun（如 ace 編輯）下，
    # 該狀態有機會被 Streamlit 丟失而 fallback 到 options[0]（字母序第一個檔＝別的專案），
    # 進而觸發下方 loaded_file 重載、把畫面跳到其他專案。這裡在渲染前先校正 key：若值遺失
    # 或不在清單，還原成「目前正在編輯的檔」(loaded_file)，否則退回第一個 → 杜絕靜默跳檔。
    if file_keys:
        _cur_sel = st.session_state.get("sidebar_file_selectbox")
        if _cur_sel not in file_keys:
            _restore = st.session_state.get("loaded_file")
            st.session_state["sidebar_file_selectbox"] = (
                _restore if _restore in file_keys else file_keys[0]
            )

    selected_key = st.selectbox(
        f"檔案 ({len(file_keys)} 個)",
        file_keys,
        format_func=lambda fk: (
            f"{STATUS_BADGE[_file_status_map[fk]]['emoji']} "
            f"[{STATUS_BADGE[_file_status_map[fk]]['label']}] {fk}"
        ),
        help="前綴 emoji 表示處理狀態，色票對應在下方圖例。"
             "🔴未處理 / 🟡處理中 / 🟢處理完（已 encode）/ 🔵已寫入庫（已 upsert 到 Qdrant）",
        key="sidebar_file_selectbox",
    )

    # 顏色圖例 + 各狀態檔案數
    legend_chips = "".join(
        f"<span style='display:inline-block; padding:3px 8px; margin:2px;"
        f" border-radius:4px; background:{v['bg']}; color:{v['fg']};"
        f" font-size:11px; font-weight:bold;'>"
        f"{v['emoji']} {v['label']}：{_status_counts.get(k, 0)}"
        f"</span>"
        for k, v in STATUS_BADGE.items()
    )
    st.markdown(legend_chips, unsafe_allow_html=True)

    dpi = st.slider(
        "PDF 渲染 DPI", 60, 400, 110, step=10,
        help="預覽清晰度。右鍵「在新分頁中開啟圖片」放大看小字時，"
             "清晰度就是這個 DPI；CAD 細字建議 250–400。DPI 越高越清楚但越慢。",
    )

    st.divider()
    st.subheader("圖片校對顯示")
    img_cols = st.slider("每列圖片數", 1, 5, 3)
    img_width = st.slider("縮圖寬 (px)", 120, 480, 240, step=20)

    st.divider()
    st.subheader("Qdrant 連線")
    qdrant_url = st.text_input(
        "Qdrant URL", value=QDRANT_DEFAULT_URL,
        help="Docker server endpoint，例如 http://localhost:6333",
    )
    qdrant_api_key = st.text_input(
        "API key（選填）", value="", type="password",
        help="本地 Docker 預設沒有 API key；server 模式如有開驗證再填",
    )
    qdrant_collection_name = st.text_input(
        "Collection 名稱", value=QDRANT_TEXT_COLLECTION,
        help="預設對齊 qdrant格式.md §2。改動代表寫到別的 collection",
    )

    # ============================================================
    # fast pipeline：全庫批次 encode → Qdrant（密碼鎖 + 模式切換）
    # ============================================================
    st.divider()
    st.subheader("⚡ fast pipeline")
    # 密碼鎖：與審閱模式同一組密碼；解鎖後才看得到選項與功能（批次寫入多一道閘）。
    if not st.session_state.get("_fastpipe_unlocked", False):
        st.caption("🔒 批次寫入操作，需密碼解鎖（與審閱模式相同）才能看到選項與使用。")
        with st.form("_fastpipe_unlock_form"):
            _fp_pw = st.text_input("密碼", type="password", key="_fastpipe_pw")
            if st.form_submit_button("解鎖 fast pipeline"):
                if _fp_pw == REVIEW_PASSWORD:
                    st.session_state["_fastpipe_unlocked"] = True
                    st.rerun()
                else:
                    st.error("密碼錯誤")
    else:
        _fp_lock_cols = st.columns([3, 1])
        _fp_lock_cols[0].caption("🔓 已解鎖")
        if _fp_lock_cols[1].button("🔒 鎖定", key="_fastpipe_lock_btn", use_container_width=True):
            st.session_state["_fastpipe_unlocked"] = False
            st.rerun()
        # 🧭 路徑修正：放在 fast pipeline 之前（同一道密碼鎖內，因為也會寫 Qdrant）
        _render_path_correction(
            data_path, selected_doc_type,
            qdrant_url, qdrant_api_key, qdrant_collection_name,
        )
        _render_fast_pipeline(
            file_keys, _file_status_map, image_root, data_path,
            qdrant_url, qdrant_api_key, qdrant_collection_name,
        )

    st.divider()
    st.caption("提示：編輯區失焦（Tab/點擊外部）後才會自動同步到完整 md。"
               "刪除圖片會搬到 ./_md_trash/，可在「圖片校對」分頁復原。")

# === 載入選中檔案 ===
md_path = derive_md_path(selected_key)
if not md_path.exists():
    st.error(f"找不到對應 Markdown: {md_path}")
    st.stop()

if st.session_state.get("loaded_file") != selected_key:
    st.session_state.current_md = md_path.read_text(encoding="utf-8")
    st.session_state.loaded_file = selected_key
    st.session_state.page_idx = 0

    # 載入 sidecar 並 hydrate session_state
    sidecar = load_sidecar(selected_key)
    st.session_state.sidecars[selected_key] = sidecar
    # custom_labels.pages：sidecar 用 str key（JSON 限制），記憶體用 int key
    sidecar_pages = sidecar["custom_labels"].get("pages", {})
    pages_dict = {int(k): list(v) for k, v in sidecar_pages.items()}
    st.session_state.custom_labels[selected_key] = {
        "document": list(sidecar["custom_labels"].get("document", [])),
        "pages": pages_dict,
    }
    st.session_state.split_settings[selected_key] = dict(sidecar["split_settings"])
    st.session_state.delete_history[selected_key] = list(sidecar["delete_history"])

    # 第一次載入時 lazy 計算 file_hash（從原始 PDF/PPT 算）
    if not sidecar.get("file_hash"):
        source_path = data_path / selected_key
        sidecar["file_hash"] = compute_file_hash(source_path)
        # 不立刻寫盤，等下次 mutation 一起 persist；除非已經是空白 sidecar 才寫
        if not derive_sidecar_path(selected_key).exists():
            save_sidecar(selected_key, sidecar)

    bump_widget_version()

# 檢索 Tab 跳回用的延遲套用：等檔案載入完才能套 page_idx
_pending_page = st.session_state.pop("_pending_page_idx", None)
if _pending_page is not None:
    st.session_state.page_idx = max(0, int(_pending_page))

# 把上一個 rerun 編輯器的最新內容 commit 進 current_md（吸收 ace / text_area 輸入）
commit_editor_if_dirty(selected_key, st.session_state.page_idx)

fm, header_section, pages = parse_md(st.session_state.current_md)
if not pages:
    st.error("解析不到任何 `## 第 N 頁` 區塊")
    st.stop()

# === 頁面導覽列 ===
# ⚠️ 遠端（有網路延遲）使用者：ace 編輯器 auto_update=True 是「失焦才回傳」，點按鈕會先觸發
# 失焦、再觸發點擊。舊寫法用「瞬時 st.button() 回傳值 + st.rerun()」判斷換頁，延遲下按鈕的
# 瞬時 True 會被失焦那次 rerun 吃掉 → 遠端「打字後無法換頁」（本機 0 延遲則無感）。
# 改用 on_click / on_change callback：點擊與失焦值會在同一次 rerun 被處理，callback 保證執行。
# callback 在 script body 之前跑 → 換頁/存檔一律「先 commit 當前頁編輯」再移動，避免編輯寫錯頁。
def _nav_commit_current():
    """把當前頁 ace/text_area 的最新值吸收進 current_md（callback 內先跑，確保不遺失編輯）。"""
    fk = st.session_state.get("loaded_file")
    if fk:
        commit_editor_if_dirty(fk, st.session_state.page_idx)

def _nav_page_count() -> int:
    _, _, _pgs = parse_md(st.session_state.current_md)
    return len(_pgs)

def _nav_go(delta: int):
    _nav_commit_current()
    total = _nav_page_count()
    st.session_state.page_idx = max(0, min(total - 1, st.session_state.page_idx + delta))

def _nav_jump():
    _nav_commit_current()
    total = _nav_page_count()
    target = int(st.session_state.get("nav_jump_page", 1))
    st.session_state.page_idx = max(0, min(total - 1, target - 1))

def _do_save():
    fk = st.session_state.get("loaded_file")
    if not fk:
        return
    _nav_commit_current()  # 存檔前先吸收當前頁編輯（含尚未 commit 的最後一次打字）
    mp = derive_md_path(fk)
    mp.write_text(st.session_state.current_md, encoding="utf-8")
    mark_processing(fk)
    # 儲存＝提交刪除：關閉圖片復原視窗（垃圾桶檔案自此視為永久刪除）
    st.session_state.delete_history[fk] = []
    # 磁碟已動：捨棄編輯無法再無痛還原 prior_status（會留在 processing）
    sc = st.session_state.sidecars.get(fk)
    if sc is not None:
        sc["disk_dirty"] = True
    persist_review_state(fk)
    # 換 editor key → 強制 ace 重新掛載（同換頁/捨棄）。避免 auto_update=True 在同一個未重掛的
    # ace 上反覆回傳值，與頂層 commit_editor_if_dirty 比對形成 rerun 迴圈（遠端表現為存檔後卡住）。
    bump_widget_version()
    st.session_state["_review_toast"] = f"已儲存 → {mp.name}"

def _do_discard():
    fk = st.session_state.get("loaded_file")
    if not fk:
        return
    # 先把軟刪除的圖檔從垃圾桶搬回原位，否則 reload 回來的 md 引用會變孤兒
    restore_trashed_files_for(fk)
    st.session_state.current_md = derive_md_path(fk).read_text(encoding="utf-8")
    st.session_state.delete_history[fk] = []
    # 還原：若進入 processing 前是 encoded/ingested 且磁碟未被儲存，回去原狀態
    sc = st.session_state.sidecars.get(fk)
    if sc is not None:
        prior = sc.get("prior_status")
        dirty = sc.get("disk_dirty", False)
        if prior and not dirty:
            sc["review_status"] = prior
            sc.pop("prior_status", None)
            sc.pop("disk_dirty", None)
    persist_review_state(fk)
    # bump 版本：換 editor key → 丟棄未 commit 的編輯（頂層 commit 會撈不到舊 key 而 no-op）
    bump_widget_version()

# 跳頁輸入框的顯示值跟著 page_idx 走（widget 建立前設定 session_state 是允許的；勿再傳 value=）
st.session_state.nav_jump_page = st.session_state.page_idx + 1

nav_cols = st.columns([1, 1, 2, 1, 1])
with nav_cols[0]:
    st.button("⟵ 上一頁", use_container_width=True,
              disabled=st.session_state.page_idx <= 0,
              on_click=_nav_go, args=(-1,))
with nav_cols[1]:
    st.button("下一頁 ⟶", use_container_width=True,
              disabled=st.session_state.page_idx >= len(pages) - 1,
              on_click=_nav_go, args=(1,))
with nav_cols[2]:
    st.number_input(
        "跳到頁碼", min_value=1, max_value=len(pages),
        key="nav_jump_page", label_visibility="collapsed",
        on_change=_nav_jump,
    )
with nav_cols[3]:
    st.button("儲存 .md", type="primary", use_container_width=True, on_click=_do_save)
with nav_cols[4]:
    st.button("捨棄編輯", use_container_width=True, on_click=_do_discard)

# 存檔成功訊息（callback 內無法直接 st.success，改用 session flag 於 render 時顯示）
_toast = st.session_state.pop("_review_toast", None)
if _toast:
    st.success(_toast)

current_idx = st.session_state.page_idx
page_num, page_md = pages[current_idx]

parts = split_key_parts(selected_key)
project_name = derive_project_name(selected_key)  # 建案 = file_key[1]（類別在 [0]）
file_name = parts[-1]
sidecar = st.session_state.sidecars[selected_key]
file_hash_display = sidecar.get("file_hash") or fm.get("file_hash", "—") or "—"
st.markdown(
    f"**建案**：`{project_name}` ｜ **檔名**：`{file_name}` ｜ "
    f"**第 {page_num} / {len(pages)} 頁** ｜ **file_hash**：`{file_hash_display[:12] if file_hash_display != '—' else '—'}`"
)

# === Review 狀態列 + 文字暫存區 ===
_status = sidecar.get("review_status", "unprocessed")
_badge = STATUS_BADGE.get(_status, STATUS_BADGE["unprocessed"])
badge_html = (
    f"<div style='display:inline-block; padding:8px 18px; border-radius:8px; "
    f"background-color:{_badge['bg']}; color:{_badge['fg']}; "
    f"font-weight:bold; font-size:1.1em;'>"
    f"{_badge['emoji']} {_badge['label']}"
    f"</div>"
)
status_cols = st.columns([1, 1, 3])
with status_cols[0]:
    st.markdown(
        "**狀態**",
        help="自動轉換：開檔=🔴未處理；任何編輯=🟡處理中；最後一頁 Qdrant 上傳成功=🟢已完成",
    )
    st.markdown(badge_html, unsafe_allow_html=True)
with status_cols[1]:
    st.markdown("**完成時間**")
    st.markdown(f"`{sidecar.get('reviewed_at') or '— (尚未完成)'}`")
with status_cols[2]:
    st.markdown(
        "**📋 文字暫存區**",
        help="per-file 暫存；高度依內容即時自動調整（3-20 行範圍）。"
             "僅當前文件 session 內有效（不寫盤、不跨檔，但跨頁保留）。",
    )
    scratch_key = f"scratch_text_{selected_key}"
    if HAS_ACE:
        # 用 st_ace 取代 text_area —— 高度按內容 auto-grow（min_lines~max_lines）、
        # 不需 Ctrl+Enter；session_state[key] 跨 rerun/跨頁穩定持久
        st_ace(
            language="plain_text",
            theme="chrome",
            font_size=13,
            show_gutter=False,
            wrap=True,
            auto_update=True,
            min_lines=3,
            max_lines=20,
            key=scratch_key,
        )
    else:
        # fallback：text_area 無 auto-grow，固定高度
        st.text_area(
            "scratch",
            height=250,
            placeholder="貼上多行文字（請安裝 streamlit-ace 以獲得 auto-grow）...",
            key=scratch_key,
            label_visibility="collapsed",
        )

# === 五分頁主區 ===
tab_pdf, tab_img, tab_meta, tab_embed, tab_qdrant, tab_search = st.tabs(
    ["1. PDF比對", "2. 圖片校對", "3. 標籤預覽", "4. Embedding", "5. Qdrant 寫入", "6. 檢索測試"]
)

# --- 分頁 1：PDF 比對 ---
with tab_pdf:
    col_pdf, col_md = st.columns(2)

    with col_pdf:
        st.subheader("PDF 原頁")
        source_path = data_path / selected_key
        ext = source_path.suffix.lower()
        if not source_path.exists():
            st.warning(f"找不到原始檔：{source_path}")
        elif ext in (".ppt", ".pptx") and not soffice_path:
            st.warning(
                "此檔為 PPT/PPTX，需 LibreOffice 才能預覽 PDF。"
                "請安裝 LibreOffice 或在左側 sidebar 指定 soffice.exe 路徑。"
                "（Markdown 編輯與圖片校對不受影響）"
            )
        else:
            try:
                pdf_path = ensure_pdf(source_path, soffice_path)
                png_bytes = render_pdf_page_png(pdf_path, page_num - 1, dpi=dpi)
                st.image(png_bytes, use_container_width=True)

                # 預覽框右上角內建 fullscreen 是 fit-to-viewport（放大網頁只會重新縮圖），
                # 看不到小字。「開啟原圖」用真實靜態 URL 在新分頁開整頁原圖 → 瀏覽器原生縮放。
                open_col, dl_col = st.columns(2)
                with open_col:
                    st.link_button(
                        "🔍 開啟原圖",
                        high_dpi_preview_url(pdf_path, page_num - 1, dpi),
                        use_container_width=True,
                        help="在新分頁開啟本頁完整解析度圖，可用瀏覽器原生縮放自由放大。"
                             "清晰度＝左側「PDF 渲染 DPI」，字太小就調高再開。",
                    )
                with dl_col:
                    st.download_button(
                        "⬇️ 下載原檔",
                        data=source_path.read_bytes(),
                        file_name=source_path.name,
                        mime=_DOWNLOAD_MIME.get(ext, "application/octet-stream"),
                        use_container_width=True,
                        key=f"dlsrc_{selected_key}_{page_num}",
                        help="下載原始檔（PDF/PPT/PPTX），用本機看圖／閱讀器開啟。",
                    )
            except Exception as e:
                st.error(f"PDF 渲染失敗：\n```\n{e}\n```")

    with col_md:
        st.subheader("Markdown 內容（可編輯）")
        # 頁標頭 `## 第 N 頁` 與頁尾分隔線 `---` 是結構錨點（Citation／頁面對映用），
        # 不放進編輯器以免被誤改；commit 時由 commit_editor_if_dirty 自動黏回。
        st.info(
            f"本頁錨點 `## 第 {page_num} 頁` 已鎖定並自動保留，請從 `###` 標題開始編輯內文。",
            icon="📌",
        )
        editor_body = strip_page_header(page_md)
        editor_key = editor_key_for(selected_key, current_idx, st.session_state.widget_version)
        if HAS_ACE:
            st_ace(
                value=editor_body,
                language="markdown",
                theme="chrome",
                keybinding="vscode",      # Alt+↑/↓ 移行、Ctrl+Alt+↑/↓ 多游標等整套
                font_size=14,
                tab_size=2,
                show_gutter=True,
                wrap=True,
                auto_update=True,          # 失焦立刻回 value，commit_editor_if_dirty 下次 rerun 撈
                min_lines=30,
                key=editor_key,
            )
        else:
            st.warning(
                "未安裝 `streamlit-ace`，暫用基本編輯器（無 Alt+↑↓ / 多游標）。"
                "安裝後可獲 VSCode 級編輯：`pip install streamlit-ace`"
            )
            st.text_area(
                "Page Markdown",
                value=editor_body,
                height=600,
                key=editor_key,
                label_visibility="collapsed",
            )

# --- 分頁 2：圖片校對 ---
with tab_img:
    st.subheader("頁面圖片")
    _img_history = st.session_state.delete_history.get(selected_key, [])
    if _img_history:
        restore_cols = st.columns([2, 3])
        with restore_cols[0]:
            if st.button(
                f"↩ 復原全部已刪除圖片（{len(_img_history)}）",
                key=f"restore_all_v{st.session_state.widget_version}",
                use_container_width=True,
            ):
                restore_all_images_action(selected_key)
                persist_review_state(selected_key)
                st.rerun()
        with restore_cols[1]:
            st.caption("復原＝把垃圾桶圖檔搬回 + 重新插入 md 引用。"
                       "**按下「儲存 .md」後刪除即提交，復原視窗關閉**。")
    refs = parse_images_in_page(page_md)
    if not refs:
        st.info("本頁無圖片引用。")
    else:
        for row_start in range(0, len(refs), img_cols):
            row_refs = refs[row_start: row_start + img_cols]
            cols = st.columns(img_cols)
            for i, ref in enumerate(row_refs):
                with cols[i]:
                    abs_img = resolve_image_path(selected_key, ref["md_path"], image_root)
                    caption = Path(ref["md_path"]).name
                    if abs_img.exists():
                        try:
                            st.image(str(abs_img), caption=caption, width=img_width)
                        except Exception as e:
                            st.warning(f"無法顯示：{e}")
                    else:
                        st.warning(f"檔案不存在：{abs_img}")
                    st.caption(f"md 引用：`{ref['md_path']}`")
                    if st.button(
                        "🗑️ 刪除",
                        key=f"del_{current_idx}_{row_start + i}_v{st.session_state.widget_version}",
                        use_container_width=True,
                    ):
                        delete_image_action(selected_key, current_idx, ref, image_root)
                        st.rerun()

    history = st.session_state.delete_history.get(selected_key, [])
    if history:
        st.divider()
        with st.expander(f"近期刪除（{len(history)} 筆，可復原）", expanded=False):
            for j, entry in enumerate(history):
                cols = st.columns([3, 2, 1])
                cols[0].markdown(
                    f"P{entry['page_num']}：`{Path(entry['md_path']).name}`"
                )
                if entry["trash_path"]:
                    cols[1].caption("已搬到垃圾桶")
                elif entry["used_elsewhere"]:
                    cols[1].caption("檔案被他頁引用，未搬移")
                else:
                    cols[1].caption("僅移除引用")
                if cols[2].button("↩ 復原", key=f"restore_{j}_v{st.session_state.widget_version}"):
                    restore_image_action(selected_key, j)
                    st.rerun()

# --- 分頁 3：標籤預覽 ---
with tab_meta:
    # 取最新內容 + 取得當前 file 的 label/split state
    # 安全網：text_area 在 tab 切換時可能未觸發 on_change，所以優先讀 widget 的即時值
    fm_now, _, pages_now = parse_md(st.session_state.current_md)
    editor_key_current = editor_key_for(
        selected_key, current_idx, st.session_state.widget_version
    )
    edited_now = st.session_state.get(editor_key_current)
    # ace 在首次掛載時可能回 None；只在拿到字串時才採用 widget 值
    if isinstance(edited_now, str):
        page_md_now = edited_now
    else:
        _, page_md_now = pages_now[current_idx]

    file_labels = st.session_state.custom_labels.setdefault(
        selected_key, {"document": [], "pages": {}}
    )
    doc_labels = file_labels["document"]
    page_labels = file_labels["pages"].setdefault(current_idx, [])

    split_cfg = st.session_state.split_settings.setdefault(
        selected_key, {"mode": "delimiter", "delim": "\\n"}
    )

    h1_value = extract_h1(st.session_state.current_md, Path(file_name).stem)
    h2_value = f"第 {page_num} 頁"
    sections = split_page_by_headings(page_md_now)

    # 收集本頁所有出現過的 heading（按 level 分群、去重保序）
    heading_pool: dict[int, list[str]] = {}
    for sec in sections:
        for lvl, title in sec["headings"].items():
            bucket = heading_pool.setdefault(lvl, [])
            if title not in bucket:
                bucket.append(title)
    total_h_count = sum(len(v) for v in heading_pool.values())

    img_labels = [
        (derive_image_label_key(s["headings"], s["body"]), s["image_paths"])
        for s in sections if s["image_paths"]
    ]

    # callback（doc_type / 加入標籤）設定的訊息，於 render 時顯示（callback 內無法直接 st.success）
    _mt = st.session_state.pop("_meta_toast", None)
    if _mt:
        (st.success if _mt[0] == "ok" else st.warning)(_mt[1])

    # === 區塊 0：文件種類（doc_type）—— 資料夾層級，套用到同資料夾所有檔案 ===
    st.markdown("### 文件種類（doc_type）")
    _dt_parts = split_key_parts(selected_key)
    _dt_folder = _dt_parts[0] if len(_dt_parts) > 1 else ""
    _dt_map = load_doc_type_map()
    _cur_dt = resolve_doc_type(selected_key, load_sidecar(selected_key), _dt_map)
    if _dt_folder:
        st.caption(
            f"資料夾 `{_dt_folder}` 目前 doc_type：**{_cur_dt}**"
            f" · 套用到此資料夾下所有檔案 · 寫入 `{DOC_TYPE_MAP_PATH.name}`"
            f"（優先序：sidecar 逐檔覆寫 > 資料夾對照表 > 未分類）"
        )
        _dt_cols = st.columns([3, 1], vertical_alignment="bottom")
        with _dt_cols[0]:
            _dt_new = st.text_input(
                "doc_type 值",
                value=_dt_map.get(_dt_folder, ""),
                key=f"_doctype_input_{_dt_folder}",
                placeholder="例如：專案報告 / 教育訓練 / 法規規範 / 部門規則（清空＝退回未分類）",
                label_visibility="collapsed",
            )
        with _dt_cols[1]:
            # 遠端修法：on_click callback + 從 session_state 讀輸入值（非 widget 回傳值），
            # 避免延遲下「點擊被失焦吞掉」或「誤讀空值 → 意外清掉整個資料夾的 doc_type 對照」。
            def _apply_doctype(folder: str, input_key: str):
                val = str(st.session_state.get(input_key, "")).strip()
                m = load_doc_type_map()
                if val:
                    m[folder] = val
                else:
                    m.pop(folder, None)  # 清空＝移除對照，退回未分類
                save_doc_type_map(m)
                st.session_state["_meta_toast"] = (
                    "ok",
                    f"已設定 `{folder}` → {val or DOC_TYPE_DEFAULT}"
                    "（只改對照表、不動向量；舊資料需重新 encode/ingest 才帶上新值）",
                )
            st.button(
                "套用到此資料夾", use_container_width=True, key="_doctype_apply",
                on_click=_apply_doctype,
                args=(_dt_folder, f"_doctype_input_{_dt_folder}"),
            )
    else:
        st.caption("此檔不在子資料夾內（file_key 無上層資料夾）→ doc_type 為「未分類」。")
    st.divider()

    # === 區塊 A：自動 heading + 圖片 標籤 ===
    st.markdown("### Heading & 圖片自動標籤")
    st.caption(
        f"🔵 heading（h1/h2 + h3-h6 任意階） · "
        f"🖼️ 圖片（key=最深 heading／單行內容／image） · "
        f"本頁 {total_h_count} 個子標題、{len(img_labels)} 個圖片標籤"
    )
    head_chips: list[tuple[str, str, str]] = [
        ("🔵", "h1", h1_value), ("🔵", "h2", h2_value),
    ]
    for lvl in sorted(heading_pool.keys()):
        titles = heading_pool[lvl]
        for hi, t in enumerate(titles):
            key = f"h{lvl}#{hi + 1}" if len(titles) > 1 else f"h{lvl}"
            head_chips.append(("🔵", key, t))
    for label_key, paths in img_labels:
        head_chips.append(("🖼️", label_key, "\n".join(paths)))

    per_row = 6
    for r_start in range(0, len(head_chips), per_row):
        row = head_chips[r_start: r_start + per_row]
        head_cols = st.columns(per_row)
        for i, (em, k, v) in enumerate(row):
            with head_cols[i]:
                with st.popover(f"{em} {k}", use_container_width=True):
                    st.markdown(f"**{k}**")
                    st.text(v)

    st.divider()

    # === 區塊 B：自訂標籤 ===
    st.markdown("### 自訂標籤")
    st.caption("🟢 整份文件 / 🟡 僅當頁。點 chip 可看內容並刪除。")

    custom_chips = (
        [("doc", i, lab) for i, lab in enumerate(doc_labels)]
        + [("page", i, lab) for i, lab in enumerate(page_labels)]
    )
    if custom_chips:
        per_row = 6
        for row_start in range(0, len(custom_chips), per_row):
            row = custom_chips[row_start: row_start + per_row]
            cols = st.columns(per_row)
            for ci, (scope, li, lab) in enumerate(row):
                emoji = "🟢" if scope == "doc" else "🟡"
                with cols[ci]:
                    with st.popover(f"{emoji} {lab['key']}", use_container_width=True):
                        scope_text = "整份文件" if scope == "doc" else f"僅 P{page_num}"
                        st.markdown(f"**{lab['key']}** ({scope_text})")
                        st.text(lab["value"])
                        btn_key = f"rmlbl_{scope}_{li}_{current_idx}_v{st.session_state.widget_version}"
                        if st.button("🗑️ 刪除此標籤", key=btn_key):
                            target = doc_labels if scope == "doc" else page_labels
                            target.pop(li)
                            mark_processing(selected_key)
                            bump_widget_version()
                            st.rerun()
    else:
        st.caption("（尚無自訂標籤）")

    with st.expander("➕ 新增標籤", expanded=False):
        nk_key = f"new_lbl_k_{selected_key}_{current_idx}_v{st.session_state.widget_version}"
        nv_key = f"new_lbl_v_{selected_key}_{current_idx}_v{st.session_state.widget_version}"
        ns_key = f"new_lbl_s_{selected_key}_{current_idx}_v{st.session_state.widget_version}"
        st.text_input("標籤名 (key)", key=nk_key, placeholder="例：context")
        st.text_area("內容 (value)", key=nv_key, height=80,
                     placeholder="例：璞真建設股份有限公司台北...")
        st.radio("適用範圍", ["🟢 整份文件", "🟡 僅當頁"],
                 horizontal=True, key=ns_key)

        # 遠端修法：on_click callback + 從 session_state 讀值。避免延遲下「打了字卻誤判空白」，
        # 且點擊不會被 text_area 失焦那次 rerun 吞掉。
        def _add_label(k_key: str, v_key: str, s_key: str):
            k = str(st.session_state.get(k_key, "")).strip()
            v = str(st.session_state.get(v_key, "")).strip()
            scope = st.session_state.get(s_key, "🟢 整份文件")
            if not k or not v:
                st.session_state["_meta_toast"] = ("warn", "標籤名與內容皆不可空白")
                return
            fk = st.session_state.get("loaded_file")
            idx = st.session_state.page_idx
            file_lbls = st.session_state.custom_labels.setdefault(
                fk, {"document": [], "pages": {}}
            )
            entry = {"key": k, "value": v}
            if str(scope).startswith("🟢"):
                file_lbls["document"].append(entry)
            else:
                file_lbls["pages"].setdefault(idx, []).append(entry)
            mark_processing(fk)
            bump_widget_version()  # 換 key → 清空輸入框
            st.session_state["_meta_toast"] = ("ok", f"已加入標籤 `{k}`")

        st.button("加入標籤", type="primary",
                  on_click=_add_label, args=(nk_key, nv_key, ns_key))

    st.divider()

    # === 區塊 C：切分設定 ===
    st.markdown("### 切分設定")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        mode_label = st.radio(
            "模式",
            ["整頁 1 chunk", "依分隔字元"],
            index=0 if split_cfg["mode"] == "page" else 1,
            key=f"split_mode_radio_{selected_key}",
        )
        new_mode = "page" if mode_label == "整頁 1 chunk" else "delimiter"
    with sc2:
        if new_mode == "delimiter":
            new_delim = st.text_input(
                "分隔字元（支援 `\\n` `\\t` 等 escape）",
                value=split_cfg["delim"],
                key=f"split_delim_input_{selected_key}",
            )
        else:
            new_delim = split_cfg["delim"]
            st.caption("整頁模式不需分隔字元。")
    if split_cfg["mode"] != new_mode or split_cfg["delim"] != new_delim:
        split_cfg["mode"] = new_mode
        split_cfg["delim"] = new_delim
        mark_processing(selected_key)

    st.divider()

    # === 區塊 D：切分結果 + chunks（對齊 qdrant格式.md v2.0.0 payload）===
    st.markdown("### 切分結果")
    st.caption(
        "每筆 chunk = Qdrant 一個 point。`text_with_prefix` 是真正餵 embedding 的內容；"
        "`text` 是純文字（給 LLM 上下文）。`id` 由 `uuid5(file_hash|page|section|chunk)` 算出，重 embed 自動覆蓋。"
    )

    all_chunks = build_all_chunks_for_doc(
        file_key=selected_key,
        sidecar=sidecar,
        pages=pages_now,
        page_md_override={current_idx: page_md_now},
        h1=h1_value,
        doc_labels=doc_labels,
        file_labels=file_labels,
        split_cfg=split_cfg,
        image_root=image_root,
    )
    current_chunks = [c for c in all_chunks if c["_page_pos"] == current_idx]

    if not current_chunks:
        st.info("本頁切分後無內容（可能是空頁或純圖片頁）。")

    for n, pkg in enumerate(current_chunks, start=1):
        payload = pkg["payload"]
        src = payload["metadata"]["source"]
        loc = payload["metadata"]["location"]
        vis = payload["visuals"]
        lbls = payload["labels"]
        content = payload["content"]
        chunking = payload["chunking"]

        with st.container(border=True):
            head_line = (
                f"**Chunk {n}** · {content['char_count']} 字元 · "
                f"`id={pkg['id'][:8]}…`"
            )
            if loc["headings"]:
                deepest_lvl = max(int(k) for k in loc["headings"].keys())
                head_line += f" · 來自 `{'#' * deepest_lvl} {loc['current_header']}`"
            if vis["has_image"]:
                head_line += f" · 含 {vis['image_count']} 張圖"
            st.markdown(head_line)

            chips: list[tuple[str, str, str]] = [
                ("🔵", "h1", src["doc_title"]),
                ("🔵", "h2", loc["page_label"]),
            ]
            for lvl_str in sorted(loc["headings"].keys(), key=int):
                chips.append(("🔵", f"h{lvl_str}", loc["headings"][lvl_str]))
            if vis["has_image"]:
                img_summary = "\n".join(img["local_path"] for img in vis["images"])
                chips.append(("🖼️", vis["image_label"], img_summary))
            for lab in lbls["document"]:
                chips.append(("🟢", lab["key"], lab["value"]))
            for lab in lbls["page"]:
                chips.append(("🟡", lab["key"], lab["value"]))
            chips.append(("🔗", "prev", chunking["prev_chunk_id"] or "—"))
            chips.append(("🔗", "next", chunking["next_chunk_id"] or "—"))

            cper_row = 6
            for r_start in range(0, len(chips), cper_row):
                chunk_row = chips[r_start: r_start + cper_row]
                cc = st.columns(cper_row)
                for ii, (em, k, v) in enumerate(chunk_row):
                    with cc[ii]:
                        with st.popover(f"{em} {k}", use_container_width=True):
                            st.markdown(f"**{k}**")
                            st.text(v)

            st.markdown("**text_with_prefix（embedding 輸入）：**")
            st.code(content["text_with_prefix"], language="markdown")

    def _strip_internal(c: dict) -> dict:
        return {k: v for k, v in c.items() if not k.startswith("_")}

    if current_chunks:
        st.download_button(
            "下載本頁所有 chunks (JSON)",
            data=json.dumps(
                [_strip_internal(c) for c in current_chunks],
                ensure_ascii=False, indent=2,
            ),
            file_name=f"{Path(file_name).stem}_p{page_num}_chunks.json",
            mime="application/json",
        )

    with st.expander(f"檢視整份文件所有 chunks payload（{len(all_chunks)} 筆）"):
        st.json([_strip_internal(c) for c in all_chunks])

# --- 分頁 4：Embedding ---
with tab_embed:
    st.subheader("Embedding (bge-m3 hybrid)")
    st.caption(
        "dense (1024) + learned sparse 一次推理產出。"
        "快取存 mkdata/{stem}.vectors.{dense.npy,sparse.json,manifest.json}。"
    )

    # === 設定列 ===
    cfg_cols = st.columns([2, 2, 1, 1, 2])
    with cfg_cols[0]:
        model_name = st.selectbox(
            "模型", AVAILABLE_EMBEDDERS, index=0, key="embed_model",
        )
    with cfg_cols[1]:
        batch_size = st.slider(
            "Batch size", 4, 128, 32, step=4, key="embed_batch",
            help="4090 24GB 可推到 64-128；OOM 就調小",
        )
    with cfg_cols[2]:
        use_fp16 = st.toggle(
            "FP16", value=True, key="embed_fp16",
            help="省 VRAM 與磁碟（dense 存 float16）。device=cpu 時自動忽略",
        )
    with cfg_cols[3]:
        st.selectbox(
            "裝置", ["cpu", "cuda"], index=0, key="embed_device",
            help="cpu：小顯存卡（如 2050 4GB）唯一安全選擇，bge-m3 載入~2-3s、單句 encode <1s。"
                 "cuda：需 ≳6GB 空閒顯存，4GB 卡會 OOM 讓行程直接崩潰。",
        )
    with cfg_cols[4]:
        st.caption(
            f"`pipeline={PIPELINE_VERSION}` · "
            f"`embed_ver={EMBEDDING_VERSION}` · "
            f"`chunking={CHUNKING_STRATEGY}`"
        )

    # === 重建當前文件的 chunks（與 Tab 3 同步邏輯）===
    fm_emb, _, pages_emb = parse_md(st.session_state.current_md)
    editor_key_emb = editor_key_for(
        selected_key, current_idx, st.session_state.widget_version
    )
    edited_emb = st.session_state.get(editor_key_emb)
    if isinstance(edited_emb, str):
        page_md_emb = edited_emb
    else:
        _, page_md_emb = pages_emb[current_idx]
    file_labels_emb = st.session_state.custom_labels.setdefault(
        selected_key, {"document": [], "pages": {}}
    )
    doc_labels_emb = file_labels_emb["document"]
    split_cfg_emb = st.session_state.split_settings.setdefault(
        selected_key, {"mode": "delimiter", "delim": "\\n"}
    )
    h1_emb = extract_h1(st.session_state.current_md, Path(file_name).stem)

    all_chunks_emb = build_all_chunks_for_doc(
        file_key=selected_key,
        sidecar=sidecar,
        pages=pages_emb,
        page_md_override={current_idx: page_md_emb},
        h1=h1_emb,
        doc_labels=doc_labels_emb,
        file_labels=file_labels_emb,
        split_cfg=split_cfg_emb,
        image_root=image_root,
    )
    expected_chunk_ids = [c["id"] for c in all_chunks_emb]
    is_last_page = current_idx == len(pages_emb) - 1
    last_page_num = pages_emb[-1][0] if pages_emb else None

    if not is_last_page:
        st.warning(
            f"⏭️ 「全檔批次 encode」需在最後一頁（P{last_page_num}）才能執行，"
            f"目前在 P{page_num}。請通讀全部內容後切到末頁再上傳。"
            "（本頁即時 encode 預覽 / 清除快取 不受限）"
        )

    st.divider()

    # === 區塊 1：快取狀態 ===
    st.markdown("### 快取狀態")
    cache_status = vector_cache_status(selected_key, sidecar, expected_chunk_ids)
    paths_disp = derive_vector_paths(selected_key)

    if not cache_status["exists"]:
        st.warning(
            f"無向量快取（共 {len(expected_chunk_ids)} 個 chunks 待 embed）"
        )
    elif cache_status["valid"]:
        m = cache_status["manifest"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("向量數", m.get("count", 0))
        c2.metric("dense dim", m.get("dense_dim", 0))
        c3.metric("dtype", m.get("dense_dtype", "—"))
        c4.metric(
            "dense 檔大小",
            f"{paths_disp['dense'].stat().st_size / 1024:.1f} KB"
            if paths_disp["dense"].exists() else "—",
        )
        st.success(f"✓ 快取有效 · 編碼於 {m.get('encoded_at', '—')}")
    else:
        st.error(f"⚠ 快取失效：{cache_status['reason']} → 需要重新編碼")

    # === 操作列 ===
    op_cols = st.columns([2, 2, 2])
    encode_disabled = (not expected_chunk_ids) or (not is_last_page)
    encode_help = (
        f"需先切到最後一頁（P{last_page_num}）。"
        if not is_last_page else None
    )
    with op_cols[0]:
        if st.button(
            "全檔批次 encode",
            type="primary",
            use_container_width=True,
            disabled=encode_disabled,
            help=encode_help,
            key="btn_full_encode",
        ):
            try:
                embedder = load_embedder(model_name, use_fp16, st.session_state.get("embed_device", "cpu"))
            except Exception as e:
                st.error(f"模型載入失敗：\n```\n{e}\n```")
                embedder = None

            if embedder is not None:
                texts = [
                    c["payload"]["content"]["text_with_prefix"]
                    for c in all_chunks_emb
                ]
                total = len(texts)
                progress = st.progress(0.0, text=f"Encoding 0 / {total}...")
                dense_chunks: list[np.ndarray] = []
                sparse_all: list[dict] = []
                t0 = time.time()
                try:
                    for start in range(0, total, batch_size):
                        batch = texts[start:start + batch_size]
                        d, s = encode_chunks(embedder, batch, batch_size=batch_size)
                        dense_chunks.append(d)
                        sparse_all.extend(s)
                        done = start + len(batch)
                        progress.progress(
                            done / total,
                            text=f"Encoding {done} / {total}...",
                        )
                    dense_all = np.concatenate(dense_chunks, axis=0)
                    save_vectors(
                        selected_key, sidecar,
                        expected_chunk_ids, dense_all, sparse_all,
                        fp16=use_fp16,
                    )
                    elapsed = time.time() - t0
                    progress.empty()
                    # ✓ 本地向量已建 → 狀態 🟢 encoded（綠）
                    mark_encoded(selected_key)
                    st.success(
                        f"✓ {total} 個 chunks 已 encode + cache · "
                        f"{elapsed:.1f}s · {total / elapsed:.1f} chunks/s"
                        f" · 狀態 → 🟢 處理完"
                    )
                    st.info("接下來到 Tab 5 上傳到 Qdrant，狀態才會轉為 🔵 已寫入庫。")
                    st.rerun()
                except Exception as e:
                    progress.empty()
                    st.error(f"Encode 失敗：\n```\n{e}\n```")

    with op_cols[1]:
        if st.button(
            "清除快取", use_container_width=True, key="btn_clear_cache",
        ):
            n = clear_vector_cache(selected_key)
            st.info(f"已刪除 {n} 個快取檔")
            st.rerun()

    with op_cols[2]:
        st.caption(
            "重 ingest 規則：file_hash／chunking_strategy／embedding_model／"
            "embedding_version 任一變動就會失效。"
        )

    st.divider()

    # === 區塊 2：本頁即時 embed 預覽 ===
    st.markdown("### 本頁 chunks 即時編碼預覽")
    st.caption("不寫快取，純粹驗證模型輸出與量級。")

    current_chunks_emb = [
        c for c in all_chunks_emb if c["_page_pos"] == current_idx
    ]
    if not current_chunks_emb:
        st.info("本頁無 chunks。")
    else:
        st.caption(f"本頁 {len(current_chunks_emb)} 個 chunks。")
        if st.button(
            "Embed 本頁",
            key="btn_preview_encode",
            use_container_width=True,
        ):
            try:
                embedder = load_embedder(model_name, use_fp16, st.session_state.get("embed_device", "cpu"))
            except Exception as e:
                st.error(f"模型載入失敗：\n```\n{e}\n```")
                embedder = None

            if embedder is not None:
                texts = [
                    c["payload"]["content"]["text_with_prefix"]
                    for c in current_chunks_emb
                ]
                t0 = time.time()
                try:
                    dense, sparse = encode_chunks(
                        embedder, texts, batch_size=batch_size
                    )
                    elapsed = time.time() - t0
                    rows = []
                    for c, d, s in zip(current_chunks_emb, dense, sparse):
                        text_prev = c["payload"]["content"]["text_with_prefix"]
                        rows.append({
                            "id": c["id"][:8] + "…",
                            "preview": text_prev[:60] + ("…" if len(text_prev) > 60 else ""),
                            "dense_norm": round(float(np.linalg.norm(d)), 4),
                            "sparse_nnz": len(s),
                        })
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    st.caption(
                        f"耗時 {elapsed:.2f}s · {len(texts) / elapsed:.1f} chunks/s · "
                        f"dense shape={tuple(dense.shape)} · "
                        f"sparse avg nnz={sum(len(s) for s in sparse) / len(sparse):.1f}"
                    )
                except Exception as e:
                    st.error(f"Encode 失敗：\n```\n{e}\n```")

# --- 分頁 5：Qdrant 寫入 ---
with tab_qdrant:
    st.subheader("Qdrant 寫入（hybrid: dense + sparse）")
    st.caption(
        "把本地快取的向量 + 即時 chunk payload upsert 到 Qdrant。"
        "deterministic point id 確保同檔重傳直接覆蓋，不會產生重複。"
        f" Collection schema 對齊 `qdrant格式.md §2`。"
    )

    if not HAS_QDRANT:
        st.error(
            "qdrant-client 未安裝：\n```\npip install qdrant-client\n```"
        )
        st.stop()

    # === 區塊 1：連線測試 ===
    st.markdown("### 連線狀態")
    conn_cols = st.columns([2, 1])
    with conn_cols[0]:
        st.markdown(
            f"**URL**：`{qdrant_url}` ｜ **Collection**：`{qdrant_collection_name}`"
        )
    with conn_cols[1]:
        ping = st.button("🔌 連線測試", use_container_width=True, key="qd_ping")

    client = None
    conn_ok = False
    try:
        client = get_qdrant_client(qdrant_url, qdrant_api_key)
        # 輕量探測：列出 collections
        _ = client.get_collections()
        conn_ok = True
        if ping:
            st.success(f"✓ 已連線：{qdrant_url}")
    except Exception as e:
        st.error(f"連線失敗：\n```\n{e}\n```")
        conn_ok = False

    if not conn_ok:
        st.stop()

    # === 區塊 2：Collection 狀態 + 管理 ===
    st.markdown("### Collection")
    coll_exists = client.collection_exists(qdrant_collection_name)
    coll_total = get_collection_total(client, qdrant_collection_name) if coll_exists else 0

    coll_cols = st.columns(4)
    coll_cols[0].metric("存在", "是" if coll_exists else "否")
    coll_cols[1].metric("總 points", coll_total)
    coll_cols[2].metric("dense dim", EMBEDDING_DIM_DENSE)
    coll_cols[3].metric("vectors", "dense + sparse")

    coll_op_cols = st.columns([1, 2, 2])
    with coll_op_cols[0]:
        if not coll_exists and st.button(
            "建立 collection",
            type="primary",
            use_container_width=True,
            key="qd_create",
        ):
            try:
                info = ensure_text_collection(client, qdrant_collection_name, recreate=False)
                st.success(
                    f"✓ collection 建立完成 · payload indexes：{len(info['indexed'])} 個"
                )
                st.rerun()
            except Exception as e:
                st.error(f"建立失敗：\n```\n{e}\n```")
    with coll_op_cols[1]:
        confirm_recreate = st.checkbox(
            "確認重建（會刪光現有資料）",
            value=False,
            key="qd_confirm_recreate",
        )
    with coll_op_cols[2]:
        if st.button(
            "🗑️ 重建 collection",
            use_container_width=True,
            disabled=not confirm_recreate or not coll_exists,
            key="qd_recreate",
        ):
            try:
                info = ensure_text_collection(client, qdrant_collection_name, recreate=True)
                st.success(
                    f"✓ collection 重建完成 · payload indexes：{len(info['indexed'])} 個"
                )
                st.rerun()
            except Exception as e:
                st.error(f"重建失敗：\n```\n{e}\n```")

    if not coll_exists:
        st.info("📌 先建立 collection 再上傳。")
        st.stop()

    st.divider()

    # === 區塊 3：本檔 ingest 預覽 ===
    st.markdown("### 本檔上傳預覽")

    # 重建 chunks（與 Tab 3/4 同邏輯）
    fm_qd, _, pages_qd = parse_md(st.session_state.current_md)
    editor_key_qd = editor_key_for(
        selected_key, current_idx, st.session_state.widget_version
    )
    edited_qd = st.session_state.get(editor_key_qd)
    if isinstance(edited_qd, str):
        page_md_qd = edited_qd
    else:
        _, page_md_qd = pages_qd[current_idx]
    file_labels_qd = st.session_state.custom_labels.setdefault(
        selected_key, {"document": [], "pages": {}}
    )
    doc_labels_qd = file_labels_qd["document"]
    split_cfg_qd = st.session_state.split_settings.setdefault(
        selected_key, {"mode": "delimiter", "delim": "\\n"}
    )
    h1_qd = extract_h1(st.session_state.current_md, Path(file_name).stem)
    all_chunks_qd = build_all_chunks_for_doc(
        file_key=selected_key,
        sidecar=sidecar,
        pages=pages_qd,
        page_md_override={current_idx: page_md_qd},
        h1=h1_qd,
        doc_labels=doc_labels_qd,
        file_labels=file_labels_qd,
        split_cfg=split_cfg_qd,
        image_root=image_root,
    )
    expected_chunk_ids_qd = [c["id"] for c in all_chunks_qd]

    file_hash_now = sidecar.get("file_hash", "")
    existing_for_file = count_existing_for_file(
        client, qdrant_collection_name, file_hash_now
    )
    cache_status_qd = vector_cache_status(
        selected_key, sidecar, expected_chunk_ids_qd
    )
    is_last_page_qd = current_idx == len(pages_qd) - 1
    last_page_num_qd = pages_qd[-1][0] if pages_qd else None

    prev_cols = st.columns(4)
    prev_cols[0].metric("本檔 chunks", len(expected_chunk_ids_qd))
    prev_cols[1].metric("Qdrant 已存在", existing_for_file)
    prev_cols[2].metric(
        "本地快取",
        "✓ valid" if cache_status_qd["valid"] else "✗ invalid",
    )
    prev_cols[3].metric(
        "操作",
        "覆蓋" if existing_for_file > 0 else "新增",
    )

    # === 區塊 4：阻擋條件 ===
    blockers: list[str] = []
    if not is_last_page_qd:
        blockers.append(
            f"⏭️ 需切到最後一頁（P{last_page_num_qd}）才能上傳；目前 P{page_num}"
        )
    if not cache_status_qd["valid"]:
        blockers.append(
            f"❌ 本地向量快取無效：{cache_status_qd.get('reason', '?')}"
            f" → 先到 Tab 4 重新 encode"
        )
    if not expected_chunk_ids_qd:
        blockers.append("❌ 本檔無可上傳 chunks")
    if not file_hash_now:
        blockers.append("⚠️ file_hash 為空（原始檔找不到？），上傳但無法重複偵測")

    for msg in blockers:
        st.warning(msg)

    # === 區塊 5：上傳 ===
    upload_disabled = bool([b for b in blockers if b.startswith("❌") or b.startswith("⏭️")])
    if st.button(
        "🚀 上傳到 Qdrant",
        type="primary",
        use_container_width=True,
        disabled=upload_disabled,
        key="qd_upload",
    ):
        try:
            with st.spinner("讀取本地快取..."):
                dense_cached, sparse_cached, manifest_cached = load_cached_vectors(
                    selected_key
                )
            # 防呆對齊
            if manifest_cached.get("chunk_ids") != expected_chunk_ids_qd:
                st.error(
                    "快取的 chunk_ids 與當前 chunks 不一致，請到 Tab 4 重新 encode。"
                )
            else:
                ingestion_ts = now_iso()
                points = build_points_for_upsert(
                    all_chunks_qd, dense_cached, sparse_cached, ingestion_ts
                )
                total = len(points)
                progress = st.progress(0.0, text=f"Upsert 0 / {total}...")
                t0 = time.time()

                def _cb(done: int, t: int) -> None:
                    progress.progress(
                        done / t if t else 1.0,
                        text=f"Upsert {done} / {t}...",
                    )

                upsert_in_batches(
                    client, qdrant_collection_name, points,
                    batch_size=QDRANT_BATCH_SIZE, progress_cb=_cb,
                )
                elapsed = time.time() - t0
                progress.empty()

                # ✓ 寫入成功 → 狀態轉 🔵 ingested
                mark_ingested(selected_key)
                st.success(
                    f"✓ {total} 個 points 已上傳 · {elapsed:.1f}s · "
                    f"{total / elapsed:.1f} pts/s · 狀態 → 🔵 已寫入庫"
                )
                st.balloons()
                st.rerun()
        except Exception as e:
            st.error(f"上傳失敗：\n```\n{e}\n```")

    st.divider()

    # === 區塊 6：smoke query 驗證 ===
    with st.expander("🔍 上傳後驗證（scroll 抓本檔前 5 個 points）"):
        if st.button(
            "查詢本檔 points", key="qd_scroll", disabled=not file_hash_now,
        ):
            try:
                hits, _ = client.scroll(
                    collection_name=qdrant_collection_name,
                    scroll_filter=qm.Filter(must=[
                        qm.FieldCondition(
                            key="metadata.source.file_hash",
                            match=qm.MatchValue(value=file_hash_now),
                        ),
                    ]),
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )
                if not hits:
                    st.info("Qdrant 內無此 file_hash 的 points。")
                else:
                    rows = []
                    for h in hits:
                        loc = h.payload.get("metadata", {}).get("location", {})
                        sysi = h.payload.get("sys_info", {})
                        rows.append({
                            "id": str(h.id)[:8] + "…",
                            "page": loc.get("page"),
                            "section": loc.get("section_idx"),
                            "chunk": loc.get("chunk_idx"),
                            "ingestion_time": sysi.get("ingestion_time"),
                            "review_status": sysi.get("review_status"),
                        })
                    st.dataframe(rows, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"查詢失敗：\n```\n{e}\n```")


# --- 分頁 6：檢索測試 ---
with tab_search:
    st.subheader("檢索測試（dense / sparse / hybrid 三欄並排）")
    st.caption(
        "對已寫入 Qdrant 的向量做語意檢索。比較三種模式對術語 / 案場名 / 工法名的回收差異。"
        "點「📂 內嵌顯示此頁」會在結果方塊內直接渲染該引用頁，肉眼驗證 citation 不必離開本 Tab。"
    )

    if not HAS_QDRANT:
        st.error("qdrant-client 未安裝：`pip install qdrant-client`")
        st.stop()

    # 共用 cache 的 client
    try:
        s_client = get_qdrant_client(qdrant_url, qdrant_api_key)
        _ = s_client.get_collections()
    except Exception as e:
        st.error(f"Qdrant 連線失敗：\n```\n{e}\n```")
        st.stop()

    if not s_client.collection_exists(qdrant_collection_name):
        st.warning(f"Collection `{qdrant_collection_name}` 不存在，請先到 Tab 5 建立。")
        st.stop()

    s_total = get_collection_total(s_client, qdrant_collection_name)
    if s_total == 0:
        st.warning("Collection 內沒有 points，請先到 Tab 5 ingest 一份檔案。")
        st.stop()
    st.caption(f"Collection `{qdrant_collection_name}` 目前共 **{s_total}** points")

    # === 控制列 ===
    ctrl_cols = st.columns([4, 1, 1])
    with ctrl_cols[0]:
        s_query = st.text_input(
            "Query",
            key="search_query_input",
            placeholder="例如：勤美璞真新洲美的機電廠商是哪一家？",
        )
    with ctrl_cols[1]:
        s_top_k = st.number_input(
            "Top-K", min_value=1, max_value=50, value=10, step=1,
            key="search_top_k",
        )
    with ctrl_cols[2]:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # 對齊
        s_run = st.button(
            "🔍 搜尋", type="primary", use_container_width=True, key="search_run",
        )

    # === Filter ===
    # 建案選項用 derive_project_name（= file_key[1]），對齊 payload 的 project_name；
    # 用 file_key[0] 會列成「類別」而與 payload 對不上、filter 撈不到。
    s_project_options = sorted({
        derive_project_name(fk)
        for fk in file_keys
        if len(split_key_parts(fk)) > 1
    })
    s_doc_type_options = list_doc_types_in_collection(
        qdrant_url, qdrant_api_key, qdrant_collection_name
    )
    s_fcols = st.columns(2)
    with s_fcols[0]:
        s_projects = st.multiselect(
            "（選填）只搜這些建案", options=s_project_options, default=[],
            key="search_filter_projects",
            help="留空 = 搜整個 collection。多選用 OR。",
        )
    with s_fcols[1]:
        s_doc_types = st.multiselect(
            "（選填）只搜這些文件種類", options=s_doc_type_options, default=[],
            key="search_filter_doc_types",
            help="依 metadata.source.doc_type 篩。多選用 OR。",
        )

    # 專案 + 文件種類各一條 must（條內 OR、條間 AND）；皆留空＝None＝不篩
    _s_must = []
    if s_projects:
        _s_must.append(qm.FieldCondition(
            key="metadata.source.project_name",
            match=qm.MatchAny(any=list(s_projects)),
        ))
    if s_doc_types:
        _s_must.append(qm.FieldCondition(
            key="metadata.source.doc_type",
            match=qm.MatchAny(any=list(s_doc_types)),
        ))
    s_filter = qm.Filter(must=_s_must) if _s_must else None

    # === 執行檢索 ===
    if s_run and s_query.strip():
        try:
            s_embedder = load_embedder(
                st.session_state.get("embed_model", EMBEDDING_MODEL),
                st.session_state.get("embed_fp16", True),
                st.session_state.get("embed_device", "cpu"),
            )
        except Exception as e:
            st.error(f"模型載入失敗：\n```\n{e}\n```")
            st.stop()

        with st.spinner("Encode query + 三種模式檢索中..."):
            t0 = time.time()
            q_dense, q_sparse = encode_query(s_embedder, s_query.strip())
            t_encode = time.time() - t0

            modes_result = {}
            for label, runner in [
                ("dense", lambda: search_dense(s_client, qdrant_collection_name, q_dense, int(s_top_k), s_filter)),
                ("sparse", lambda: search_sparse(s_client, qdrant_collection_name, q_sparse, int(s_top_k), s_filter)),
                ("hybrid", lambda: search_hybrid(s_client, qdrant_collection_name, q_dense, q_sparse, int(s_top_k), s_filter)),
            ]:
                ts = time.time()
                try:
                    hits = runner()
                    err = None
                except Exception as e:
                    hits = []
                    err = str(e)
                modes_result[label] = {
                    "hits": hits,
                    "elapsed": time.time() - ts,
                    "error": err,
                }

        st.session_state["_search_result"] = {
            "query": s_query.strip(),
            "t_encode": t_encode,
            "modes": modes_result,
        }

    # === 結果渲染 ===
    sr = st.session_state.get("_search_result")
    if sr:
        st.markdown(f"**Query**：`{sr['query']}` · encode {sr['t_encode']*1000:.0f} ms")

        result_cols = st.columns(3)
        mode_titles = [
            ("dense", "🟦 Dense（語意 / bge-m3 dense）"),
            ("sparse", "🟧 Sparse（詞彙 / bge-m3 sparse）"),
            ("hybrid", "🟪 Hybrid（RRF 融合）"),
        ]
        for col, (key, title) in zip(result_cols, mode_titles):
            with col:
                info = sr["modes"][key]
                st.markdown(f"**{title}**")
                if info["error"]:
                    st.error(f"檢索失敗：\n```\n{info['error']}\n```")
                    continue
                st.caption(f"⏱ {info['elapsed']*1000:.0f} ms · {len(info['hits'])} 筆")
                if not info["hits"]:
                    st.info("無結果")
                    continue
                for rank, h in enumerate(info["hits"], start=1):
                    payload = h.payload or {}
                    src = payload.get("metadata", {}).get("source", {}) or {}
                    loc = payload.get("metadata", {}).get("location", {}) or {}
                    content = payload.get("content", {}) or {}

                    project = src.get("project_name", "—")
                    file_key_hit = src.get("file_key", "")
                    page_hit = loc.get("page")
                    headings_flat = loc.get("headings_flat", []) or []
                    text_preview = (content.get("text", "") or "")[:240]
                    score = h.score if hasattr(h, "score") else None

                    with st.container(border=True):
                        head_cols = st.columns([3, 1])
                        head_cols[0].markdown(
                            f"**#{rank}** ｜ `{project}` · P{page_hit}"
                        )
                        if score is not None:
                            head_cols[1].markdown(
                                f"<div style='text-align:right; color:#666;'>"
                                f"score <code>{score:.4f}</code></div>",
                                unsafe_allow_html=True,
                            )
                        if headings_flat:
                            st.caption(" › ".join(str(h_) for h_ in headings_flat))
                        st.caption(f"📄 `{Path(file_key_hit).name}`")
                        st.markdown(
                            f"<div style='font-size:0.88em; color:#333; "
                            f"max-height:160px; overflow:auto; "
                            f"background:#f7f7f8; padding:8px; border-radius:4px;'>"
                            f"{text_preview.replace('<', '&lt;').replace('>', '&gt;')}"
                            f"{'…' if len(content.get('text','') or '') > 240 else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        # 內嵌頁面預覽（取代舊「跳回 Tab 1」）：比照對話模式 _render_source_page_inline，
                        # 伺服器把該引用頁渲染成 PNG 原地展開／收合，肉眼驗證 citation 不必離開本 Tab。
                        if file_key_hit and page_hit and file_key_hit in file_keys:
                            prev_token = f"{key}_{rank}_{h.id}"
                            is_open = st.session_state.get("_search_preview_token") == prev_token
                            if st.button(
                                "📂 收合此頁" if is_open else "📂 內嵌顯示此頁",
                                key=f"srch_prev_{prev_token}",
                                use_container_width=True,
                            ):
                                st.session_state["_search_preview_token"] = (
                                    None if is_open else prev_token
                                )
                                st.rerun()
                            if st.session_state.get("_search_preview_token") == prev_token:
                                _render_source_page_inline(
                                    data_path / file_key_hit, int(page_hit),
                                    key_suffix=f"srch_{prev_token}",
                                )
                        elif file_key_hit and file_key_hit not in file_keys:
                            st.caption(f"⚠ 本機未載入此檔，無法預覽：`{Path(file_key_hit).name}`")
