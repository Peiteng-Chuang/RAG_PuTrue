"""pre_convert_ppt.py — 批次把 PPT/PPTX 預轉成 PDF，塞進 md_review_ui 的比對快取。

目的（治本）：把「LibreOffice 轉檔」這個慢動作**搬離 md_review_ui 的互動熱路徑**。
預轉後，UI 的 ensure_pdf 只會在 `_compare_cache/{stem}.pdf` 找到現成 PDF → 秒開，
不再於網頁裡同步轉檔、也不會觸發「轉檔阻塞 → websocket 斷線 → 重連重跑 → 再轉」的無限迴圈。

**與 md_review_ui.ensure_pdf 完全對齊**（勿改動這些慣例，否則 UI 找不到）：
- 輸出目錄：<專案>/_compare_cache/
- 輸出檔名：{來源檔 stem}.pdf（LibreOffice --convert-to pdf --outdir 的預設命名）
- 轉檔旗標 / 獨立 profile / 失敗換 profile 重試一次 / 180s timeout：照抄。

**必為序列化**：同時多個 soffice 共用/並發會 heap corruption（0xC0000374），故一次只轉一個。

**範圍**：獨立批次工具，不 import md_review_ui（那是 streamlit app，import 會執行整個 UI）。
只寫入 _compare_cache（快取，非來源資料）；預設跳過已存在、加 --force 才重轉。

用法：
    python pre_convert_ppt.py                 # 掃 DATA_ROOT（.env）底下所有 ppt/pptx
    python pre_convert_ppt.py --dir "D:\\某資料夾"   # 指定掃描根目錄
    python pre_convert_ppt.py --force         # 連已有快取的也重轉
    python pre_convert_ppt.py --soffice "C:\\...\\soffice.exe"   # 手動指定 LibreOffice
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 與 md_review_ui 對齊：比對快取在專案目錄下 _compare_cache（用檔案自身位置，不依賴 CWD）
PDF_CACHE_DIR = (Path(__file__).resolve().parent / "_compare_cache")

LIBREOFFICE_CANDIDATES_WIN = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    r"D:\Program Files\LibreOffice\program\soffice.exe",
    r"D:\LibreOffice\program\soffice.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\LibreOffice\program\soffice.exe"),
]


def find_soffice(override: str | None = None) -> str | None:
    if override:
        return str(Path(override)) if Path(override).exists() else None
    if platform.system() == "Windows":
        for cand in LIBREOFFICE_CANDIDATES_WIN:
            if cand and Path(cand).exists():
                return cand
        for name in ("soffice.exe", "soffice"):
            found = shutil.which(name)
            if found:
                return found
        return None
    return shutil.which("libreoffice") or shutil.which("soffice")


def _lo_profiles_root() -> Path:
    root = (PDF_CACHE_DIR / "_lo_profiles").absolute()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_soffice_convert(soffice_path: str, source_path: Path, profile_dir: Path):
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


def convert_one(soffice_path: str, source_path: Path) -> tuple[bool, str]:
    """轉一個檔。回 (成功?, 訊息)。與 ensure_pdf 同邏輯：獨立 profile、失敗換 profile 重試一次。"""
    cached = PDF_CACHE_DIR / f"{source_path.stem}.pdf"
    last_err = ""
    result = None
    for _attempt in range(2):
        profile_dir = Path(tempfile.mkdtemp(prefix="lo_", dir=str(_lo_profiles_root())))
        try:
            result = _run_soffice_convert(soffice_path, source_path, profile_dir)
        except subprocess.TimeoutExpired:
            last_err = "轉檔逾時 (>180s)"
            result = None
            continue
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)
        if result.returncode == 0:
            break
        code = result.returncode & 0xFFFFFFFF
        last_err = f"exit={result.returncode}" + (
            " (0xC0000374 heap corruption：關掉其他 LibreOffice 視窗再試)"
            if code == 0xC0000374 else ""
        )

    if result is None or result.returncode != 0:
        return False, last_err or "轉檔失敗"
    if cached.exists() and cached.stat().st_size > 0:
        return True, str(cached.name)
    # 寬鬆比對：LO 可能因檔名特殊字元輸出不同名 → 取最新的 .pdf rename 回 {stem}.pdf
    sibs = sorted((p for p in PDF_CACHE_DIR.glob("*.pdf") if p.stat().st_size > 0),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if sibs and sibs[0] != cached:
        try:
            sibs[0].rename(cached)
            return True, str(cached.name)
        except Exception:  # noqa: BLE001
            return True, str(sibs[0].name)
    return False, "回報成功但找不到輸出 PDF"


def main() -> int:
    ap = argparse.ArgumentParser(description="批次預轉 PPT/PPTX → PDF 到 _compare_cache")
    ap.add_argument("--dir", default=os.environ.get("DATA_ROOT", "."),
                    help="掃描根目錄（遞迴找 ppt/pptx）；預設 .env 的 DATA_ROOT")
    ap.add_argument("--soffice", default="", help="手動指定 soffice 路徑")
    ap.add_argument("--force", action="store_true", help="連已有快取的也重轉")
    args = ap.parse_args()

    soffice = find_soffice(args.soffice or None)
    if not soffice:
        print("❌ 找不到 LibreOffice（soffice）。請安裝或用 --soffice 指定路徑。")
        return 2
    scan_dir = Path(args.dir)
    if not scan_dir.exists():
        print(f"❌ 掃描目錄不存在：{scan_dir}")
        return 2
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in scan_dir.rglob("*")
                     if p.suffix.lower() in (".ppt", ".pptx") and not p.name.startswith("~$"))
    print(f"[soffice] {soffice}")
    print(f"[掃描] {scan_dir}  找到 {len(sources)} 個 PPT/PPTX")
    print(f"[快取] {PDF_CACHE_DIR}\n")

    # stem 撞號偵測（ensure_pdf 只用 stem 命名 → 同名不同檔會互蓋，先警告）
    seen: dict[str, Path] = {}
    converted = skipped = failed = 0
    for i, src in enumerate(sources, 1):
        stem = src.stem
        if stem in seen:
            print(f"  ⚠️ [{i}/{len(sources)}] stem 撞號：{src}  與  {seen[stem]} "
                  f"→ 兩者共用 {stem}.pdf 會互蓋！請改名其一。")
        else:
            seen[stem] = src
        cached = PDF_CACHE_DIR / f"{stem}.pdf"
        if cached.exists() and cached.stat().st_size > 0 and not args.force:
            skipped += 1
            print(f"  · [{i}/{len(sources)}] 已快取，略過：{stem}.pdf")
            continue
        print(f"  → [{i}/{len(sources)}] 轉檔中：{src.name} …", flush=True)
        ok, msg = convert_one(soffice, src)
        if ok:
            converted += 1
            print(f"    ✅ {msg}")
        else:
            failed += 1
            print(f"    ❌ 失敗：{src}  ({msg})")

    print(f"\n[結果] 轉出 {converted} | 略過(已快取) {skipped} | 失敗 {failed}")
    print("完成後 md_review_ui 開這些 PPT 會直接命中快取、秒開，不再於網頁內轉檔。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
