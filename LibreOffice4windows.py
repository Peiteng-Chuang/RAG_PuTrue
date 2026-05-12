"""
Windows: PPT/PPTX → PDF (LibreOffice headless) + 全圖提取（無過濾）。

驗證用途：
- 不套 v4 的任何過濾（60px 閾值 / banned hash / dedup），只把 PDF 內所有 raster image 全部 dump，
  以便對「PPT 原始圖數 → PDF 內可見圖數」的落差做量化。
- drawings 數附帶輸出，提示「向量圖盲區」（v4 fast path 抓不到向量繪圖）。

依賴：LibreOffice for Windows、PyMuPDF（fitz）。
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF


PPT_EXTS = {".ppt", ".pptx", ".pptm"}
DEFAULT_SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")


def find_soffice() -> Path:
    if DEFAULT_SOFFICE.exists():
        return DEFAULT_SOFFICE
    found = shutil.which("soffice") or shutil.which("soffice.exe")
    if found:
        return Path(found)
    raise RuntimeError(
        r"找不到 soffice.exe；預期路徑：C:\Program Files\LibreOffice\program\soffice.exe"
    )


def convert_ppt_to_pdf(soffice: Path, ppt: Path, out_dir: Path, timeout: int = 300) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = out_dir / f".uno_profile_{ppt.stem}"
    # Windows 路徑必須用 Path.as_uri() 產出 file:///C:/... ；
    # f"file://{profile.resolve()}" 在 Windows 會變成 file://C:\... ，soffice 會 rc=1 直接退。
    user_install = profile.resolve().as_uri()
    cmd = [
        str(soffice),
        f"-env:UserInstallation={user_install}",
        "--headless",
        "--norestore",
        "--nologo",
        "--convert-to", "pdf",
        "--outdir", str(out_dir.resolve()),
        str(ppt.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    shutil.rmtree(profile, ignore_errors=True)

    pdf = out_dir / f"{ppt.stem}.pdf"
    if proc.returncode != 0 or not pdf.exists():
        raise RuntimeError(
            f"soffice rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return pdf


def extract_all_images(pdf_path: Path, img_dir: Path) -> dict:
    img_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    stats = {
        "pages": len(doc),
        "raw_image_refs": 0,
        "unique_xrefs": 0,
        "saved_files": 0,
        "extract_failures": 0,
        "drawings_per_page": [],
    }
    seen_xrefs = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        stats["drawings_per_page"].append(len(page.get_drawings()))
        for img_info in page.get_images(full=True):
            stats["raw_image_refs"] += 1
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base = doc.extract_image(xref)
            except Exception as e:
                stats["extract_failures"] += 1
                print(f"WARN p{page_idx+1} xref={xref} extract 失敗: {e}", file=sys.stderr)
                continue
            ext = base.get("ext", "png")
            data = base["image"]
            h = hashlib.md5(data).hexdigest()[:8]
            name = f"{pdf_path.stem}_p{page_idx+1}_x{xref}_{h}.{ext}"
            (img_dir / name).write_bytes(data)
            stats["saved_files"] += 1

    stats["unique_xrefs"] = len(seen_xrefs)
    doc.close()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Windows: PPT → PDF (LibreOffice) + 全圖提取（無過濾，驗證用）"
    )
    parser.add_argument("input", help="PPT/PPTX 檔")
    parser.add_argument(
        "-o", "--output", default="./tmp/libreoffice4win_out",
        help="輸出資料夾（預設 ./tmp/libreoffice4win_out）",
    )
    parser.add_argument("--timeout", type=int, default=300, help="soffice 逾時秒數")
    args = parser.parse_args()

    ppt = Path(args.input)
    if not ppt.is_file() or ppt.suffix.lower() not in PPT_EXTS:
        print(f"非 PPT/PPTX 檔或不存在: {ppt}", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    soffice = find_soffice()
    print(f"soffice : {soffice}")
    print(f"input   : {ppt}")
    print(f"output  : {out_dir.resolve()}")

    pdf = convert_ppt_to_pdf(soffice, ppt, out_dir, args.timeout)
    print(f"PDF OK  : {pdf} ({pdf.stat().st_size:,} bytes)")

    img_dir = out_dir / f"{ppt.stem}_image"
    stats = extract_all_images(pdf, img_dir)
    drawings_total = sum(stats["drawings_per_page"])
    drawings_max = max(stats["drawings_per_page"]) if stats["drawings_per_page"] else 0
    over_200_pages = sum(1 for n in stats["drawings_per_page"] if n > 200)

    print()
    print(f"頁數                : {stats['pages']}")
    print(f"圖片引用次數        : {stats['raw_image_refs']}    (page.get_images 條目總和，含跨頁重用)")
    print(f"唯一 xref 數        : {stats['unique_xrefs']}    (= 儲存檔案數: {stats['saved_files']})")
    print(f"extract 失敗數      : {stats['extract_failures']}")
    print(f"drawings 總數       : {drawings_total}")
    print(f"  每頁最大 drawings : {drawings_max}")
    print(f"  drawings>200 頁數 : {over_200_pages}    (v4 中這類頁會走 Marker)")
    print(f"圖片輸出資料夾      : {img_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
