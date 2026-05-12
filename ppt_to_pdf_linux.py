"""
Ubuntu/Linux 下將 PPT/PPTX 轉為 PDF。
依賴：LibreOffice（apt install libreoffice fonts-noto-cjk）；不需 Python 套件。
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PPT_EXTS = {".ppt", ".pptx", ".pptm"}


def find_soffice() -> str:
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("找不到 soffice/libreoffice，請先 apt install libreoffice")


def collect_inputs(src: Path) -> list[Path]:
    if src.is_file():
        if src.suffix.lower() not in PPT_EXTS:
            raise ValueError(f"非 PPT 檔: {src}")
        return [src]
    if src.is_dir():
        return sorted(p for p in src.rglob("*") if p.suffix.lower() in PPT_EXTS)
    raise FileNotFoundError(src)


def convert_one(soffice: str, ppt: Path, out_dir: Path, timeout: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 每檔獨立 user profile，避免並行/殘留 lock 卡死
    profile = out_dir / f".uno_profile_{ppt.stem}"
    cmd = [
        soffice,
        f"-env:UserInstallation=file://{profile.resolve()}",
        "--headless",
        "--norestore",
        "--nologo",
        "--convert-to", "pdf",
        "--outdir", str(out_dir.resolve()),
        str(ppt.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    shutil.rmtree(profile, ignore_errors=True)

    pdf = out_dir / (ppt.stem + ".pdf")
    if proc.returncode != 0 or not pdf.exists():
        raise RuntimeError(
            f"soffice rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="PPT/PPTX → PDF（Linux + LibreOffice）")
    parser.add_argument("input", help="單一 PPT 檔或資料夾")
    parser.add_argument("-o", "--output", help="輸出資料夾（預設與來源同層）")
    parser.add_argument("--timeout", type=int, default=300, help="單檔逾時秒數")
    args = parser.parse_args()

    soffice = find_soffice()
    src = Path(args.input)
    files = collect_inputs(src)
    if not files:
        print("找不到可轉檔的 PPT。")
        return 1

    fail = 0
    for ppt in files:
        out_dir = Path(args.output) if args.output else ppt.parent
        try:
            pdf = convert_one(soffice, ppt, out_dir, args.timeout)
            print(f"OK  {ppt.name} -> {pdf}")
        except Exception as e:
            fail += 1
            print(f"FAIL {ppt.name}: {e}", file=sys.stderr)

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
