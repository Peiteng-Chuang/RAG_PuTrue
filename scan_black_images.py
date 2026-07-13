"""掃描已抽出的圖片，找出「全黑（無效）」圖片並統計。

背景：PPT/PPTX 經 LibreOffice → PDF → fitz `extract_image()` 抽圖時，
帶透明度的元素（半透明色塊/陰影/去背 PNG/漸層）會被存成「base 影像 + 獨立 SMask」，
而 `extract_image()` 只回 base、丟掉 alpha。透明區的 base 像素多為 (0,0,0)，
剝掉 alpha 後就變成大小不一的「全黑矩形」。本工具就是把這些黑圖找出來。

**唯讀**：只掃描與統計，不刪除、不搬移、不改寫任何檔案。

用法：
    python scan_black_images.py [ROOT] [--threshold N] [--list] [--csv OUT.csv]

    ROOT         要掃描的根目錄（遞迴），預設為目前目錄
    --threshold  近黑容忍值 0-255，像素最大值 <= 此值即視為黑（預設 8）
    --list       逐一列出每張無效圖片
    --csv        另存一份 CSV 明細
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Windows 主控台預設 cp950/big5，中文輸出會變亂碼 → 強制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

try:
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow：pip install Pillow")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".ppm"}


def classify(path: Path, threshold: int) -> tuple[bool, str, int, int]:
    """判斷單張圖是否為無效（全黑/全透明）。

    回 (is_invalid, reason, width, height)。reason 為空字串表示有效或無法判定。
    """
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            bands = im.getbands()

            # 帶 alpha：先看是否整張全透明（alpha 全 0）→ 無效
            if "A" in bands:
                alpha = im.getchannel("A")
                _amin, amax = alpha.getextrema()
                if amax == 0:
                    return True, "fully-transparent", w, h
                # 有可見像素時，只判可見(RGB)部分是否全黑：
                # 轉成 RGB（PIL 會把透明區合到黑底，但這裡我們要看 base 本身），
                # 直接看 RGB 通道的極值即可。
                rgb = im.convert("RGB")
                extrema = rgb.getextrema()
            else:
                extrema = im.convert("RGB").getextrema() if im.mode != "RGB" else im.getextrema()

            # extrema：((minR,maxR),(minG,maxG),(minB,maxB))；取各通道 max 的最大值
            channel_max = max(hi for _lo, hi in extrema)
            if channel_max <= threshold:
                reason = "all-black" if channel_max == 0 else f"near-black(max={channel_max})"
                return True, reason, w, h
            return False, "", w, h
    except Exception as e:  # noqa: BLE001 — 壞檔也算一種「無效」，但另類標記
        return True, f"unreadable({type(e).__name__})", 0, 0


def main() -> int:
    ap = argparse.ArgumentParser(description="掃描全黑/無效圖片")
    ap.add_argument("root", nargs="?", default=".", help="掃描根目錄（遞迴），預設目前目錄")
    ap.add_argument("--threshold", type=int, default=8, help="近黑容忍值 0-255（預設 8）")
    ap.add_argument("--list", action="store_true", help="逐一列出無效圖片")
    ap.add_argument("--csv", metavar="OUT.csv", help="輸出明細 CSV")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"路徑不存在：{root}")

    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    total = len(files)
    print(f"掃描目錄：{root.resolve()}")
    print(f"共發現 {total} 張圖片，開始判定（threshold={args.threshold}）…")

    invalid: list[tuple[Path, str, int, int]] = []
    for i, path in enumerate(files, 1):
        is_bad, reason, w, h = classify(path, args.threshold)
        if is_bad:
            invalid.append((path, reason, w, h))
        if i % 500 == 0:
            print(f"  …已處理 {i}/{total}", file=sys.stderr)

    # 依原因分組統計
    from collections import Counter
    by_reason = Counter(r for _p, r, _w, _h in invalid)

    if args.list:
        for path, reason, w, h in invalid:
            print(f"  [{reason}] {w}x{h}  {path}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "reason", "width", "height"])
            for path, reason, w, h in invalid:
                writer.writerow([str(path), reason, w, h])
        print(f"明細已寫入：{args.csv}")

    print("-" * 50)
    for reason, count in by_reason.most_common():
        print(f"  {reason:<28} {count} 張")
    print("-" * 50)
    print(f"找到 {len(invalid)} 張無效圖片（共掃描 {total} 張）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
