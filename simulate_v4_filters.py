"""
模擬 mkd_generator_v4.py 的圖片過濾鏈，輸出落差分布。

支援兩種模式（--mode）：
  legacy : 復刻舊版邏輯
    - pre-scan 用 page.get_images()（resource dict）算 hash 出現率
    - extract 用 bbox width > 5 判斷可見性
  v2     : 新版邏輯（visibility 一致化）
    - pre-scan 跟 extract 都用 _visible_image_xrefs_on_page：
      bbox 寬高 > min_pt 且與頁面矩形相交，且必須是 placement（非 resource-only）

兩端共用：
  - hash 在 >50% 頁面出現 → banned_image_hashes
  - extract: bbox 不過 → skip / hash banned → skip / width or height < 60 → skip
  - 同 hash 只存一份檔案，後續頁面只追加 md 引用

慢路徑（drawings>200 走 Marker）的頁面不走 fitz 抽圖，模擬時跳過。
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import fitz


def _visible_xrefs(page, min_pt: int = 5) -> set[int]:
    page_rect = page.rect
    visible = set()
    for info in page.get_image_info(xrefs=True):
        x0, y0, x1, y1 = info["bbox"]
        if (x1 - x0) <= min_pt or (y1 - y0) <= min_pt:
            continue
        if x1 <= page_rect.x0 or x0 >= page_rect.x1:
            continue
        if y1 <= page_rect.y0 or y0 >= page_rect.y1:
            continue
        visible.add(info["xref"])
    return visible


def simulate(pdf_path: Path, mode: str = "v2") -> dict:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    xref_to_hash: dict[int, str] = {}
    page_xrefs: list[set[int]] = []
    page_visible_xrefs: list[set[int]] = []
    drawings_per_page: list[int] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        drawings_per_page.append(len(page.get_drawings()))

        if mode == "legacy":
            actual_images = page.get_image_info(xrefs=True)
            visible = {
                img["xref"]
                for img in actual_images
                if img["bbox"][2] - img["bbox"][0] > 5
            }
        else:
            visible = _visible_xrefs(page)
        page_visible_xrefs.append(visible)

        xrefs_on_page = set()
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            xrefs_on_page.add(xref)
            if xref not in xref_to_hash:
                base = doc.extract_image(xref)
                xref_to_hash[xref] = hashlib.md5(base["image"]).hexdigest()
        page_xrefs.append(xrefs_on_page)

    image_occurrence: Counter[str] = Counter()
    if mode == "legacy":
        for xrefs in page_xrefs:
            page_hashes = {xref_to_hash[x] for x in xrefs}
            for h in page_hashes:
                image_occurrence[h] += 1
    else:
        for visible in page_visible_xrefs:
            page_hashes = {xref_to_hash[x] for x in visible}
            for h in page_hashes:
                image_occurrence[h] += 1
    banned_hashes = {
        h for h, count in image_occurrence.items() if count / total_pages > 0.5
    }
    marker_pages = {i for i, n in enumerate(drawings_per_page) if n > 200}

    breakdown = Counter()
    saved_hashes: set[str] = set()
    md_refs = 0
    saved_size_total = 0

    for page_idx in range(total_pages):
        if page_idx in marker_pages:
            for xref in page_xrefs[page_idx]:
                breakdown["routed_to_marker_path"] += 1
            continue

        visible = page_visible_xrefs[page_idx]
        for xref in page_xrefs[page_idx]:
            h = xref_to_hash[xref]
            if xref not in visible:
                breakdown["skipped_bbox<=5"] += 1
                continue
            if h in banned_hashes:
                breakdown["skipped_banned_global"] += 1
                continue
            if h in saved_hashes:
                md_refs += 1
                breakdown["ref_existing_file"] += 1
                continue
            base = doc.extract_image(xref)
            if base["width"] < 60 or base["height"] < 60:
                breakdown["skipped_<60px"] += 1
                continue
            saved_hashes.add(h)
            saved_size_total += len(base["image"])
            md_refs += 1
            breakdown["saved_new_file"] += 1

    doc.close()

    return {
        "total_pages": total_pages,
        "marker_pages": sorted(marker_pages),
        "all_unique_hashes": len(image_occurrence),
        "banned_count": len(banned_hashes),
        "saved_unique_files": len(saved_hashes),
        "md_refs": md_refs,
        "saved_size_total": saved_size_total,
        "breakdown": dict(breakdown),
    }


def _print_report(label: str, r: dict) -> None:
    print(f"=== {label} ===")
    print(f"PDF pages              : {r['total_pages']}")
    print(f"Marker pages (drawings>200): {len(r['marker_pages'])}  indices={r['marker_pages']}")
    print(f"PDF 內唯一圖 hash 數   : {r['all_unique_hashes']}")
    print(f"被 banned (>50% 頁覆蓋): {r['banned_count']}")
    print()
    print(f"實體儲存檔案數 (saved): {r['saved_unique_files']}")
    print(f"md `![]()` 引用次數   : {r['md_refs']}")
    print(f"儲存圖片總位元組      : {r['saved_size_total']:,}")
    print()
    print("過濾鏈 breakdown（以「每次 reference」計）:")
    for k, v in sorted(r["breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s}: {v}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate v4 image filter chain on a PDF")
    parser.add_argument("pdf", help="PDF 檔（用 LibreOffice4windows.py 產出的那份）")
    parser.add_argument(
        "--mode", choices=["legacy", "v2", "both"], default="both",
        help="legacy=舊版 / v2=可見性一致化 / both=兩個都跑並對比（預設）",
    )
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"找不到 PDF: {pdf}", file=sys.stderr)
        return 1

    if args.mode in ("legacy", "both"):
        _print_report("LEGACY (舊版 — pre-scan 用 resource dict)", simulate(pdf, "legacy"))
    if args.mode in ("v2", "both"):
        _print_report("V2 (新版 — pre-scan 用 visibility)", simulate(pdf, "v2"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
