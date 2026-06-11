"""Diag: 同一頁 get_text('blocks') vs get_text('dict')->lines->spans 的 norm key 差異。

驗證假設：v5 SpanLevelTextCleaner 用 dict 拼出來的字串當黑名單 key，跟
PositionalTemplateFilter 用 blocks 建出來的 key 對不齊，導致黑名單失效。

跑法：
    python _diag_blocks_vs_dict.py <pdf_path> <page_idx_0based> [more pages...]

只讀 PDF，不寫任何輸出檔。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def dict_block_text(b: dict) -> str:
    """SpanLevelTextCleaner 的拼法：lines 內 spans "" join、lines 之間 ""(在 norm 之前)。

    對齊 text_cleaner.py:218-220
        original_text_lines.append("".join(orig_parts))
        original_norm = re.sub(r"\\s+", "", "".join(original_text_lines))
    """
    lines: list[str] = []
    for line in b.get("lines", []):
        parts = [s.get("text", "") for s in line.get("spans", []) if s.get("text", "")]
        if parts:
            lines.append("".join(parts))
    return "".join(lines)


def run(pdf_path: Path, page_indices: list[int], out_path: Path) -> None:
    doc = fitz.open(pdf_path)
    lines: list[str] = []

    def w(msg: str = "") -> None:
        lines.append(msg)

    for pidx in page_indices:
        if pidx >= len(doc):
            w(f"[skip] page {pidx} out of range (doc has {len(doc)})")
            continue
        page = doc[pidx]
        w("=" * 80)
        w(f"PAGE {pidx + 1}  (page_idx={pidx})")
        w("=" * 80)

        w("\n--- get_text('blocks') ---")
        blocks_norms: list[tuple[tuple, str, str]] = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:
                continue
            n = norm(text)
            if not n:
                continue
            blocks_norms.append(((x0, y0, x1, y1), text, n))
            w(f"  bbox=({x0:7.2f},{y0:7.2f},{x1:7.2f},{y1:7.2f}) norm={n!r}")

        w("\n--- get_text('dict') reconstructed (SpanLevelTextCleaner 拼法) ---")
        dict_norms: list[tuple[tuple, str, str]] = []
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type", 0) != 0:
                continue
            bbox = b.get("bbox")
            if not bbox:
                continue
            raw = dict_block_text(b)
            n = norm(raw)
            if not n:
                continue
            dict_norms.append((tuple(bbox), raw, n))
            w(f"  bbox=({bbox[0]:7.2f},{bbox[1]:7.2f},{bbox[2]:7.2f},{bbox[3]:7.2f}) norm={n!r}")

        w("\n--- DIFF (blocks-norm vs dict-norm) ---")
        bset = {n for _, _, n in blocks_norms}
        dset = {n for _, _, n in dict_norms}
        only_blocks = bset - dset
        only_dict = dset - bset
        if not only_blocks and not only_dict:
            w("  (一致)")
        if only_blocks:
            w("  ONLY in blocks (template scan 看得到, SpanLevel cleaner 對不上):")
            for n in only_blocks:
                w(f"    {n!r}")
        if only_dict:
            w("  ONLY in dict (cleaner 拼出來, blocks 沒這個 norm):")
            for n in only_dict:
                w(f"    {n!r}")

    doc.close()
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"diag written: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python _diag_blocks_vs_dict.py <pdf> <page_idx_0based> ...")
        sys.exit(1)
    pdf = Path(sys.argv[1])
    pages = [int(x) for x in sys.argv[2:]]
    out = Path("_diag_blocks_vs_dict.out.txt").absolute()
    run(pdf, pages, out)
