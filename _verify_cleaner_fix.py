"""Verify: 套用 fix 後，SpanLevelTextCleaner 是否真的把 P25/P26 的
KAICHU 浮水印 + 軟橋段71地號 頁首給刪掉。

跑法：
    python _verify_cleaner_fix.py

只讀 PDF，輸出寫到 _verify_cleaner_fix.out.txt。
"""
from __future__ import annotations

from pathlib import Path

import fitz

from mkd_generator_v5.page_cache import CachedPage
from mkd_generator_v5.strategies import (
    HierarchicalTitleExtractor,
    PositionalTemplateFilter,
    SpanLevelTextCleaner,
)

PDF = Path(
    "璞真RAG_rawdata/文林中正/1110119_璞真文林中正案_結構銷售簡報.pdf"
).absolute()
PAGES = [24, 25]  # 0-based: 第 25、26 頁


def main() -> None:
    out_lines: list[str] = []

    def w(s: str = "") -> None:
        out_lines.append(s)

    doc = fitz.open(PDF)
    template = PositionalTemplateFilter()
    title_extractor = HierarchicalTitleExtractor()
    cleaner = SpanLevelTextCleaner()

    w(f"PDF: {PDF}")
    w(f"pages in doc: {len(doc)}")
    w("")
    w("=== template scan ===")
    state = template.scan(doc)
    w(f"banned_global_texts ({len(state.banned_global_texts)}):")
    for t in sorted(state.banned_global_texts):
        w(f"  {t!r}")
    w(f"blacklisted_regions keys ({len(state.blacklisted_regions)}):")
    for k, boxes in state.blacklisted_regions.items():
        w(f"  key={k!r}  n_bboxes={len(boxes)}")
        for b in boxes[:3]:
            w(f"    bbox={tuple(round(v, 2) for v in b)}")

    for pidx in PAGES:
        page = CachedPage(doc[pidx])
        title_hit = title_extractor.extract(page)
        cleaned = cleaner.clean_structured(page, pidx, title_hit, state)
        w("")
        w("=" * 80)
        w(f"PAGE {pidx + 1}  title={title_hit.text if title_hit else None!r}")
        w("=" * 80)
        for y0, text in cleaned:
            preview = text.replace("\n", " | ")
            w(f"  y0={y0:7.2f}  text={preview!r}")

        # 紅旗檢查：cleaned 文字串接後是否還有 KAICHU / 軟橋段
        all_text = "".join(t for _, t in cleaned)
        flagged: list[str] = []
        for needle in ("KAICHU", "軟橋段71地號"):
            if needle in all_text:
                flagged.append(needle)
        if flagged:
            w(f"  ❌ STILL CONTAINS: {flagged}")
        else:
            w(f"  ✅ clean (KAICHU & 軟橋段71地號 both removed)")

    doc.close()
    out = Path("_verify_cleaner_fix.out.txt").absolute()
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"verify written: {out}")


if __name__ == "__main__":
    main()
