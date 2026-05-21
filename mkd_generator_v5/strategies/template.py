"""模板偵測：相鄰頁同位置同文字 → blacklist；高出現率圖 hash → banned。"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any

from ..types import FilterState


class TemplateFilter(ABC):
    """掃整份 doc 偵測模板雜訊。回 FilterState 給後續 strategy 共用。"""

    @abstractmethod
    def scan(self, doc) -> FilterState: ...


def _visible_image_xrefs_on_page(page, min_pt: float = 5.0) -> set[int]:
    """只認實際 place 在頁面上的 image xref，過濾僅列名於 resource dict 的圖。

    LibreOffice 從 PPT 轉 PDF 時會把 master 圖塞進每頁 resources；
    用 get_images() 會把每張 master 圖當成「全頁都出現」，造成 banned_image_hashes 誤殺。
    """
    page_rect = page.rect
    visible: set[int] = set()
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


class PositionalTemplateFilter(TemplateFilter):
    """v4 等價：
    - 文字：相鄰兩頁同 normalized text 出現在 dx/dy ≤ 10% 頁面尺寸 → blacklist
    - 圖片：某 hash 出現比例 > image_ratio_threshold → banned
    """

    def __init__(
        self,
        position_tolerance: float = 0.1,    # 相鄰頁同位置允許偏移比例
        image_ratio_threshold: float = 0.5,  # 圖片出現比例超此值視為 banner
        min_text_len: int = 2,
    ):
        self.position_tolerance = position_tolerance
        self.image_ratio_threshold = image_ratio_threshold
        self.min_text_len = min_text_len

    def scan(self, doc) -> FilterState:
        state = FilterState()
        total_pages = len(doc)
        if total_pages < 2:
            return state

        prev_page_texts: dict[str, list[float]] = {}
        image_occurrence: dict[str, int] = {}

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_w, page_h = page.rect.width, page.rect.height

            # --- 文字 ---
            curr_page_texts: dict[str, list[float]] = {}
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, _block_no, _block_type = b
                norm_text = re.sub(r"\s+", "", text.strip())
                if not norm_text or len(norm_text) < self.min_text_len:
                    continue
                curr_bbox = [x0, y0, x1, y1]
                if norm_text in prev_page_texts:
                    prev_bbox = prev_page_texts[norm_text]
                    dx = abs(x0 - prev_bbox[0]) / page_w
                    dy = abs(y0 - prev_bbox[1]) / page_h
                    if dx <= self.position_tolerance and dy <= self.position_tolerance:
                        state.blacklisted_regions.setdefault(norm_text, []).append(curr_bbox)
                        state.banned_global_texts.add(norm_text)
                curr_page_texts[norm_text] = curr_bbox
            prev_page_texts = curr_page_texts

            # --- 圖片 ---
            seen_hashes_on_page: set[str] = set()
            for xref in _visible_image_xrefs_on_page(page):
                if xref not in state.xref_to_hash_cache:
                    img_data = doc.extract_image(xref)
                    state.xref_to_hash_cache[xref] = hashlib.md5(img_data["image"]).hexdigest()
                h = state.xref_to_hash_cache[xref]
                if h in seen_hashes_on_page:
                    continue
                seen_hashes_on_page.add(h)
                image_occurrence[h] = image_occurrence.get(h, 0) + 1

        for h, count in image_occurrence.items():
            if count / total_pages > self.image_ratio_threshold:
                state.banned_image_hashes.add(h)

        return state
