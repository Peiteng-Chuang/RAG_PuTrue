"""文字清理：黑名單 + 重影過濾 + 標題排除。"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..types import FilterState


class TextCleaner(ABC):
    @abstractmethod
    def clean(self, page, page_idx: int, title: str | None,
              filter_state: FilterState) -> str: ...


class BlocksTextCleaner(TextCleaner):
    """v4 等價，三層過濾：
    1. 黑名單位置（template scan 偵測到的雜訊）
    2. 同頁重影（same text 在 ±3pt 範圍）
    3. 已當標題的 text（避免標題在 header + 內文重複）
    """

    def __init__(
        self,
        position_tolerance: float = 0.1,
        dedup_tolerance_pt: float = 3.0,
    ):
        self.position_tolerance = position_tolerance
        self.dedup_tolerance_pt = dedup_tolerance_pt

    def clean(self, page, page_idx, title, filter_state) -> str:
        page_w, page_h = page.rect.width, page.rect.height
        norm_title = re.sub(r"\s+", "", title) if title else None
        seen_in_page: dict[str, tuple[float, float]] = {}
        unique_texts: list[str] = []

        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _block_no, _block_type = b
            norm_text = re.sub(r"\s+", "", text.strip())
            if not norm_text:
                continue

            # 1. 黑名單排除
            if norm_text in filter_state.blacklisted_regions:
                banned = False
                for b_box in filter_state.blacklisted_regions[norm_text]:
                    if (abs(x0 - b_box[0]) / page_w <= self.position_tolerance and
                            abs(y0 - b_box[1]) / page_h <= self.position_tolerance):
                        banned = True
                        break
                if banned:
                    continue

            # 2. 重影過濾
            if norm_text in seen_in_page:
                px, py = seen_in_page[norm_text]
                if abs(x0 - px) < self.dedup_tolerance_pt and abs(y0 - py) < self.dedup_tolerance_pt:
                    continue
            seen_in_page[norm_text] = (x0, y0)

            # 3. 標題排除
            if norm_title and norm_text == norm_title:
                continue

            unique_texts.append(text.strip())

        return "\n".join(unique_texts)
