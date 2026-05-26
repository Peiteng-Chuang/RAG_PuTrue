"""標題抽取：兩種實作並存。

- FontSizeTitleExtractor (v4 等價)：取最大字體 span 當 single title
- HierarchicalTitleExtractor (P1+P2)：抽出 page_title + subtitle + inline_headings
"""
from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections import Counter

from ..types import HeadingSpan, PageHeadings, TitleHit


# R6：縮窄 except 範圍。預期的 PyMuPDF 解析錯誤 → 印 stderr warning 並 return None；
# 其他 exception（程式 bug 等）不抓，往上拋。
_TITLE_PARSE_ERRORS = (ValueError, RuntimeError, KeyError, TypeError)


class TitleExtractor(ABC):
    @abstractmethod
    def extract(self, page) -> TitleHit | None: ...


class FontSizeTitleExtractor(TitleExtractor):
    """v4 等價 + 回傳 bbox：
    - 取所有 spans，排除過短、純數字、清單符號開頭
    - 找最大字體 group
    - **多個獨立塊 → return None**（避免目錄/並列標題誤抓）
    - 最大字體 < small_threshold → "續前頁內容"（bbox=None，合成 title）
    - 否則回 TitleHit(text, bbox=該 span 的 bbox)，給 cleaner 做 block-containment 比對
    """

    DEFAULT_LIST_PREFIXES = ("*", "-", "•", "1.", "2.")

    def __init__(
        self,
        min_text_len: int = 2,
        small_size_threshold: float = 13.0,
        continuation_text: str = "續前頁內容",
        size_tolerance: float = 0.1,
        list_prefixes: tuple[str, ...] = DEFAULT_LIST_PREFIXES,
    ):
        self.min_text_len = min_text_len
        self.small_size_threshold = small_size_threshold
        self.continuation_text = continuation_text
        self.size_tolerance = size_tolerance
        self.list_prefixes = list_prefixes

    def extract(self, page) -> TitleHit | None:
        try:
            blocks = page.get_text("dict")["blocks"]
            candidates = []
            for b in blocks:
                if "lines" not in b:
                    continue
                for line in b["lines"]:
                    for s in line["spans"]:
                        text = s["text"].strip()
                        if len(text) < self.min_text_len or text.isdigit():
                            continue
                        if text.startswith(self.list_prefixes):
                            continue
                        candidates.append({
                            "text": text,
                            "size": round(s["size"], 1),
                            "y0": s["bbox"][1],
                            "bbox": tuple(s["bbox"]),
                        })
            if not candidates:
                return None

            candidates.sort(key=lambda x: (-x["size"], x["y0"]))
            max_size = candidates[0]["size"]
            max_group = [c for c in candidates if abs(c["size"] - max_size) < self.size_tolerance]

            # 多個獨立塊 → 不設標題（避免目錄頁誤抓）
            if len(max_group) > 1:
                return None

            if max_size < self.small_size_threshold:
                return TitleHit(text=self.continuation_text, bbox=None)

            return TitleHit(text=candidates[0]["text"], bbox=candidates[0]["bbox"])
        except _TITLE_PARSE_ERRORS as e:
            page_idx = getattr(page, "number", -1)
            print(
                f"[WARN] FontSizeTitleExtractor.extract P{page_idx + 1} failed "
                f"({type(e).__name__}: {e}) — 跳過該頁 title",
                file=sys.stderr, flush=True,
            )
            return None


class HierarchicalTitleExtractor(TitleExtractor):
    """B-surgical 配套：抽出 page_title + subtitle 階層。

    流程：
    1. get_text("dict") 取全部 spans
    2. 過濾：太短、純數字、頁碼、bullet 開頭、太寬太長（視為 body 不算 heading）
    3. 排序：size desc → y0 asc
    4. 取最大字體 cluster = page_title
       - non-cover 多 candidate → return None (v4 等價，避免目錄頁誤抓)
       - cover page (idx=0) 多 candidate → 取最上面那個
    5. 取次大字體 cluster = subtitle（需在 page_title 下方 size*ratio 之內）
    6. 回傳 TitleHit + PageHeadings（backward-compat 包裝）

    P2 會擴充 inline_headings cluster；目前先空。
    """

    DEFAULT_LIST_PREFIXES = ("*", "-", "•", "1.", "2.")
    PAGE_NUM_PATTERN = re.compile(
        r"^第\s*\d+\s*頁$|^P\.?\s*\d+$|^Page\s*\d+$|^-\s*\d+\s*-$",
        re.IGNORECASE,
    )
    # 列點 bullet（必須有 trailing space），與「1.1 背景」這類章節編號區隔
    LIST_BULLET_PATTERN = re.compile(r"^(\*|-|•)\s|^\d+\.\s")

    def __init__(
        self,
        min_text_len: int = 2,
        small_size_threshold: float = 13.0,
        continuation_text: str = "續前頁內容",
        size_tolerance: float = 0.1,
        list_prefixes: tuple[str, ...] = DEFAULT_LIST_PREFIXES,
        body_text_min_chars: int = 30,
        body_text_min_width_ratio: float = 0.95,
        subtitle_max_vertical_gap_ratio: float = 5.0,
        max_inline_headings: int = 20,
        inline_heading_min_size_gap: float = 0.5,
    ):
        self.min_text_len = min_text_len
        self.small_size_threshold = small_size_threshold
        self.continuation_text = continuation_text
        self.size_tolerance = size_tolerance
        self.list_prefixes = list_prefixes
        self.body_text_min_chars = body_text_min_chars
        self.body_text_min_width_ratio = body_text_min_width_ratio
        self.subtitle_max_vertical_gap_ratio = subtitle_max_vertical_gap_ratio
        self.max_inline_headings = max_inline_headings
        self.inline_heading_min_size_gap = inline_heading_min_size_gap

    def extract(self, page) -> TitleHit | None:
        """A2：orchestrator — 串接四個 step。每個 step 可單獨單元測試。

        1. _collect_candidates: 從 page 取出 candidate spans
        2. _pick_page_title: 從 candidates 選 page_title（含小字 → 續前頁 fallback）
        3. _pick_subtitle: cover page 額外抓 subtitle（內頁回 None）
        4. _pick_inline_headings: 比 body 大且非 page_title/subtitle 的 spans
        """
        candidates = self._collect_candidates(page)
        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x["size"], x["y0"]))

        page_idx = getattr(page, "number", -1)
        is_cover = page_idx == 0

        page_title = self._pick_page_title(candidates, is_cover)
        if page_title is None:
            return None

        # 續前頁 fallback：bbox=None 表合成 title，無 subtitle / inline 概念
        if page_title.bbox is None:
            return TitleHit(
                text=page_title.text, bbox=None,
                headings=PageHeadings(page_title=page_title),
            )

        subtitle = self._pick_subtitle(candidates, page_title, is_cover)

        taken_bboxes: set = {page_title.bbox}
        if subtitle and subtitle.bbox:
            taken_bboxes.add(subtitle.bbox)
        inline_headings = self._pick_inline_headings(candidates, taken_bboxes)

        ph = PageHeadings(
            page_title=page_title, subtitle=subtitle,
            inline_headings=inline_headings,
        )
        return TitleHit(text=page_title.text, bbox=page_title.bbox, headings=ph)

    # ---- internal: 四個 step ----

    def _collect_candidates(self, page) -> list[dict]:
        """Step 1：從 page 取出 candidate span dict list。

        過濾：太短 / 純數字 / 頁碼 pattern / 列點 bullet / 寬度 ≥95% 且 ≥30 字（body 段落）。
        Exception → 印 stderr warning 並回空 list。
        """
        try:
            page_w = page.rect.width
            blocks = page.get_text("dict")["blocks"]
        except _TITLE_PARSE_ERRORS as e:
            page_idx = getattr(page, "number", -1)
            print(
                f"[WARN] HierarchicalTitleExtractor.extract P{page_idx + 1} "
                f"page.get_text failed ({type(e).__name__}: {e}) — 跳過該頁 title",
                file=sys.stderr, flush=True,
            )
            return []

        candidates: list[dict] = []
        for b in blocks:
            if b.get("type", 0) != 0:
                continue
            for line in b.get("lines", []):
                for s in line.get("spans", []):
                    text = s.get("text", "").strip()
                    if len(text) < self.min_text_len or text.isdigit():
                        continue
                    if self.PAGE_NUM_PATTERN.match(text):
                        continue
                    # 列點 bullet 用 regex（"1. text"/"* text"），不誤殺 "1.1 章節"
                    if self.LIST_BULLET_PATTERN.match(text):
                        continue
                    bbox = tuple(s.get("bbox", (0, 0, 0, 0)))
                    span_w = bbox[2] - bbox[0]
                    # 排除 body：太寬且太長
                    if (page_w > 0
                            and span_w / page_w >= self.body_text_min_width_ratio
                            and len(text) >= self.body_text_min_chars):
                        continue
                    candidates.append({
                        "text": text,
                        "size": round(s.get("size", 0), 1),
                        "y0": bbox[1],
                        "bbox": bbox,
                    })
        return candidates

    def _pick_page_title(
        self, candidates: list[dict], is_cover: bool,
    ) -> HeadingSpan | None:
        """Step 2：候選排序後選最大字體 cluster 當 page_title。

        - 整頁字都很小（< small_size_threshold）→ 回「續前頁內容」(bbox=None)
        - non-cover 且 max_group 有多個 → 回 None（避免目錄頁誤抓）
        - cover 多個 max_group → 取 y0 最上面那個
        """
        if not candidates:
            return None
        max_size = candidates[0]["size"]

        # 整頁字都很小 → 續前頁
        if max_size < self.small_size_threshold:
            return HeadingSpan(
                text=self.continuation_text, bbox=None, size=max_size,
                level=1, y0=0.0,
            )

        max_group = [
            c for c in candidates
            if abs(c["size"] - max_size) < self.size_tolerance
        ]
        if not is_cover and len(max_group) > 1:
            return None

        picked = (
            min(max_group, key=lambda c: c["y0"]) if is_cover else max_group[0]
        )
        return HeadingSpan(
            text=picked["text"], bbox=picked["bbox"],
            size=picked["size"], level=1, y0=picked["y0"],
        )

    def _pick_subtitle(
        self, candidates: list[dict], page_title: HeadingSpan, is_cover: bool,
    ) -> HeadingSpan | None:
        """Step 3：subtitle 只在 cover page 偵測（內頁的 subtitle 概念不明確；
        body 第一行容易誤判）。需在 page_title 下方 size*ratio 之內。
        """
        if not is_cover or page_title.bbox is None:
            return None
        max_size = page_title.size
        smaller = [
            c for c in candidates if c["size"] < max_size - self.size_tolerance
        ]
        if not smaller:
            return None
        second_size = smaller[0]["size"]
        second_group = [
            c for c in smaller
            if abs(c["size"] - second_size) < self.size_tolerance
        ]
        page_title_y1 = page_title.bbox[3]
        for c in second_group:
            gap = c["y0"] - page_title_y1
            if 0 <= gap <= second_size * self.subtitle_max_vertical_gap_ratio:
                return HeadingSpan(
                    text=c["text"], bbox=c["bbox"],
                    size=c["size"], level=2, y0=c["y0"],
                )
        return None

    def _pick_inline_headings(
        self, candidates: list[dict], taken_bboxes: set,
    ) -> list[HeadingSpan]:
        """Step 4：比 body (推估 size) 大但不是 page_title/subtitle 的 span。

        body_size 推估：出現 ≥2 次的最小 size（避免 Counter.most_common tie
        挑到 heading 而非 body）。fallback 用 min(size)。
        """
        if not candidates:
            return []
        size_counts = Counter(c["size"] for c in candidates)
        repeated = [s for s, n in size_counts.items() if n >= 2]
        body_size = (
            min(repeated) if repeated else min(c["size"] for c in candidates)
        )

        inline_headings: list[HeadingSpan] = []
        taken = set(taken_bboxes)
        for c in candidates:
            if c["bbox"] in taken:
                continue
            if c["size"] <= body_size + self.inline_heading_min_size_gap:
                continue
            inline_headings.append(HeadingSpan(
                text=c["text"], bbox=c["bbox"],
                size=c["size"], level=4, y0=c["y0"],
            ))
            taken.add(c["bbox"])
            if len(inline_headings) >= self.max_inline_headings:
                break
        inline_headings.sort(key=lambda h: h.y0)
        return inline_headings
