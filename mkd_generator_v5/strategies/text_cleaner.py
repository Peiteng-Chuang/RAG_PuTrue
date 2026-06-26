"""文字清理：黑名單 + 重影過濾 + 標題排除。

兩種實作並存：
- BlocksTextCleaner（v4 等價）：block 級雙條件丟整塊，副標會被誤殺
- SpanLevelTextCleaner（B-surgical）：span 級比對，只丟 title span 本身
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..types import FilterState, TitleHit


class TextCleaner(ABC):
    @abstractmethod
    def clean(self, page, page_idx: int, title: TitleHit | None,
              filter_state: FilterState) -> str: ...

    def clean_structured(self, page, page_idx: int, title: TitleHit | None,
                         filter_state: FilterState) -> list[tuple[float, str]]:
        """回 [(y0, text), ...]，給 pipeline 與 inline_headings merge 用。

        預設實作從 clean() 推導，每行假 y0 = index。
        SpanLevelTextCleaner override 提供真實 y0（用 block_bbox y0）。
        """
        text = self.clean(page, page_idx, title, filter_state)
        return [
            (float(i), line)
            for i, line in enumerate(text.split("\n"))
            if line.strip()
        ]


_PAGE_NUM_PATTERNS = [
    re.compile(r"^\d{1,4}$"),                       # 3
    re.compile(r"^\d{1,4}[/／]\d{1,4}$"),           # 3/52
    re.compile(r"^第?\d{1,4}[頁页]$"),              # 第3頁 / 3頁
    re.compile(r"^[-–—]\d{1,4}[-–—]$"),            # - 3 -
    re.compile(r"^[Pp](?:age)?\.?\d{1,4}$"),       # P3 / Page3 / P.3
]


def _is_page_number_noise(
    text: str, y0: float, y1: float, page_h: float, margin_ratio: float = 0.08,
) -> bool:
    """頁碼/頁尾雜訊判定：**位於上/下邊界帶** 且 **符合頁碼樣式**（雙條件才剃）。

    PPT→PDF 後頁碼/頁尾是每頁變動的文字（1、2、3…／3/52），模板過濾靠「內容相同」
    比對抓不到。這裡改用「位置（邊界帶）+ 樣式」抓——正文不會只是『3』又落在頁尾，
    雙條件可避免誤刪。"""
    s = text.strip()
    if not s or len(s) > 16:
        return False
    in_band = (y1 <= page_h * margin_ratio) or (y0 >= page_h * (1 - margin_ratio))
    if not in_band:
        return False
    compact = re.sub(r"\s+", "", s)
    return any(p.match(compact) for p in _PAGE_NUM_PATTERNS)


def _bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float] | None,
    tolerance: float = 1.0,
) -> bool:
    """outer 是否（含 tolerance）包住 inner。inner=None 一律回 False。"""
    if inner is None:
        return False
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        ox0 - tolerance <= ix0
        and oy0 - tolerance <= iy0
        and ox1 + tolerance >= ix1
        and oy1 + tolerance >= iy1
    )


def _is_title_span(
    span_bbox: tuple[float, float, float, float],
    title_bbox: tuple[float, float, float, float] | None,
    tolerance: float = 1.0,
) -> bool:
    """span 是否就是 title 那個 span。

    title 從 get_text("dict") spans 取出，cleaner 也用 dict 時 bbox 應幾乎吻合。
    雙向 contains + tolerance 容錯 PyMuPDF 浮點微差。
    """
    if title_bbox is None:
        return False
    return (
        _bbox_contains(title_bbox, span_bbox, tolerance)
        or _bbox_contains(span_bbox, title_bbox, tolerance)
    )


class BlocksTextCleaner(TextCleaner):
    """v4 等價 + bbox 雙條件，三層過濾：
    1. 黑名單位置（template scan 偵測到的雜訊）
    2. 同頁重影（same text 在 ±3pt 範圍）
    3. 標題排除（避免標題在 header + 內文重複）。兩條規則任一命中即 skip：
       a. norm_text == norm_title（v4 原行為，title block 剛好只含 title 時生效）
       b. title.bbox 被 block.bbox 包住 **且** norm_text 含 norm_title
          → 處理「PyMuPDF 把 title span 跟其他 span 合進同 block」的常見情境
          （PPT→LibreOffice→PDF 經常發生）
    """

    def __init__(
        self,
        position_tolerance: float = 0.1,
        dedup_tolerance_pt: float = 3.0,
        bbox_tolerance_pt: float = 1.0,
    ):
        self.position_tolerance = position_tolerance
        self.dedup_tolerance_pt = dedup_tolerance_pt
        self.bbox_tolerance_pt = bbox_tolerance_pt

    def clean(self, page, page_idx, title: TitleHit | None, filter_state) -> str:
        page_w, page_h = page.rect.width, page.rect.height
        norm_title = re.sub(r"\s+", "", title.text) if title else None
        title_bbox = title.bbox if title else None
        seen_in_page: dict[str, tuple[float, float]] = {}
        unique_texts: list[str] = []

        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _block_no, _block_type = b
            block_bbox = (x0, y0, x1, y1)
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

            # 3. 標題排除（雙條件，任一命中即 skip）
            if norm_title:
                # (a) 完全相等：title block 剛好只裝 title
                if norm_text == norm_title:
                    continue
                # (b) bbox 包住 + 文字含 title：title span 被合進更大的 block
                if (_bbox_contains(block_bbox, title_bbox, self.bbox_tolerance_pt)
                        and norm_title in norm_text):
                    continue

            unique_texts.append(text.strip())

        return "\n".join(unique_texts)


class SpanLevelTextCleaner(TextCleaner):
    """B-surgical：用 get_text("dict") span 級比對 title.bbox，只丟 title 那個 span。

    解決 BlocksTextCleaner 在「title span + subtitle span 合在同一 block」時，
    因 block 粒度過粗導致副標被一起丟掉的問題。

    流程：
    1. 走訪 block → line → span
    2. 標記每個 span 是否 = title span（bbox 雙向 contains + tolerance）
    3. 黑名單比對：用「原始 block 文字」（含 title）為 key，因 template scan 用 blocks 粒度
       blacklisted_regions 的 key 是含 title 的整塊 norm_text。若 norm 對不上，退到位置 fallback
    4. 重組 block 文字（line 級 \\n，line 內 span concat），輸出
    """

    def __init__(
        self,
        position_tolerance: float = 0.1,
        dedup_tolerance_pt: float = 5.0,  # 從 B-strict 的 3 放寬到 5（span 重組後位置可能微移）
        bbox_tolerance_pt: float = 1.0,
        strip_page_numbers: bool = True,   # 剃除頁尾/頁首的頁碼類雜訊
        page_number_margin_ratio: float = 0.08,
    ):
        self.position_tolerance = position_tolerance
        self.dedup_tolerance_pt = dedup_tolerance_pt
        self.bbox_tolerance_pt = bbox_tolerance_pt
        self.strip_page_numbers = strip_page_numbers
        self.page_number_margin_ratio = page_number_margin_ratio

    def _collect_title_bboxes(self, title: TitleHit | None) -> list[tuple]:
        """彙整 page_title + subtitle + inline_headings 全部要 skip 的 bbox。"""
        result: list[tuple] = []
        if title and title.bbox:
            result.append(title.bbox)
        if title and title.headings:
            for bbox in title.headings.all_title_bboxes:
                if bbox not in result:
                    result.append(bbox)
        return result

    def clean_structured(
        self, page, page_idx, title: TitleHit | None, filter_state
    ) -> list[tuple[float, str]]:
        """回 [(block_y0, block_text), ...]，給 pipeline merge inline_headings 用。"""
        page_w, page_h = page.rect.width, page.rect.height
        title_bboxes = self._collect_title_bboxes(title)
        norm_title = re.sub(r"\s+", "", title.text) if title else None

        seen_in_page: dict[str, tuple[float, float]] = {}
        results: list[tuple[float, str]] = []

        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception:
            return results

        for b in blocks:
            if b.get("type", 0) != 0:  # 0=text block, 1=image
                continue
            if "lines" not in b:
                continue

            bbox = b.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = bbox

            # 收集 lines（保留結構）；分別記 含 title / 不含 title 兩版
            original_text_lines: list[str] = []
            kept_text_lines: list[str] = []
            for line in b["lines"]:
                orig_parts: list[str] = []
                kept_parts: list[str] = []
                for s in line.get("spans", []):
                    text = s.get("text", "")
                    if not text:
                        continue
                    orig_parts.append(text)
                    span_bbox = tuple(s.get("bbox", (0, 0, 0, 0)))
                    if any(
                        _is_title_span(span_bbox, tb, self.bbox_tolerance_pt)
                        for tb in title_bboxes
                    ):
                        continue
                    kept_parts.append(text)
                if orig_parts:
                    original_text_lines.append("".join(orig_parts))
                if kept_parts:
                    kept_text_lines.append("".join(kept_parts))

            if not original_text_lines:
                continue

            # 1. 黑名單比對：先用 含 title 的原始文字（對齊 template scan 用 blocks 看到的）
            original_norm = re.sub(r"\s+", "", "".join(original_text_lines))
            banned = False
            if original_norm in filter_state.blacklisted_regions:
                for b_box in filter_state.blacklisted_regions[original_norm]:
                    if (abs(x0 - b_box[0]) / page_w <= self.position_tolerance and
                            abs(y0 - b_box[1]) / page_h <= self.position_tolerance):
                        banned = True
                        break

            # 1b. Fallback：blocks↔dict 邊界切法不一致時的補抓
            #
            # PyMuPDF 對「左側細長 vertical text 條」這類版面，blocks 模式常會把整條當
            # 一個 block，dict 模式卻拆成多個 sub-block（典型例：CAD/結構簡報的
            # KAICHU 浮水印 + 頁首地號合在一個垂直區）。
            #
            # 結果：template scan（用 blocks）登錄的 key 是長串，cleaner（用 dict）
            # 拼出來的 key 是其中一段，相等比對永遠 miss → 黑名單失效，每頁漏刪。
            #
            # 條件三件必滿足才視為命中，避免正文偶然撞到 banner 字片段被誤殺：
            #   (a) original_norm 是 banned_key 的真子字串（且長度 >= 3）
            #   (b) 當前 dict block 的 bbox 被 banned_key 對應的 banner bbox 包住
            #   (c) 已經 contained，毋須再看 position_tolerance 偏移
            if not banned and len(original_norm) >= 3:
                tol = self.bbox_tolerance_pt
                for banned_key, banned_bboxes in filter_state.blacklisted_regions.items():
                    if banned_key == original_norm or original_norm not in banned_key:
                        continue
                    for b_box in banned_bboxes:
                        bx0, by0, bx1, by1 = b_box
                        if (bx0 - tol <= x0 and by0 - tol <= y0
                                and bx1 + tol >= x1 and by1 + tol >= y1):
                            banned = True
                            break
                    if banned:
                        break

            if banned:
                continue

            # title 全部丟光 → 此 block 內容空，整塊 skip
            if not kept_text_lines:
                continue

            block_text = "\n".join(kept_text_lines).strip()
            norm_text = re.sub(r"\s+", "", block_text)
            if not norm_text:
                continue

            # 2. 重影過濾（同頁同文字 ±5pt）
            if norm_text in seen_in_page:
                px, py = seen_in_page[norm_text]
                if abs(x0 - px) < self.dedup_tolerance_pt and abs(y0 - py) < self.dedup_tolerance_pt:
                    continue
            seen_in_page[norm_text] = (x0, y0)

            # 3. 標題殘留排除（剝完 title span 後內容若剛好 == title，少數邊界情境）
            if norm_title and norm_text == norm_title:
                continue

            # 4. 頁碼/頁尾雜訊剃除（邊界帶 + 頁碼樣式雙條件）
            if self.strip_page_numbers and _is_page_number_noise(
                block_text, y0, y1, page_h, self.page_number_margin_ratio,
            ):
                continue

            results.append((y0, block_text))

        return results

    def clean(self, page, page_idx, title: TitleHit | None, filter_state) -> str:
        return "\n".join(
            t for _, t in self.clean_structured(page, page_idx, title, filter_state)
        )
