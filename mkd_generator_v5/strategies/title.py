"""標題抽取：最大字體當 h3。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TitleExtractor(ABC):
    @abstractmethod
    def extract(self, page) -> str | None: ...


class FontSizeTitleExtractor(TitleExtractor):
    """v4 等價：
    - 取所有 spans，排除過短、純數字、清單符號開頭
    - 找最大字體 group
    - **多個獨立塊 → return None**（避免目錄/並列標題誤抓）
    - 最大字體 < small_threshold → "續前頁內容"
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

    def extract(self, page) -> str | None:
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
                return self.continuation_text

            return candidates[0]["text"]
        except Exception:
            return None
