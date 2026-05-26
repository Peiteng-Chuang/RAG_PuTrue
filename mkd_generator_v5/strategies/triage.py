"""Page routing：決定該頁走 fast (fitz) 或 slow (marker) 路徑。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class TriageStrategy(ABC):
    @abstractmethod
    def route(self, page) -> Literal["fast", "slow"]: ...

    def prepare(self, doc) -> None:
        """Optional：跑單頁 loop 前先看整份 doc，給需要 whole-doc 統計的策略用。"""
        return None

    def explain(self, page) -> dict:
        """Optional：回 route() 用到的訊號（給 decision log 寫入用）。"""
        return {}


class DrawingCountTriage(TriageStrategy):
    """v4 等價：page.get_drawings() 數量超閾值 → slow（CAD 結構符號頁吃 Marker）。

    閾值 200 來自 [[triage-threshold-rationale]]：CAD 符號每個就上百向量。
    """

    def __init__(self, threshold: int = 200):
        self.threshold = threshold

    def route(self, page):
        return "slow" if len(page.get_drawings()) > self.threshold else "fast"

    def explain(self, page) -> dict:
        try:
            n = len(page.get_drawings())
        except Exception:
            n = -1
        return {"drawings": n, "threshold": self.threshold,
                "strategy": "DrawingCountTriage"}


class MultiSignalTriage(TriageStrategy):
    """S1：加權多訊號分數判斷。

    score = w_d * drawings + w_i * images + w_t * text_density + w_f * font_diversity
    score > threshold → slow

    - text_density = 字數 / 頁面面積（高文字密度頁通常 fast 即可）
    - font_diversity = 不同字級數量（表格頁通常 font_diversity 高）
    - 預設 w_text_density=0、w_font_diversity=0 → 行為等價於只看 drawings + images

    調整建議：對表格密集 PDF 開大 w_font_diversity；對 OCR PDF 開大 w_images。
    """

    def __init__(
        self,
        threshold: float = 200.0,
        w_drawings: float = 1.0,
        w_images: float = 5.0,
        w_text_density: float = 0.0,
        w_font_diversity: float = 0.0,
    ):
        self.threshold = threshold
        self.w_drawings = w_drawings
        self.w_images = w_images
        self.w_text_density = w_text_density
        self.w_font_diversity = w_font_diversity
        self._last_signals: dict = {}

    def _compute(self, page) -> dict:
        try:
            n_draw = len(page.get_drawings())
        except Exception:
            n_draw = 0
        try:
            n_img = len(page.get_images(full=False))
        except Exception:
            n_img = 0
        chars = 0
        sizes: set[float] = set()
        try:
            for b in page.get_text("dict").get("blocks", []):
                if "lines" not in b:
                    continue
                for line in b["lines"]:
                    for s in line.get("spans", []):
                        chars += len(s.get("text", ""))
                        sz = round(s.get("size", 0), 1)
                        if sz > 0:
                            sizes.add(sz)
            area = max(1.0, page.rect.width * page.rect.height)
        except Exception:
            area = 1.0
        text_density = chars / area
        score = (self.w_drawings * n_draw
                 + self.w_images * n_img
                 + self.w_text_density * text_density
                 + self.w_font_diversity * len(sizes))
        route = "slow" if score > self.threshold else "fast"
        return {
            "drawings": n_draw, "images": n_img,
            "text_density": round(text_density, 4),
            "font_diversity": len(sizes),
            "score": round(score, 2),
            "threshold": self.threshold,
            "route": route,
            "strategy": "MultiSignalTriage",
        }

    def route(self, page):
        sig = self._compute(page)
        self._last_signals = sig
        return sig["route"]

    def explain(self, page) -> dict:
        # 若 route() 剛跑過、_last_signals 是這頁的，省一次計算
        if self._last_signals and self._last_signals.get("page_id") == id(page):
            return dict(self._last_signals)
        sig = self._compute(page)
        sig["page_id"] = id(page)
        self._last_signals = sig
        return dict(sig)


class AdaptiveTriage(TriageStrategy):
    """S1：先掃整份 doc 取 drawings 分位數，動態決定閾值。

    閾值範圍卡在 [min_threshold, max_threshold] 之間。
    fallback：若 prepare() 未呼叫 → 用 fixed_fallback_threshold（預設 200，跟 v4 對齊）
    """

    def __init__(
        self,
        percentile: float = 80.0,
        min_threshold: float = 50.0,
        max_threshold: float = 500.0,
        fixed_fallback_threshold: float = 200.0,
    ):
        self.percentile = percentile
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.fixed_fallback_threshold = fixed_fallback_threshold
        self._threshold: float | None = None
        self._sample_count = 0

    def prepare(self, doc) -> None:
        counts: list[int] = []
        for i in range(len(doc)):
            try:
                counts.append(len(doc[i].get_drawings()))
            except Exception:
                counts.append(0)
        if not counts:
            self._threshold = self.fixed_fallback_threshold
            return
        sorted_counts = sorted(counts)
        idx = max(0, min(len(sorted_counts) - 1,
                         int(len(sorted_counts) * self.percentile / 100)))
        p = float(sorted_counts[idx])
        self._threshold = max(self.min_threshold, min(self.max_threshold, p))
        self._sample_count = len(counts)

    def _current_threshold(self) -> float:
        return (self._threshold
                if self._threshold is not None
                else self.fixed_fallback_threshold)

    def route(self, page):
        try:
            n = len(page.get_drawings())
        except Exception:
            n = 0
        return "slow" if n > self._current_threshold() else "fast"

    def explain(self, page) -> dict:
        try:
            n = len(page.get_drawings())
        except Exception:
            n = -1
        return {
            "drawings": n,
            "threshold": self._current_threshold(),
            "percentile": self.percentile,
            "sample_count": self._sample_count,
            "strategy": "AdaptiveTriage",
        }


class AllFastTriage(TriageStrategy):
    """不走 marker。fast path 萬用版（測試 / 不裝 GPU 環境）。"""
    def route(self, page):
        return "fast"

    def explain(self, page) -> dict:
        return {"strategy": "AllFastTriage"}


class AllSlowTriage(TriageStrategy):
    """全部走 marker。給 OCR-heavy 文件用。"""
    def route(self, page):
        return "slow"

    def explain(self, page) -> dict:
        return {"strategy": "AllSlowTriage"}
