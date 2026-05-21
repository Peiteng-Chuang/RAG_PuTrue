"""Page routing：決定該頁走 fast (fitz) 或 slow (marker) 路徑。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class TriageStrategy(ABC):
    @abstractmethod
    def route(self, page) -> Literal["fast", "slow"]: ...


class DrawingCountTriage(TriageStrategy):
    """v4 等價：page.get_drawings() 數量超閾值 → slow（CAD 結構符號頁吃 Marker）。

    閾值 200 來自 [[triage-threshold-rationale]]：CAD 符號每個就上百向量。
    """

    def __init__(self, threshold: int = 200):
        self.threshold = threshold

    def route(self, page):
        return "slow" if len(page.get_drawings()) > self.threshold else "fast"


class AllFastTriage(TriageStrategy):
    """不走 marker。fast path 萬用版（測試 / 不裝 GPU 環境）。"""
    def route(self, page):
        return "fast"


class AllSlowTriage(TriageStrategy):
    """全部走 marker。給 OCR-heavy 文件用。"""
    def route(self, page):
        return "slow"
