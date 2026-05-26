"""S4：PyMuPDF Page 結果快取。

⚠️ 未在 baseline 量測前**不確定值不值得**。S0 bench 要先量單頁 get_text 重複呼叫成本，
≥50ms 才有意義；若 <10ms cache 收益微小不值得引入抽象層。

設計：CachedPage wrapper 用 `__getattr__` 把所有非 cache 屬性 forward 到底層 page，
快取 `get_text(mode)` / `get_drawings()` / `get_images(full)` / `get_image_info(xrefs)`。
strategy 用 duck typing，不需要改 import 或宣告型別。

**前提**：page read-only。v5 strategy 都只讀不寫 PyMuPDF page，OK。
返回的 dict/list 共享 reference — strategy 不可 mutate（v5 都沒 mutate）。
"""
from __future__ import annotations

from typing import Any


class CachedPage:
    """PyMuPDF Page wrapper：快取常見 read-only API 結果。"""

    __slots__ = (
        "_page",
        "_text_cache", "_drawings_cache", "_drawings_cached",
        "_images_cache", "_image_info_cache",
    )

    def __init__(self, page):
        # 用 object.__setattr__ 因為 __slots__
        object.__setattr__(self, "_page", page)
        object.__setattr__(self, "_text_cache", {})
        object.__setattr__(self, "_drawings_cache", None)
        object.__setattr__(self, "_drawings_cached", False)
        object.__setattr__(self, "_images_cache", {})
        object.__setattr__(self, "_image_info_cache", {})

    def __getattr__(self, name):
        # __getattr__ 只在標準屬性查找失敗時呼叫；slot 屬性走 __getattribute__ 直接命中
        return getattr(self._page, name)

    def get_text(self, mode: str = "text", **kwargs) -> Any:
        # 只 cache 無額外 kwargs 的常見呼叫（含 mode）
        if kwargs:
            return self._page.get_text(mode, **kwargs)
        cache = self._text_cache
        if mode not in cache:
            cache[mode] = self._page.get_text(mode)
        return cache[mode]

    def get_drawings(self):
        if not self._drawings_cached:
            object.__setattr__(self, "_drawings_cache", self._page.get_drawings())
            object.__setattr__(self, "_drawings_cached", True)
        return self._drawings_cache

    def get_images(self, full: bool = False):
        cache = self._images_cache
        if full not in cache:
            cache[full] = self._page.get_images(full=full)
        return cache[full]

    def get_image_info(self, xrefs: bool = False, **kwargs):
        if kwargs:
            return self._page.get_image_info(xrefs=xrefs, **kwargs)
        cache = self._image_info_cache
        if xrefs not in cache:
            cache[xrefs] = self._page.get_image_info(xrefs=xrefs)
        return cache[xrefs]
