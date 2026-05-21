"""共用 dataclass 與型別。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FilterState:
    """TemplateFilter.scan() 輸出。

    blacklisted_regions: text(去空白) → 多個出現位置 bbox [x0,y0,x1,y1]
    banned_global_texts: 出現在多頁的文字 set（norm 後）
    banned_image_hashes: 出現比例超過閾值的圖片 hash set
    xref_to_hash_cache: image xref → md5 hash，避免重算
    """
    blacklisted_regions: dict[str, list[list[float]]] = field(default_factory=dict)
    banned_global_texts: set[str] = field(default_factory=set)
    banned_image_hashes: set[str] = field(default_factory=set)
    xref_to_hash_cache: dict[int, str] = field(default_factory=dict)


@dataclass
class ImageRef:
    """ImageExtractor 輸出。"""
    md_path: str          # MD 內 ![](...) 的 path 部分，例如 `folder/file.jpeg`
    abs_path: Path        # 實體位置
    img_hash: str         # md5 hex（完整 32 字元；命名取前 8 碼）

    def to_md_line(self) -> str:
        """MD 內單行 image 引用（含換行）。"""
        return f"![{self.abs_path.name}]({self.md_path})\n"


@dataclass
class ExtractContext:
    """傳給 ImageExtractor 的 per-file 環境。

    saved_image_hashes / hash_to_filename 是跨頁、跨 fast/slow 路徑共享的 state，
    pipeline 持有同一份 dict 傳入，extractor 直接 mutate。
    """
    stem: str                              # 原檔 stem，用於命名
    img_dir: Path                          # 輸出 image 資料夾絕對路徑
    folder_name: str                       # img_dir.name，用於 md ref 相對路徑
    filter_state: FilterState
    saved_image_hashes: set[str]
    hash_to_filename: dict[str, str]
    doc: Any                               # fitz.Document，fast 路徑要 doc.extract_image(xref)
    min_image_dim: int = 60                # 過濾過小圖片的閾值


@dataclass
class PageFragment:
    """單頁產出。pipeline 收集成 list，最後丟給 Stitcher。

    body 可能含 placeholder（如 `[[MARKER_REPLACE_PN]]`），由 marker stage 替換。
    """
    page_idx: int                          # 0-based
    page_num: int                          # 1-based（= page_idx + 1）
    title: str | None
    body: str
    is_marker_page: bool
    marker_placeholder: str | None = None  # 慢路徑用，標示要被替換的 anchor

    def render(self) -> str:
        """組裝為 MD 區塊。慢路徑頁的 body 仍含 placeholder 時，render 不展開。"""
        header = f"## 第 {self.page_num} 頁\n"
        if self.title:
            header += f"### {self.title}\n\n"
        else:
            header += "\n"
        return f"{header}{self.body}\n---\n"


@dataclass
class MarkerOutput:
    """Marker 跑完單頁的原始輸出。"""
    page_idx: int
    text: str                              # markdown 主體
    images: dict[str, Any]                 # marker 原始 image map：orig_name → bytes / PIL Image
