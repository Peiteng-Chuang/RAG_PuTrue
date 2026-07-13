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
    xref_to_hash_cache: dict[int, str | None] = field(default_factory=dict)
    # None = bad xref (PyMuPDF extract_image raise ValueError)，下游看到直接 skip 不重試
    warnings: list[str] = field(default_factory=list)
    # P1: template scan 階段累積的 warnings，pipeline 在 FITZ_SCAN_DONE 後統一 emit reporter.warning
    black_image_hashes: set[str] = field(default_factory=set)
    # W6：確認為「全黑/全透明」的無效圖 hash（PPT→LibreOffice→PDF 剝 SMask 後的黑塊）。
    # 一旦某 hash 判為黑就 cache，跨頁/跨 fast|slow 路徑不再重複解碼判定。
    black_image_skipped: int = 0
    # W6：本檔累積跳過的黑圖「出現次數」（含跨頁重複出現），供 FileStats 顯示。


@dataclass
class HeadingSpan:
    """單一 heading 的位置與字級資訊。"""
    text: str
    bbox: tuple[float, float, float, float] | None
    size: float                                   # font size, 四捨五入到 1 位
    level: int                                    # 1=page_title, 2=subtitle, 3=inline H3, 4=inline H4
    y0: float                                     # 排序用


@dataclass
class PageHeadings:
    """HierarchicalTitleExtractor 輸出。包整頁所有 heading 階層。"""
    page_title: HeadingSpan | None = None
    subtitle: HeadingSpan | None = None
    inline_headings: list[HeadingSpan] = field(default_factory=list)  # P2 才填

    @property
    def all_title_bboxes(self) -> list[tuple[float, float, float, float]]:
        """cleaner 要排除的全部 heading span bbox。"""
        out: list[tuple[float, float, float, float]] = []
        if self.page_title and self.page_title.bbox:
            out.append(self.page_title.bbox)
        if self.subtitle and self.subtitle.bbox:
            out.append(self.subtitle.bbox)
        for h in self.inline_headings:
            if h.bbox:
                out.append(h.bbox)
        return out


@dataclass
class TitleHit:
    """TitleExtractor.extract() 輸出。

    bbox 是被選為 page_title 的 span bbox（x0, y0, x1, y1），給 cleaner 做
    「title span」判斷用。合成 title（如 "續前頁內容"）bbox=None。
    headings 為 HierarchicalTitleExtractor 額外帶的完整階層；舊 extractor 留 None。
    """
    text: str
    bbox: tuple[float, float, float, float] | None = None
    headings: PageHeadings | None = None


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
    black_threshold: int = 8               # W6：像素各通道最大值 <= 此值即判「全黑」（0-255）


@dataclass
class PageFragment:
    """單頁產出。pipeline 收集成 list，最後丟給 Stitcher。

    body 可能含 placeholder（如 `[[MARKER_REPLACE_PN]]`），由 marker stage 替換。
    階層約定：`##` = page anchor、`###` = title (page_title)、`####` = subtitle 與 inline。
    """
    page_idx: int                          # 0-based
    page_num: int                          # 1-based（= page_idx + 1）
    title: str | None
    body: str
    is_marker_page: bool
    marker_placeholder: str | None = None  # 慢路徑用，標示要被替換的 anchor
    subtitle: str | None = None            # P2: 副標，渲染為緊接 ### title 下的 ####

    def render(self) -> str:
        """組裝為 MD 區塊。慢路徑頁的 body 仍含 placeholder 時，render 不展開。"""
        header = f"## 第 {self.page_num} 頁\n"
        if self.title:
            header += f"### {self.title}\n"
            if self.subtitle:
                header += f"#### {self.subtitle}\n"
            header += "\n"
        else:
            header += "\n"
        return f"{header}{self.body}\n---\n"


@dataclass
class FileStats:
    """P2：單檔處理結果濃縮統計。pipeline.run() 結尾 attach 到 reporter."""
    file_name: str
    total_pages: int = 0
    fast_pages: int = 0
    slow_pages: int = 0
    bad_xref_count: int = 0          # template scan 偵測到的 bad xref 張數
    triage_fallback_count: int = 0    # triage.route 失敗 fallback fast 的頁數
    marker_unresolved_pages: int = 0  # Marker 未解析、填註記的頁數（避免 placeholder 洩漏）
    black_image_skipped: int = 0      # W6：跳過的全黑/全透明無效圖出現次數
    warning_count: int = 0            # 本檔累積 warning 總數

    def summary_line(self) -> str:
        """印一行濃縮（reporter ConsoleReporter 用）。"""
        parts = [
            f"📊 {self.total_pages} pages",
            f"fast={self.fast_pages}/slow={self.slow_pages}",
        ]
        if self.bad_xref_count:
            parts.append(f"bad_xref={self.bad_xref_count}")
        if self.triage_fallback_count:
            parts.append(f"triage_fallback={self.triage_fallback_count}")
        if self.marker_unresolved_pages:
            parts.append(f"marker_unresolved={self.marker_unresolved_pages}")
        if self.black_image_skipped:
            parts.append(f"black_img_skipped={self.black_image_skipped}")
        if self.warning_count:
            parts.append(f"warnings={self.warning_count}")
        return " · ".join(parts)


@dataclass
class MarkerOutput:
    """Marker 跑完單頁的原始輸出。"""
    page_idx: int
    text: str                              # markdown 主體
    images: dict[str, Any]                 # marker 原始 image map：orig_name → bytes / PIL Image
