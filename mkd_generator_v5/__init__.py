"""mkd_generator_v5 — plugin/strategy-based ETL pipeline。

主要對外 API：
- RAGPipeline: 單檔 pipeline，由 strategies 組成
- BatchProcessor: 批次 driver
- default_pipeline(): factory，回傳 v4 行為等價的 pipeline（但 image naming 改用 _h{hash} 統一）
- ConsoleReporter / TqdmReporter / SilentReporter: 進度回報器

使用範例：
    from mkd_generator_v5 import default_pipeline, BatchProcessor, ConsoleReporter

    # 需要 marker converter（若無 GPU 環境，傳 None 則只跑 fast path）
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    marker_conv = PdfConverter(artifact_dict=create_model_dict())

    pipeline = default_pipeline(marker_converter=marker_conv)
    bp = BatchProcessor("./input", "./mkdata", pipeline=pipeline)
    bp.process_all()
"""
from .pipeline import RAGPipeline
from .batch import BatchProcessor
from .progress import ProgressReporter, ConsoleReporter, TqdmReporter, SilentReporter
from .types import (
    TitleHit, FilterState, ImageRef, ExtractContext, PageFragment,
    HeadingSpan, PageHeadings,
)
from .strategies import (
    FormatConverter, LibreOfficeConverter,
    TemplateFilter, PositionalTemplateFilter,
    TitleExtractor, FontSizeTitleExtractor, HierarchicalTitleExtractor,
    TextCleaner, BlocksTextCleaner, SpanLevelTextCleaner,
    ImageExtractor, HashNamedImageExtractor,
    TriageStrategy, DrawingCountTriage,
    MultiSignalTriage, AdaptiveTriage, AllFastTriage, AllSlowTriage,
    Stitcher, PageFragmentStitcher,
)


def default_pipeline(
    marker_converter=None,
    reporter: ProgressReporter | None = None,
    triage: TriageStrategy | None = None,
    enable_triage_log: bool = True,
    marker_pool_workers: int = 0,
    backfill_title_from_marker: bool = True,
    converter: FormatConverter | None = None,
    template_filter: TemplateFilter | None = None,
    title_extractor: TitleExtractor | None = None,
    text_cleaner: TextCleaner | None = None,
    image_extractor: ImageExtractor | None = None,
    stitcher: Stitcher | None = None,
):
    """v5 預設配置：HierarchicalTitleExtractor + B-surgical cleaner + 統一圖檔命名。

    A3：所有 strategy 都可由 caller 直接覆蓋（傳 instance）。預設 None → 用工廠預設。

    Args:
        marker_converter: marker.PdfConverter instance；None 則跑 fast path only
        reporter: ProgressReporter；None → ConsoleReporter()
        triage: TriageStrategy；None → DrawingCountTriage(200)
        enable_triage_log: 寫 per-file _triage_log/{stem}.jsonl
        marker_pool_workers: S3，>0 → process pool 跑 Marker（4090 max=2 安全）
        backfill_title_from_marker: Marker 路徑頁是否補抓 H3 為 frag.title（預設開）
        converter / template_filter / title_extractor / text_cleaner / image_extractor / stitcher:
            傳 instance 直接覆蓋預設 strategy；None → 用工廠預設
    """
    return RAGPipeline(
        converter=converter or LibreOfficeConverter(),
        template_filter=template_filter or PositionalTemplateFilter(),
        title_extractor=title_extractor or HierarchicalTitleExtractor(),
        text_cleaner=text_cleaner or SpanLevelTextCleaner(),
        image_extractor=image_extractor or HashNamedImageExtractor(),
        triage=triage or DrawingCountTriage(threshold=200),
        stitcher=stitcher or PageFragmentStitcher(),
        marker_converter=marker_converter,
        reporter=reporter or ConsoleReporter(),
        enable_triage_log=enable_triage_log,
        marker_pool_workers=marker_pool_workers,
        backfill_title_from_marker=backfill_title_from_marker,
    )


__all__ = [
    # core
    "RAGPipeline", "BatchProcessor", "default_pipeline",
    # progress
    "ProgressReporter", "ConsoleReporter", "TqdmReporter", "SilentReporter",
    # data types
    "TitleHit", "FilterState", "ImageRef", "ExtractContext", "PageFragment",
    "HeadingSpan", "PageHeadings",
    # strategies (ABC + default impl)
    "FormatConverter", "LibreOfficeConverter",
    "TemplateFilter", "PositionalTemplateFilter",
    "TitleExtractor", "FontSizeTitleExtractor", "HierarchicalTitleExtractor",
    "TextCleaner", "BlocksTextCleaner", "SpanLevelTextCleaner",
    "ImageExtractor", "HashNamedImageExtractor",
    "TriageStrategy", "DrawingCountTriage",
    "MultiSignalTriage", "AdaptiveTriage", "AllFastTriage", "AllSlowTriage",
    "Stitcher", "PageFragmentStitcher",
]
