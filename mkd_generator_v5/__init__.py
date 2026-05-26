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
):
    """v5 預設配置：HierarchicalTitleExtractor + B-surgical cleaner + 統一圖檔命名。

    triage：可注入自訂策略，預設 DrawingCountTriage(200)
    enable_triage_log：寫 per-file _triage_log/{stem}.jsonl，方便事後 audit
    marker_pool_workers：S3，>0 → process pool 平行跑 Marker，**4090 上 max=2 安全**
    """
    return RAGPipeline(
        converter=LibreOfficeConverter(),
        template_filter=PositionalTemplateFilter(),
        title_extractor=HierarchicalTitleExtractor(),
        text_cleaner=SpanLevelTextCleaner(),
        image_extractor=HashNamedImageExtractor(),
        triage=triage or DrawingCountTriage(threshold=200),
        stitcher=PageFragmentStitcher(),
        marker_converter=marker_converter,
        reporter=reporter or ConsoleReporter(),
        enable_triage_log=enable_triage_log,
        marker_pool_workers=marker_pool_workers,
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
