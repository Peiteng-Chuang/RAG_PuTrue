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
from .strategies import (
    FormatConverter, LibreOfficeConverter,
    TemplateFilter, PositionalTemplateFilter,
    TitleExtractor, FontSizeTitleExtractor,
    TextCleaner, BlocksTextCleaner,
    ImageExtractor, HashNamedImageExtractor,
    TriageStrategy, DrawingCountTriage,
    Stitcher, PageFragmentStitcher,
)


def default_pipeline(
    marker_converter=None,
    reporter: ProgressReporter | None = None,
):
    """v4 行為等價 + W5/C 圖檔命名統一。"""
    return RAGPipeline(
        converter=LibreOfficeConverter(),
        template_filter=PositionalTemplateFilter(),
        title_extractor=FontSizeTitleExtractor(),
        text_cleaner=BlocksTextCleaner(),
        image_extractor=HashNamedImageExtractor(),
        triage=DrawingCountTriage(threshold=200),
        stitcher=PageFragmentStitcher(),
        marker_converter=marker_converter,
        reporter=reporter or ConsoleReporter(),
    )


__all__ = [
    # core
    "RAGPipeline", "BatchProcessor", "default_pipeline",
    # progress
    "ProgressReporter", "ConsoleReporter", "TqdmReporter", "SilentReporter",
    # strategies (ABC + default impl)
    "FormatConverter", "LibreOfficeConverter",
    "TemplateFilter", "PositionalTemplateFilter",
    "TitleExtractor", "FontSizeTitleExtractor",
    "TextCleaner", "BlocksTextCleaner",
    "ImageExtractor", "HashNamedImageExtractor",
    "TriageStrategy", "DrawingCountTriage",
    "Stitcher", "PageFragmentStitcher",
]
