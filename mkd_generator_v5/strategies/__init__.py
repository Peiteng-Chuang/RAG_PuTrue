"""Pipeline strategies — 每個 ABC 各自獨立檔案。

對外曝光 ABC + default 實作。"""
from .converter import FormatConverter, LibreOfficeConverter
from .template import TemplateFilter, PositionalTemplateFilter
from .title import TitleExtractor, FontSizeTitleExtractor
from .text_cleaner import TextCleaner, BlocksTextCleaner
from .image import ImageExtractor, HashNamedImageExtractor
from .triage import TriageStrategy, DrawingCountTriage
from .stitcher import Stitcher, PageFragmentStitcher

__all__ = [
    "FormatConverter", "LibreOfficeConverter",
    "TemplateFilter", "PositionalTemplateFilter",
    "TitleExtractor", "FontSizeTitleExtractor",
    "TextCleaner", "BlocksTextCleaner",
    "ImageExtractor", "HashNamedImageExtractor",
    "TriageStrategy", "DrawingCountTriage",
    "Stitcher", "PageFragmentStitcher",
]
