"""RAGPipeline orchestrator — 純編排，不含演算法。

設計：
- 持有一組 strategy（converter / template_filter / title / text_cleaner /
  image_extractor / triage / stitcher）+ marker_converter + reporter
- run(input_path, output_base) 跑單一檔案
- 等價於 v4 RAGSmartPipeline，但行為由 strategy 決定
"""
from __future__ import annotations

import gc
import hashlib
import shutil
from pathlib import Path
from typing import Any

import fitz

from .progress import ProgressReporter, SilentReporter
from .strategies import (
    FormatConverter, TemplateFilter, TitleExtractor, TextCleaner,
    ImageExtractor, TriageStrategy, Stitcher,
)
from .types import ExtractContext, PageFragment

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)


def _file_md5(path: Path, chunk: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            hasher.update(buf)
    return hasher.hexdigest()


class RAGPipeline:
    """單檔 ETL pipeline，由 strategies 組成。"""

    def __init__(
        self,
        *,
        converter: FormatConverter,
        template_filter: TemplateFilter,
        title_extractor: TitleExtractor,
        text_cleaner: TextCleaner,
        image_extractor: ImageExtractor,
        triage: TriageStrategy,
        stitcher: Stitcher,
        marker_converter: Any = None,           # 可 None（無 marker 環境）
        reporter: ProgressReporter | None = None,
        tmp_dir: Path | None = None,
        backfill_title_from_marker: bool = True,  # marker 補抓 h3
    ):
        self.converter = converter
        self.template_filter = template_filter
        self.title_extractor = title_extractor
        self.text_cleaner = text_cleaner
        self.image_extractor = image_extractor
        self.triage = triage
        self.stitcher = stitcher
        self.marker_converter = marker_converter
        self.reporter = reporter or SilentReporter()
        self.tmp_dir = (tmp_dir or Path("./_process_tmp")).absolute()
        self.backfill_title_from_marker = backfill_title_from_marker

    # ---- public ----

    def run(self, input_path: Path | str, output_base: Path | str) -> bool:
        input_path = Path(input_path).absolute()
        output_base = Path(output_base).absolute()
        stem = input_path.stem
        img_dir = output_base / f"{stem}_image"
        md_path = output_base / f"{stem}.md"

        try:
            self._prepare_env(output_base, img_dir)

            # Phase 1: convert
            ext = input_path.suffix.lower()
            if self.converter.supports(ext):
                self.reporter.phase("CONVERTING_FORMAT", input_path.name)
                working_pdf = self.converter.convert(input_path, self.tmp_dir)
            else:
                working_pdf = input_path

            # Phase 2: open + template scan
            self.reporter.phase("FITZ_SCANNING")
            doc = fitz.open(working_pdf)
            filter_state = self.template_filter.scan(doc)
            self.reporter.phase(
                "FITZ_SCAN_DONE",
                f"banned text={len(filter_state.banned_global_texts)}, "
                f"banned img={len(filter_state.banned_image_hashes)}",
            )

            # 主標題（取第一頁）
            first_title = self.title_extractor.extract(doc[0]) if len(doc) > 0 else None
            main_title = first_title or "未命名簡報"

            # Phase 3: per-page triage + fast path
            ctx = ExtractContext(
                stem=stem,
                img_dir=img_dir,
                folder_name=img_dir.name,
                filter_state=filter_state,
                saved_image_hashes=set(),
                hash_to_filename={},
                doc=doc,
            )
            fragments: list[PageFragment] = []
            vector_indices: list[int] = []
            total_pages = len(doc)
            self.reporter.phase("PER_PAGE", f"{total_pages} pages")
            for page_idx in range(total_pages):
                page = doc[page_idx]
                title = self.title_extractor.extract(page)
                route = self.triage.route(page)
                if route == "slow":
                    placeholder = f"[[MARKER_REPLACE_P{page_idx}]]"
                    fragments.append(PageFragment(
                        page_idx=page_idx,
                        page_num=page_idx + 1,
                        title=title,
                        body=f"{placeholder}\n",
                        is_marker_page=True,
                        marker_placeholder=placeholder,
                    ))
                    vector_indices.append(page_idx)
                else:
                    text = self.text_cleaner.clean(page, page_idx, title, filter_state)
                    image_refs = self.image_extractor.extract_fast(page, ctx)
                    body = text + "\n\n"
                    for ref in image_refs:
                        body += ref.to_md_line()
                    fragments.append(PageFragment(
                        page_idx=page_idx,
                        page_num=page_idx + 1,
                        title=title,
                        body=body,
                        is_marker_page=False,
                    ))
                self.reporter.page_progress(page_idx + 1, total_pages)

            # Phase 4: marker stage（可選）
            if vector_indices:
                if self.marker_converter is None:
                    self.reporter.warning(
                        f"{len(vector_indices)} 頁需 Marker 但未提供 converter — 留 placeholder"
                    )
                else:
                    self.reporter.phase("MARKER", f"{len(vector_indices)} pages")
                    self._run_marker_stage(
                        doc, working_pdf, vector_indices, fragments, ctx,
                    )

            # Phase 5: stitch
            self.reporter.phase("STITCHING")
            self.stitcher.stitch(
                fragments,
                meta={
                    "filename_stem": stem,
                    "main_title": main_title,
                    "file_hash": _file_md5(input_path),
                },
                out=md_path,
            )

            doc.close()
            return True

        except Exception as e:
            self.reporter.warning(f"pipeline failed for {input_path.name}: {e}")
            return False
        finally:
            self._cleanup()

    # ---- internal ----

    def _prepare_env(self, output_base: Path, img_dir: Path) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_base.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

    def _run_marker_stage(
        self,
        doc,
        working_pdf: Path,
        vector_indices: list[int],
        fragments: list[PageFragment],
        ctx: ExtractContext,
    ) -> None:
        """每頁抽單頁 PDF → Marker → 改寫圖檔 → 補抓 h3 → 替換 placeholder。"""
        # 延遲 import marker（避免無 GPU 環境也能載 v5 package）
        try:
            import torch
            from marker.output import text_from_rendered
        except ImportError as e:
            self.reporter.warning(f"marker 套件未安裝: {e}")
            return

        for i, page_idx in enumerate(vector_indices):
            try:
                v_doc = fitz.open()
                v_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                tmp_pdf = self.tmp_dir / f"p{page_idx}.pdf"
                v_doc.save(str(tmp_pdf))
                v_doc.close()

                with torch.no_grad():
                    rendered = self.marker_converter(str(tmp_pdf))
                    page_text, _, images = text_from_rendered(rendered)

                if not page_text:
                    continue

                # banned_global_texts 移除
                for banned in ctx.filter_state.banned_global_texts:
                    if banned in page_text:
                        page_text = page_text.replace(banned, "")

                # 圖檔處理（含 hash 統一命名）+ markdown 引用改寫
                if images:
                    page_text = self.image_extractor.rewrite_marker_images(
                        page_text, images, ctx,
                    )

                # 補抓 h3
                frag = fragments[page_idx]
                if self.backfill_title_from_marker and frag.title is None:
                    import re as _re
                    title_m = _re.search(r"^##\s+\*\*(.+?)\*\*\s*$", page_text, _re.MULTILINE)
                    if not title_m:
                        title_m = _re.search(r"^##\s+(.+?)\s*$", page_text, _re.MULTILINE)
                    if title_m:
                        title = title_m.group(1).strip().strip("*").strip()
                        if 2 <= len(title) <= 80:
                            frag.title = title
                            # 把 marker 內那行 ## 移除避免雙標題
                            page_text = page_text.replace(title_m.group(0), "", 1)
                            page_text = _re.sub(r"\n{3,}", "\n\n", page_text)

                # 替換 placeholder
                frag.body = frag.body.replace(frag.marker_placeholder, f"\n{page_text}\n", 1)

                self.reporter.page_progress(
                    i + 1, len(vector_indices), label=f"P{page_idx + 1}",
                )
            except Exception as e:
                self.reporter.warning(f"marker page {page_idx + 1} failed: {e}")

    def _cleanup(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
