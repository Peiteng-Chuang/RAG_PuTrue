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
import json
import re
import shutil
from pathlib import Path
from typing import Any

import fitz

from .page_cache import CachedPage
from .progress import Phase, ProgressReporter, SilentReporter
from .strategies import (
    FormatConverter, TemplateFilter, TitleExtractor, TextCleaner,
    ImageExtractor, TriageStrategy, Stitcher,
)
from .strategies.marker_post import normalize_marker_headings
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
        enable_triage_log: bool = True,           # S1: 寫 triage decision log
        marker_pool_workers: int = 0,             # S3: 0=sequential, >0=process pool
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
        self.enable_triage_log = enable_triage_log
        self.marker_pool_workers = marker_pool_workers

    # ---- public ----

    def run(self, input_path: Path | str, output_base: Path | str) -> bool:
        input_path = Path(input_path).absolute()
        output_base = Path(output_base).absolute()
        stem = input_path.stem
        img_dir = output_base / f"{stem}_image"
        md_path = output_base / f"{stem}.md"

        # P2：per-file 統計累積。warning_count 用 reporter wrap 計數
        from .types import FileStats
        stats = FileStats(file_name=input_path.name)
        _real_warning = self.reporter.warning

        def _counting_warning(msg: str) -> None:
            stats.warning_count += 1
            _real_warning(msg)
        self.reporter.warning = _counting_warning  # type: ignore[method-assign]

        try:
            self._prepare_env(output_base, img_dir)

            # Phase 1: convert
            ext = input_path.suffix.lower()
            if self.converter.supports(ext):
                self.reporter.phase(Phase.CONVERTING_FORMAT, input_path.name)
                working_pdf = self.converter.convert(input_path, self.tmp_dir)
            else:
                working_pdf = input_path

            # Phase 2: open + template scan
            self.reporter.phase(Phase.FITZ_SCANNING)
            # R3: fitz.open 獨立 try/except，accurate 錯誤訊息 + 早 return
            try:
                doc = fitz.open(working_pdf)
            except Exception as e:
                self.reporter.warning(
                    f"pipeline failed at fitz.open({working_pdf.name}): "
                    f"{type(e).__name__}: {e}"
                )
                import traceback as _tb
                _tb.print_exc()
                return False
            filter_state = self.template_filter.scan(doc)
            self.reporter.phase(
                Phase.FITZ_SCAN_DONE,
                f"banned text={len(filter_state.banned_global_texts)}, "
                f"banned img={len(filter_state.banned_image_hashes)}",
            )
            # P1：template scan 累積的 warnings 統一 emit
            for w in filter_state.warnings:
                self.reporter.warning(w)
            filter_state.warnings.clear()

            # 主標題：掃前 n 頁取最大字體 page_title，全空 fallback 用 stem
            main_title = self._resolve_main_title(doc, stem, n_scan=3)

            # S1: 給 AdaptiveTriage 之類需 whole-doc 統計的策略 prepare 機會
            try:
                self.triage.prepare(doc)
            except Exception as e:
                import traceback
                self.reporter.warning(f"triage.prepare failed: {e}")
                traceback.print_exc()

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
            triage_log: list[dict] = []                 # S1: per-page decision log
            total_pages = len(doc)
            stats.total_pages = total_pages
            # P2：bad_xref 從 xref_to_hash_cache 的 None sentinel 數（穩定，不依賴 warning 字串）
            stats.bad_xref_count = sum(
                1 for v in filter_state.xref_to_hash_cache.values() if v is None
            )
            self.reporter.phase(Phase.PER_PAGE, f"{total_pages} pages")
            for page_idx in range(total_pages):
                page = CachedPage(doc[page_idx])
                title_hit = self.title_extractor.extract(page)
                title_text = title_hit.text if title_hit else None
                # R2: triage.route 任何 exception 不擋頁；fallback fast 並紀錄到 decision log
                route_fallback = False
                try:
                    route = self.triage.route(page)
                except Exception as e:
                    self.reporter.warning(
                        f"triage.route P{page_idx + 1} failed ({type(e).__name__}: {e})"
                        f"; fallback to fast"
                    )
                    route = "fast"
                    route_fallback = True
                    stats.triage_fallback_count += 1
                if self.enable_triage_log:
                    try:
                        explain = self.triage.explain(page)
                    except Exception:
                        explain = {}
                    triage_log.append({
                        "page": page_idx + 1, "route": route,
                        "route_fallback": route_fallback, **explain,
                    })
                if route == "slow":
                    placeholder = f"[[MARKER_REPLACE_P{page_idx}]]"
                    fragments.append(PageFragment(
                        page_idx=page_idx,
                        page_num=page_idx + 1,
                        title=title_text,
                        body=f"{placeholder}\n",
                        is_marker_page=True,
                        marker_placeholder=placeholder,
                    ))
                    vector_indices.append(page_idx)
                else:
                    # P2: 用 clean_structured 拿 (y0, text) 對，與 inline_headings merge
                    body_blocks = self.text_cleaner.clean_structured(
                        page, page_idx, title_hit, filter_state,
                    )
                    if title_hit and title_hit.headings:
                        for h in title_hit.headings.inline_headings:
                            body_blocks.append((h.y0, f"#### {h.text}"))
                    body_blocks.sort(key=lambda x: x[0])
                    image_refs = self.image_extractor.extract_fast(page, ctx)

                    # R7：body 空時不留虛空白；image refs 也空就 body=""
                    if body_blocks:
                        body = "\n".join(t for _, t in body_blocks).rstrip("\n")
                    else:
                        body = ""
                    if image_refs:
                        if body:
                            body += "\n\n"
                        for ref in image_refs:
                            body += ref.to_md_line()
                    # 連續 \n{3,} 正規化為 \n\n（重影 / 空行清理）
                    body = re.sub(r"\n{3,}", "\n\n", body)

                    subtitle_text = None
                    if title_hit and title_hit.headings and title_hit.headings.subtitle:
                        subtitle_text = title_hit.headings.subtitle.text

                    fragments.append(PageFragment(
                        page_idx=page_idx,
                        page_num=page_idx + 1,
                        title=title_text,
                        body=body,
                        is_marker_page=False,
                        subtitle=subtitle_text,
                    ))
                self.reporter.page_progress(page_idx + 1, total_pages)

            # P2：per-page loop 跑完，更新 fast/slow 計數
            stats.slow_pages = len(vector_indices)
            stats.fast_pages = total_pages - stats.slow_pages

            # P1：PER_PAGE 結束後 emit title extractor 累積的 warnings
            extractor_warnings = getattr(
                self.title_extractor, "collected_warnings", None,
            )
            if extractor_warnings:
                for w in extractor_warnings:
                    self.reporter.warning(w)
                extractor_warnings.clear()

            # Phase 4: marker stage（可選）
            if vector_indices:
                if self.marker_converter is None:
                    self.reporter.warning(
                        f"{len(vector_indices)} 頁需 Marker 但未提供 converter — 留 placeholder"
                    )
                else:
                    self.reporter.phase(Phase.MARKER, f"{len(vector_indices)} pages")
                    self._run_marker_stage(
                        doc, working_pdf, vector_indices, fragments, ctx,
                    )

            # S1: 寫 triage decision log（在 stitch 前，避免影響 ETL 成功與否）
            if self.enable_triage_log and triage_log:
                try:
                    log_dir = output_base / "_triage_log"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / f"{stem}.jsonl"
                    with open(log_path, "w", encoding="utf-8") as f:
                        for record in triage_log:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    self.reporter.warning(f"triage log write failed: {e}")

            # Phase 5: stitch
            self.reporter.phase(Phase.STITCHING)
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
            import traceback
            self.reporter.warning(f"pipeline failed for {input_path.name}: {e}")
            traceback.print_exc()
            return False
        finally:
            try:
                self.reporter.phase(Phase.CLEANING_UP)
            except Exception:
                pass
            self._cleanup()
            # P2：發 file_stats，再還原 reporter.warning
            try:
                self.reporter.file_stats(stats)
            except Exception:
                pass
            self.reporter.warning = _real_warning  # type: ignore[method-assign]

    # ---- internal ----

    def _resolve_main_title(self, doc, stem: str, n_scan: int = 3) -> str:
        """掃前 n_scan 頁的 title_extractor，取最大字體 page_title 當主標。

        排序鍵 = HeadingSpan.size（HierarchicalTitleExtractor 才有），FontSizeTitleExtractor
        無此資訊則 size=0、按頁序。全空 fallback 用 stem（取代寫死的「未命名簡報」）。
        """
        candidates: list[tuple[float, int, str]] = []
        for i in range(min(n_scan, len(doc))):
            try:
                hit = self.title_extractor.extract(doc[i])
            except Exception:
                continue
            if not hit or not hit.text:
                continue
            # 跳過「續前頁內容」之類合成 title（bbox=None 通常代表合成）
            if hit.bbox is None:
                continue
            size = 0.0
            if hit.headings and hit.headings.page_title:
                size = hit.headings.page_title.size
            candidates.append((size, i, hit.text))
        if not candidates:
            return stem
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]

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
        """Dispatch：marker_pool_workers > 0 → process pool；否則 sequential。"""
        if self.marker_pool_workers > 0:
            self._run_marker_pool(doc, vector_indices, fragments, ctx)
        else:
            self._run_marker_sequential(doc, vector_indices, fragments, ctx)

    def _run_marker_sequential(
        self, doc, vector_indices, fragments, ctx,
    ) -> None:
        """v4 等價：每頁抽單頁 PDF → marker_converter → 套用結果。"""
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

                self._apply_marker_result(page_idx, page_text, images, fragments, ctx)
                self.reporter.page_progress(
                    i + 1, len(vector_indices), label=f"P{page_idx + 1}",
                )
            except Exception as e:
                import traceback
                self.reporter.warning(f"marker page {page_idx + 1} failed: {e}")
                traceback.print_exc()

    def _run_marker_pool(
        self, doc, vector_indices, fragments, ctx,
    ) -> None:
        """S3 (B1)：process pool 平行跑 Marker。**未在 4090 驗證**，需 bench。

        每 worker 持久 PdfConverter。max_workers=2 在 24GB 4090 安全；3+ 邊緣或 OOM。
        """
        try:
            from .marker_pool import MarkerPool, deserialize_images
        except ImportError as e:
            self.reporter.warning(f"marker_pool 載入失敗 ({e})；回退 sequential")
            self._run_marker_sequential(doc, vector_indices, fragments, ctx)
            return

        # Step 1: 切單頁 PDF（sequential，便宜）
        tmp_paths: list[tuple[int, Path]] = []
        for page_idx in vector_indices:
            try:
                v_doc = fitz.open()
                v_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                tmp_pdf = self.tmp_dir / f"p{page_idx}.pdf"
                v_doc.save(str(tmp_pdf))
                v_doc.close()
                tmp_paths.append((page_idx, tmp_pdf))
            except Exception as e:
                import traceback
                self.reporter.warning(f"marker pool extract P{page_idx + 1}: {e}")
                traceback.print_exc()
        if not tmp_paths:
            return

        # Step 2: pool batch
        try:
            with MarkerPool(max_workers=self.marker_pool_workers) as pool:
                results = pool.process_batch([str(p) for _, p in tmp_paths])
        except Exception as e:
            import traceback
            self.reporter.warning(f"marker pool 失敗 ({e})；回退 sequential")
            traceback.print_exc()
            self._run_marker_sequential(doc, vector_indices, fragments, ctx)
            return

        # Step 3: 套結果（sequential，主程式做）
        for i, ((page_idx, _), result) in enumerate(zip(tmp_paths, results)):
            err = result.get("error")
            if err:
                self.reporter.warning(f"marker pool P{page_idx + 1}: {err}")
                continue
            # R5：worker 蒐集的單張圖序列化錯誤 → 在主程式 emit warning
            for img_err in result.get("image_errors", []) or []:
                self.reporter.warning(
                    f"marker pool P{page_idx + 1} image serialize "
                    f"orig={img_err.get('orig_name')}: {img_err.get('error')}"
                )
            page_text = result.get("text", "")
            if not page_text:
                continue
            try:
                images = deserialize_images(result.get("images", {}))
                self._apply_marker_result(page_idx, page_text, images, fragments, ctx)
            except Exception as e:
                import traceback
                self.reporter.warning(f"marker pool post P{page_idx + 1}: {e}")
                traceback.print_exc()
            self.reporter.page_progress(
                i + 1, len(tmp_paths), label=f"P{page_idx + 1}",
            )

    def _apply_marker_result(
        self, page_idx: int, page_text: str, images: Any,
        fragments: list[PageFragment], ctx: ExtractContext,
    ) -> None:
        """單頁 Marker 結果套到 fragment：banned 移除 + 圖檔重命名 + title backfill + heading normalize + 替換 placeholder。

        sequential / pool 兩種路徑共用此 helper。
        """
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
            title_m = re.search(r"^##\s+\*\*(.+?)\*\*\s*$", page_text, re.MULTILINE)
            if not title_m:
                title_m = re.search(r"^##\s+(.+?)\s*$", page_text, re.MULTILINE)
            if title_m:
                title = title_m.group(1).strip().strip("*").strip()
                if 2 <= len(title) <= 80:
                    frag.title = title
                    # 把 marker 內那行 ## 移除避免雙標題
                    page_text = page_text.replace(title_m.group(0), "", 1)
                    page_text = re.sub(r"\n{3,}", "\n\n", page_text)

        # P3: 把剩餘的 # / ## / ### 階層 normalize 成 ### / ####，跟 fast path 對齊
        page_text = normalize_marker_headings(page_text)

        # 替換 placeholder
        frag.body = frag.body.replace(frag.marker_placeholder, f"\n{page_text}\n", 1)

    def _cleanup(self) -> None:
        """Per-file cleanup。S2：**不**每檔呼叫 torch.cuda.empty_cache()
        （頻繁呼叫反而拖慢），VRAM 釋放靠 cleanup_gpu() 在批次結束統一做。
        """
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        gc.collect()

    @staticmethod
    def cleanup_gpu() -> None:
        """S2：批次結束呼叫一次，釋放 GPU memory。

        每檔呼叫 empty_cache 反而觸發頻繁同步、降低吞吐。
        4090 + Marker 環境 VRAM 峰值需先 bench 觀察是否會撞 24GB。
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
