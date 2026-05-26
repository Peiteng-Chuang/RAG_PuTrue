"""批次處理：掃 data_root 所有 PDF/PPT/PPTX → 跑 RAGPipeline → 寫 tracker。

⚠️ 資料安全設計（feedback-data-dependency-audit-first）：
- 預設「sidecar-aware skip」：若對應 .review.json 的 review_status != "unprocessed"
  就 skip（保護使用者已標記/已 embed/已 ingest 的檔）。
- 「pipeline 已 SUCCESS 過」的 skip 也預設開啟（resume 友善）。
- 想覆寫已標記檔，必須**明確傳** allow_overwrite_labeled=True 並承擔後果。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .pipeline import RAGPipeline
from .progress import ProgressReporter, ConsoleReporter, SilentReporter


# Worker globals（S5 file-level pool 用，per-process 持久）
_worker_pipeline = None
_worker_output_root: Path | None = None


def _file_worker_init(output_root_str: str, no_marker: bool) -> None:
    """S5：file worker 啟動時 build pipeline 一次（含 marker），持久於 process 生命週期。

    R1：整個 init 流程包 try/except。Marker build 失敗 → fallback 無 marker pipeline，
    印 stderr warning，不 raise（raise 會讓 ProcessPoolExecutor BrokenProcessPool 整個 pool 跪）。
    default_pipeline build 失敗才 _worker_pipeline = None sentinel，run 時 short-circuit return error。
    """
    global _worker_pipeline, _worker_output_root
    import sys as _sys
    import traceback as _tb

    marker = None
    if not no_marker:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            marker = PdfConverter(artifact_dict=create_model_dict())
        except Exception as e:
            print(
                f"[WARN] worker_init: marker build 失敗，回退無 marker pipeline "
                f"({type(e).__name__}: {e})",
                file=_sys.stderr, flush=True,
            )
            _tb.print_exc(file=_sys.stderr)
            marker = None

    try:
        from mkd_generator_v5 import default_pipeline
        _worker_pipeline = default_pipeline(
            marker_converter=marker, reporter=SilentReporter(),
        )
        _worker_output_root = Path(output_root_str)
    except Exception as e:
        print(
            f"[ERROR] worker_init: default_pipeline build 失敗，"
            f"worker 進入 sentinel 模式 ({type(e).__name__}: {e})",
            file=_sys.stderr, flush=True,
        )
        _tb.print_exc(file=_sys.stderr)
        _worker_pipeline = None
        _worker_output_root = Path(output_root_str)


def _file_worker_run(file_path_str: str) -> dict:
    """Worker：跑單檔 pipeline，回 dict 給主程式 aggregate。

    R1：worker init 進入 sentinel (_worker_pipeline is None) 時直接 return failure，
    避免每檔都跳 AttributeError 噴一堆無用 traceback。
    """
    global _worker_pipeline, _worker_output_root
    t0 = time.time()
    if _worker_pipeline is None:
        return {
            "file": file_path_str, "ok": False,
            "elapsed": 0.0, "error": "worker_init failed; pipeline unavailable",
        }
    try:
        ok = _worker_pipeline.run(Path(file_path_str), _worker_output_root)
        return {"file": file_path_str, "ok": ok, "elapsed": time.time() - t0}
    except Exception as e:
        return {
            "file": file_path_str, "ok": False,
            "elapsed": time.time() - t0, "error": str(e),
        }


# 與 md_review_ui.py 的 sidecar schema 對齊（[[implementation-progress-md-review-ui-py-qdrant-pipeline]]）
SIDECAR_PROTECTED_STATUSES = {"processing", "encoded", "ingested"}
LEGACY_PROTECTED_STATUSES = {"in_progress", "approved", "needs_rework", "done"}


def _derive_sidecar_path(output_root: Path, file_path: Path) -> Path:
    """輸入檔對應的 sidecar JSON 路徑。"""
    return output_root / f"{file_path.stem}.review.json"


def _load_sidecar_status(sidecar_path: Path) -> str | None:
    """讀 sidecar 的 review_status；不存在或讀失敗回 None。"""
    if not sidecar_path.exists():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return data.get("review_status")
    except (json.JSONDecodeError, OSError):
        return None


def _is_protected(status: str | None) -> bool:
    """status 是否屬於「不可破壞」的狀態（已標記 / 已 embed / 已 ingest）。"""
    if status is None:
        return False
    return status in SIDECAR_PROTECTED_STATUSES or status in LEGACY_PROTECTED_STATUSES


class BatchProcessor:
    """v4 BatchProcessor 等價 + 預設安全（不覆寫已標記檔）。

    Skip 邏輯（依序）：
    1. allow_overwrite_labeled=False（預設）：sidecar.review_status ∈ {processing/encoded/ingested} → skip
    2. skip_success=True（預設）：tracker[file_key] == SUCCESS → skip
    3. 否則：跑 pipeline

    對外明確區分「safe skip」（保護資料）跟「resume skip」（避免重做）。
    """

    def __init__(
        self,
        data_root: Path | str,
        output_root: Path | str,
        pipeline: RAGPipeline,
        reporter: ProgressReporter | None = None,
        extensions: tuple[str, ...] = ("*.pdf", "*.ppt", "*.pptx"),
    ):
        self.data_root = Path(data_root).absolute()
        self.output_root = Path(output_root).absolute()
        self.pipeline = pipeline
        self.reporter = reporter or ConsoleReporter()
        self.extensions = extensions
        self.tracker_path = self.output_root / "process_tracker.json"
        self.processed_files = self._load_tracker()

    def _load_tracker(self) -> dict:
        if self.tracker_path.exists():
            with open(self.tracker_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_tracker(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(self.tracker_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=4)

    def preview(self) -> dict:
        """跑 process_all 前先 dry-run，回 {will_process / safe_skip / resume_skip}。

        讓使用者在實際動 disk 前確認哪些檔會被處理、哪些被保護。
        """
        files: list[Path] = []
        for ext in self.extensions:
            files.extend(self.data_root.rglob(ext))
        files.sort()

        will_process: list[str] = []
        safe_skip: list[tuple[str, str]] = []   # (file_key, status)
        resume_skip: list[str] = []

        for file_path in files:
            file_key = str(file_path.relative_to(self.data_root))
            sidecar = _derive_sidecar_path(self.output_root, file_path)
            status = _load_sidecar_status(sidecar)
            if _is_protected(status):
                safe_skip.append((file_key, status))
            elif self.processed_files.get(file_key) == "SUCCESS":
                resume_skip.append(file_key)
            else:
                will_process.append(file_key)

        return {
            "total": len(files),
            "will_process": will_process,
            "safe_skip_labeled": safe_skip,
            "resume_skip_success": resume_skip,
        }

    def process_all(
        self,
        skip_success: bool = True,
        allow_overwrite_labeled: bool = False,
    ) -> tuple[int, int]:
        """跑批次。回 (成功數, 失敗數)。

        Args:
            skip_success: tracker[key]=SUCCESS 的 skip（resume 用）
            allow_overwrite_labeled: 是否覆寫 sidecar.review_status ∈
                {processing/encoded/ingested} 的檔。**強烈不建議設 True**，
                會破壞使用者標記資料 + 失效 Qdrant 內的 chunks 對應關係。
                若真要覆寫某幾個檔，建議手動刪掉那幾個 sidecar 而不是用此 flag。
        """
        files: list[Path] = []
        for ext in self.extensions:
            files.extend(self.data_root.rglob(ext))
        files.sort()

        total = len(files)
        self.reporter.batch_start(total)
        ok = failed = 0

        for idx, file_path in enumerate(files, start=1):
            file_key = str(file_path.relative_to(self.data_root))

            # 1. Safe skip：保護已標記檔（除非明確 allow_overwrite_labeled）
            if not allow_overwrite_labeled:
                sidecar = _derive_sidecar_path(self.output_root, file_path)
                status = _load_sidecar_status(sidecar)
                if _is_protected(status):
                    self.reporter.file_start(
                        idx, total, f"{file_key} (SAFE-SKIP — review_status={status})",
                    )
                    self.reporter.file_done(file_key, ok=True, elapsed=0.0)
                    ok += 1
                    continue

            # 2. Resume skip：tracker 內標 SUCCESS（避免重複跑）
            if skip_success and self.processed_files.get(file_key) == "SUCCESS":
                self.reporter.file_start(idx, total, f"{file_key} (skip — SUCCESS)")
                self.reporter.file_done(file_key, ok=True, elapsed=0.0)
                ok += 1
                continue

            # 3. 跑 pipeline
            self.reporter.file_start(idx, total, file_key)
            t0 = time.time()
            success = self.pipeline.run(file_path, self.output_root)
            elapsed = time.time() - t0

            if success:
                self.processed_files[file_key] = "SUCCESS"
                ok += 1
            else:
                self.processed_files[file_key] = "FAILED"
                failed += 1

            self.reporter.file_done(file_key, ok=success, elapsed=elapsed)
            self._save_tracker()

        # S2：批次結束統一釋放 GPU memory（per-file empty_cache 反而拖慢吞吐）
        RAGPipeline.cleanup_gpu()
        self.reporter.batch_done(ok, failed)
        return ok, failed

    def process_all_parallel(
        self,
        workers: int,
        no_marker: bool = False,
        skip_success: bool = True,
        allow_overwrite_labeled: bool = False,
    ) -> tuple[int, int]:
        """S5：file-level 平行化。每 worker 自建 pipeline（含 marker）。

        ⚠️ 高 VRAM 成本：N workers × marker model 必須 < GPU。**與 marker_pool 互斥**
        （default_pipeline 預設 marker_pool_workers=0，這裡走 sequential marker，
        VRAM 預估 N × 6GB；24GB GPU 上 N ≤ 3 安全）。

        workers < 2 → fallback 到 process_all。
        """
        if workers < 2:
            return self.process_all(skip_success, allow_overwrite_labeled)

        # Step 1: scan + filter
        files: list[Path] = []
        for ext in self.extensions:
            files.extend(self.data_root.rglob(ext))
        files.sort()

        process_list: list[Path] = []
        safe_skipped: list[tuple[str, str]] = []
        resume_skipped: list[str] = []
        for file_path in files:
            file_key = str(file_path.relative_to(self.data_root))
            if not allow_overwrite_labeled:
                sidecar = _derive_sidecar_path(self.output_root, file_path)
                status = _load_sidecar_status(sidecar)
                if _is_protected(status):
                    safe_skipped.append((file_key, status))
                    continue
            if skip_success and self.processed_files.get(file_key) == "SUCCESS":
                resume_skipped.append(file_key)
                continue
            process_list.append(file_path)

        total = len(process_list)
        self.reporter.batch_start(total + len(safe_skipped) + len(resume_skipped))
        # 先發 skipped 事件（保持與 sequential 行為對等）
        ok = failed = 0
        for fk, status in safe_skipped:
            self.reporter.file_start(0, total, f"{fk} (SAFE-SKIP — {status})")
            self.reporter.file_done(fk, ok=True, elapsed=0.0)
            ok += 1
        for fk in resume_skipped:
            self.reporter.file_start(0, total, f"{fk} (skip — SUCCESS)")
            self.reporter.file_done(fk, ok=True, elapsed=0.0)
            ok += 1

        if total == 0:
            RAGPipeline.cleanup_gpu()
            self.reporter.batch_done(ok, failed)
            return ok, failed

        # Step 2: parallel
        # R1：pool 建立 / 跑檔過程任何 BrokenProcessPool / 其他 fatal exception，
        # fallback 到 sequential process_all。pool 在 with-block 已開始派工的 future
        # 結果無法救，但至少剩下的 file 不會 lost。
        ctx = mp.get_context("spawn")
        pool_broken = False
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_file_worker_init,
                initargs=(str(self.output_root), no_marker),
                mp_context=ctx,
            ) as executor:
                future_to_path = {
                    executor.submit(_file_worker_run, str(p)): p for p in process_list
                }
                for idx, fut in enumerate(as_completed(future_to_path), start=1):
                    file_path = future_to_path[fut]
                    file_key = str(file_path.relative_to(self.data_root))
                    self.reporter.file_start(idx, total, file_key)
                    try:
                        result = fut.result()
                        success = bool(result.get("ok", False))
                        elapsed = float(result.get("elapsed", 0.0))
                        if result.get("error"):
                            self.reporter.warning(
                                f"{file_key} worker error: {result['error']}"
                            )
                    except Exception as e:
                        success, elapsed = False, 0.0
                        self.reporter.warning(f"{file_key} future raised: {e}")

                    if success:
                        self.processed_files[file_key] = "SUCCESS"
                        ok += 1
                    else:
                        self.processed_files[file_key] = "FAILED"
                        failed += 1
                    self.reporter.file_done(file_key, ok=success, elapsed=elapsed)
                    self._save_tracker()
        except Exception as e:
            self.reporter.warning(
                f"file-level pool fatal ({type(e).__name__}: {e})；"
                f"剩餘未跑檔將以 sequential 接手"
            )
            pool_broken = True

        if pool_broken:
            # 計算還沒處理的（沒進 processed_files 的） → 接手 sequential 跑
            remaining = [
                p for p in process_list
                if self.processed_files.get(str(p.relative_to(self.data_root)))
                not in {"SUCCESS", "FAILED"}
            ]
            if remaining:
                # 用 process_all 對剩餘檔 sequential 跑（會自己掃整個 data_root，
                # 但 skip_success 會自然跳過已處理的；此為合理 fallback）
                seq_ok, seq_failed = self.process_all(
                    skip_success=True, allow_overwrite_labeled=allow_overwrite_labeled,
                )
                ok += seq_ok
                failed += seq_failed

        RAGPipeline.cleanup_gpu()
        self.reporter.batch_done(ok, failed)
        return ok, failed
