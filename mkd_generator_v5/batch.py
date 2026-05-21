"""批次處理：掃 data_root 所有 PDF/PPT/PPTX → 跑 RAGPipeline → 寫 tracker。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .pipeline import RAGPipeline
from .progress import ProgressReporter, ConsoleReporter


class BatchProcessor:
    """v4 BatchProcessor 等價。差別：
    - 接受預先建好的 RAGPipeline（不再內建 hard-coded strategy 組合）
    - 用 ProgressReporter 取代 print
    - 跑單檔失敗不中斷整批
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

    def process_all(self, skip_success: bool = True) -> tuple[int, int]:
        """回 (成功數, 失敗數)。"""
        files: list[Path] = []
        for ext in self.extensions:
            files.extend(self.data_root.rglob(ext))
        # 一致順序，便於 resume
        files.sort()

        total = len(files)
        self.reporter.batch_start(total)
        ok = failed = 0

        for idx, file_path in enumerate(files, start=1):
            file_key = str(file_path.relative_to(self.data_root))
            if skip_success and self.processed_files.get(file_key) == "SUCCESS":
                self.reporter.file_start(idx, total, f"{file_key} (skip — SUCCESS)")
                self.reporter.file_done(file_key, ok=True, elapsed=0.0)
                ok += 1
                continue

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
            # 每檔結束立刻 persist，避免中途斷線損失進度
            self._save_tracker()

        self.reporter.batch_done(ok, failed)
        return ok, failed
