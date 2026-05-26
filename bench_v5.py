"""v5 ETL baseline 量測腳本。S0 — 軌道 2（速度/硬體）量測基礎建設。

用法：
    python bench_v5.py --manifest bench_manifest.json --tag baseline
    python bench_v5.py --manifest bench_manifest.json --tag triage300 --threshold 300

manifest.json 範例：
    {"pdfs": ["path/to/a.pdf", "path/to/b.pdf"], "notes": "10-PDF mixed set"}

輸出：`bench_results/{tag}.json`，含 per-phase 時間、per-file breakdown、triage 計數、GPU
時序樣本、config 快照。tag 命名建議：baseline / triage<value> / multi_signal / marker_pool2 等。

⚠️ Bench 不寫 sidecar、不更新 process_tracker；輸出寫到 `bench_results/_workdir_{tag}/`
獨立暫存區，跑完不清理（方便檢視）。要保留正式產出請用 run_v5.py。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# 對齊 run_v5.py 的 CUDA env 設定；必須在 import torch 前
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from mkd_generator_v5 import default_pipeline
from mkd_generator_v5.progress import SilentReporter
from mkd_generator_v5.strategies import TriageStrategy, DrawingCountTriage


# ─────────────────────────────────────────────────────────────────────────────
# 量測元件
# ─────────────────────────────────────────────────────────────────────────────

class BenchReporter(SilentReporter):
    """記錄每個 file 的 phase 時間，不印 console。"""

    def __init__(self):
        self.file_records: list[dict[str, Any]] = []
        self._curr: dict[str, Any] | None = None
        self._current_phase: str | None = None
        self._phase_t0: float | None = None

    def file_start(self, idx, total, name):
        self._curr = {"name": name, "phases": {}, "start_wall": time.time()}

    def file_done(self, name, ok, elapsed):
        self._close_phase()
        if self._curr is None:
            return
        self._curr["ok"] = ok
        self._curr["elapsed"] = elapsed
        self.file_records.append(self._curr)
        self._curr = None

    def phase(self, phase_name, detail=""):
        self._close_phase()
        if not phase_name.endswith("_DONE"):
            self._current_phase = phase_name
            self._phase_t0 = time.perf_counter()

    def _close_phase(self):
        if self._current_phase is None or self._phase_t0 is None or self._curr is None:
            return
        dt = time.perf_counter() - self._phase_t0
        phases = self._curr["phases"]
        phases[self._current_phase] = phases.get(self._current_phase, 0.0) + dt
        self._current_phase = None
        self._phase_t0 = None


class CountingTriage(TriageStrategy):
    """裝飾任一 TriageStrategy，計算 fast/slow 各幾頁 + 收集 per-page decision log。"""

    def __init__(self, inner: TriageStrategy):
        self.inner = inner
        self.counts = {"fast": 0, "slow": 0}
        self.decisions: list[dict] = []  # [{file, page, route, drawings, ...}]

    def route(self, page):
        r = self.inner.route(page)
        self.counts[r] = self.counts.get(r, 0) + 1
        # 蒐集診斷訊號（不影響決策）
        try:
            n_draw = len(page.get_drawings())
        except Exception:
            n_draw = -1
        self.decisions.append({
            "page_idx": getattr(page, "number", -1),
            "route": r,
            "drawings": n_draw,
        })
        return r


class GPUMonitor:
    """背景 thread sample nvidia-smi，記錄 (timestamp, util%, mem_mb) timeseries。

    無 nvidia-smi（CPU-only 環境）→ samples 空，不報錯。
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available: bool | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                self._available = (out.returncode == 0)
                if out.returncode == 0:
                    line = out.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        self.samples.append({
                            "t": round(time.time(), 2),
                            "util": int(parts[0]),
                            "mem_mb": int(parts[1]),
                        })
            except (subprocess.SubprocessError, ValueError, OSError, IndexError):
                self._available = False
            # 即使 nvidia-smi 不存在也要 sleep，避免 busy loop
            self._stop.wait(self.interval)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_marker_converter():
    """跟 run_v5.py 對齊，可選載入 Marker。"""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        print("🔧 載入 Marker（首次需數十秒）...")
        return PdfConverter(artifact_dict=create_model_dict())
    except ImportError as e:
        print(f"⚠️ Marker 套件未裝（{e}）。CAD 頁將留 placeholder。")
        return None
    except Exception as e:
        print(f"⚠️ Marker 載入失敗（{e}）。CAD 頁將留 placeholder。")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="v5 ETL bench")
    ap.add_argument("--manifest", required=True,
                    help="JSON: {\"pdfs\": [\"path1\", ...]}")
    ap.add_argument("--tag", required=True, help="輸出檔 tag, e.g. baseline / triage300")
    ap.add_argument("--output", default="bench_results")
    ap.add_argument("--no-marker", action="store_true")
    ap.add_argument("--threshold", type=int, default=None,
                    help="覆寫 DrawingCountTriage threshold（測閾值用）")
    ap.add_argument("--gpu-interval", type=float, default=1.0)
    ap.add_argument("--clean-workdir", action="store_true",
                    help="跑完清掉 _workdir（預設留著方便檢視）")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"❌ 找不到 manifest：{manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdfs = [Path(p) for p in manifest.get("pdfs", [])]
    if not pdfs:
        print("❌ manifest 無 pdfs 欄位或為空", file=sys.stderr)
        return 1

    out_dir = Path(args.output).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / f"_workdir_{args.tag}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 建立 pipeline
    reporter = BenchReporter()
    marker = None if args.no_marker else _build_marker_converter()
    pipeline = default_pipeline(marker_converter=marker, reporter=reporter)

    # 閾值覆寫
    if args.threshold is not None:
        pipeline.triage = DrawingCountTriage(threshold=args.threshold)

    # 包 CountingTriage 蒐集 decisions
    triage_count = CountingTriage(pipeline.triage)
    pipeline.triage = triage_count

    # GPU monitor
    gpu = GPUMonitor(interval=args.gpu_interval)
    gpu.start()

    # 跑
    print(f"\n📊 Bench tag={args.tag} | files={len(pdfs)} | "
          f"marker={'off' if args.no_marker else 'on'}")
    t_start_wall = time.time()
    t_start_perf = time.perf_counter()
    failed: list[str] = []
    reporter.batch_start(len(pdfs))
    for i, pdf in enumerate(pdfs, 1):
        if not pdf.exists():
            print(f"  [{i}/{len(pdfs)}] ❌ {pdf} 不存在")
            failed.append(str(pdf))
            continue
        print(f"  [{i}/{len(pdfs)}] {pdf.name} ...", end=" ", flush=True)
        reporter.file_start(i, len(pdfs), pdf.name)
        t_file = time.perf_counter()
        try:
            ok = pipeline.run(pdf, work_dir)
            elapsed = time.perf_counter() - t_file
            reporter.file_done(pdf.name, ok=ok, elapsed=elapsed)
            print(f"{'✓' if ok else '✗'} {elapsed:.1f}s")
            if not ok:
                failed.append(str(pdf))
        except Exception as e:
            elapsed = time.perf_counter() - t_file
            reporter.file_done(pdf.name, ok=False, elapsed=elapsed)
            print(f"✗ ({e})")
            failed.append(str(pdf))
    reporter.batch_done(len(pdfs) - len(failed), len(failed))

    total_time = time.perf_counter() - t_start_perf
    # S2：bench 結束統一清 GPU
    from mkd_generator_v5.pipeline import RAGPipeline
    RAGPipeline.cleanup_gpu()
    gpu.stop()

    # 整理結果
    result = {
        "tag": args.tag,
        "start_wall": t_start_wall,
        "total_time_sec": round(total_time, 3),
        "n_files": len(pdfs),
        "n_failed": len(failed),
        "failed_files": failed,
        "triage_counts": dict(triage_count.counts),
        "triage_decisions_sample": triage_count.decisions[:200],  # 截前 200 筆避免肥
        "per_file": reporter.file_records,
        "gpu_available": gpu._available,
        "gpu_samples_count": len(gpu.samples),
        "gpu_samples": gpu.samples,
        "config": {
            "title_extractor": type(pipeline.title_extractor).__name__,
            "text_cleaner": type(pipeline.text_cleaner).__name__,
            "triage_inner": type(triage_count.inner).__name__,
            "triage_threshold_arg": args.threshold,
            "no_marker": args.no_marker,
            "marker_loaded": marker is not None,
        },
        "manifest": str(manifest_path),
        "notes": manifest.get("notes", ""),
    }
    out_file = out_dir / f"{args.tag}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    if args.clean_workdir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    # 顯眼摘要
    print()
    print("─" * 60)
    print(f"📊 Bench done. tag={args.tag}")
    print(f"   total: {total_time:.1f}s | files: {len(pdfs)} ({len(failed)} failed)")
    print(f"   triage: {dict(triage_count.counts)}")
    print(f"   gpu samples: {len(gpu.samples)} (available={gpu._available})")
    print(f"   result → {out_file}")
    if work_dir.exists():
        print(f"   workdir kept at: {work_dir}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
