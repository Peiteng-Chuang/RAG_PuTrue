"""進度回報抽象。ConsoleReporter 為 default；TqdmReporter / SilentReporter 替代。

Pipeline 在 phase 切換、單頁完成時呼叫對應 method；reporter 自行決定怎麼呈現。

P4 emoji 規範（ConsoleReporter / TqdmReporter / run_v5.py CLI 對齊）：

| 用途 | Emoji | Reporter method |
|---|---|---|
| 批次 START | 🚀 | batch_start |
| 批次 DONE | 🏁 | batch_done |
| 單檔 START | 📄 | file_start |
| 單檔 SUCCESS | ✅ | file_done (ok=True) |
| 單檔 FAIL | ❌ | file_done (ok=False) |
| Phase 切換 | ▶ | phase |
| Warning | ⚠️ | warning |
| File stats | 📊 | file_stats |
| ETA 預估 | ⏳ | (ConsoleReporter.file_done 內每 N 檔印) |

CLI 額外用 emoji（run_v5.py）：
- 📋 掃描結果 summary
- 🛡️ SAFE-SKIP（受保護檔，不覆寫）
- ⏭ RESUME-SKIP（tracker 標 SUCCESS）

避免重複 emoji 表達同一語意。新增 reporter method 時對齊此表，不要自創。
"""
from __future__ import annotations

import statistics
import sys
import time
from abc import ABC, abstractmethod
from enum import StrEnum


class Phase(StrEnum):
    """A5：pipeline phase 名稱列舉。

    StrEnum 自動是 str subclass，傳給 reporter.phase() 跟過去傳 string 完全相容；
    使用 enum 讓 IDE / type checker 抓 typo。`_DONE` 結尾的視為「事件標記」，
    _PhaseTimer 不會把它算成新 phase。
    """
    CONVERTING_FORMAT = "CONVERTING_FORMAT"
    FITZ_SCANNING = "FITZ_SCANNING"
    FITZ_SCAN_DONE = "FITZ_SCAN_DONE"
    PER_PAGE = "PER_PAGE"
    MARKER = "MARKER"
    STITCHING = "STITCHING"
    CLEANING_UP = "CLEANING_UP"


class ProgressReporter(ABC):
    """抽象介面。所有 method 不允許 raise（reporter 失敗不可中斷 pipeline）。"""

    @abstractmethod
    def batch_start(self, total: int) -> None: ...

    @abstractmethod
    def batch_done(self, ok: int, failed: int) -> None: ...

    @abstractmethod
    def file_start(self, idx: int, total: int, name: str) -> None: ...

    @abstractmethod
    def file_done(self, name: str, ok: bool, elapsed: float) -> None: ...

    @abstractmethod
    def phase(self, phase_name: str, detail: str = "") -> None: ...

    @abstractmethod
    def page_progress(self, curr: int, total: int, label: str = "") -> None: ...

    @abstractmethod
    def warning(self, msg: str) -> None: ...

    def file_stats(self, stats) -> None:
        """P2：optional hook，pipeline 在 file_done 之前 attach 單檔濃縮統計。
        default 空，子類想印就 override。stats 是 mkd_generator_v5.types.FileStats。"""
        pass


class SilentReporter(ProgressReporter):
    """全靜音。給測試或自動化用。"""
    def batch_start(self, total): pass
    def batch_done(self, ok, failed): pass
    def file_start(self, idx, total, name): pass
    def file_done(self, name, ok, elapsed): pass
    def phase(self, phase_name, detail=""): pass
    def page_progress(self, curr, total, label=""): pass
    def warning(self, msg): pass
    def file_stats(self, stats): pass


class _PhaseTimer:
    """Phase 計時助手，供 ConsoleReporter / TqdmReporter 內部使用。

    每次 phase() 切換把上一個 phase 的 elapsed 累積進 per-file 跟 batch 兩份 dict。
    `_DONE` 結尾的 phase name 視為「事件標記」（如 FITZ_SCAN_DONE），不算新 phase 起點，
    只觸發前一 phase 的 close。
    """
    def __init__(self):
        self.batch_totals: dict[str, float] = {}
        self.batch_counts: dict[str, int] = {}
        self._file_totals: dict[str, float] = {}
        self._current: str | None = None
        self._t0: float | None = None

    def file_start(self) -> None:
        self._current = None
        self._t0 = None
        self._file_totals = {}

    def phase(self, name: str) -> None:
        self._close_current()
        if name.endswith("_DONE"):
            return
        self._current = name
        self._t0 = time.time()
        self.batch_counts[name] = self.batch_counts.get(name, 0) + 1

    def file_done(self) -> str:
        """關掉跑中的 phase，回傳 per-file breakdown 字串（無資料回 ""）。"""
        self._close_current()
        if not self._file_totals:
            return ""
        total = sum(self._file_totals.values())
        items = sorted(self._file_totals.items(), key=lambda kv: -kv[1])
        if total <= 0:
            parts = [f"{name} {secs:.1f}s" for name, secs in items]
        else:
            parts = [f"{name} {secs:.1f}s ({secs/total*100:.0f}%)" for name, secs in items]
        return "    ⏱  " + " · ".join(parts)

    def batch_done(self) -> str:
        """跨檔 phase 加總 breakdown（無資料回 ""）。"""
        self._close_current()
        if not self.batch_totals:
            return ""
        total = sum(self.batch_totals.values())
        items = sorted(self.batch_totals.items(), key=lambda kv: -kv[1])
        lines = ["⏱ phase breakdown (batch, sum across files):"]
        for name, secs in items:
            pct = (secs / total * 100) if total > 0 else 0.0
            cnt = self.batch_counts.get(name, 0)
            lines.append(f"   {name:<22} {secs:7.1f}s ({pct:5.1f}%)  ×{cnt}")
        return "\n".join(lines)

    def _close_current(self) -> None:
        if self._current is None or self._t0 is None:
            return
        elapsed = time.time() - self._t0
        self._file_totals[self._current] = self._file_totals.get(self._current, 0.0) + elapsed
        self.batch_totals[self._current] = self.batch_totals.get(self._current, 0.0) + elapsed
        self._current = None
        self._t0 = None


class ConsoleReporter(ProgressReporter):
    """純 print 版。對齊 v4 風格（emoji + 中文），但更勤奮回報。

    page_progress 用 \\r 在同一行更新（terminal-aware）；非 terminal 環境每 10% 才印一次。
    """
    def __init__(self, page_progress_step_pct: int = 10, eta_every_n: int = 10):
        self._page_step = max(1, page_progress_step_pct)
        self._eta_every_n = max(1, eta_every_n)
        self._batch_t0: float | None = None
        self._file_t0: float | None = None
        self._last_page_print_pct: int = -1
        self._is_tty = sys.stdout.isatty()
        self._timer = _PhaseTimer()
        # P5：per-file elapsed 累積，用於 P50/P95 + ETA
        self._file_elapsed: list[float] = []
        self._batch_total: int = 0

    def batch_start(self, total):
        self._batch_t0 = time.time()
        self._batch_total = total
        self._file_elapsed.clear()
        print(f"🚀 批次開始：共 {total} 個檔案")

    def batch_done(self, ok, failed):
        elapsed = (time.time() - self._batch_t0) if self._batch_t0 else 0.0
        print(f"\n🏁 批次完成：成功 {ok} · 失敗 {failed} · 耗時 {elapsed:.1f}s")
        # P5：per-file P50/P95/mean
        if self._file_elapsed:
            mean = statistics.mean(self._file_elapsed)
            p50 = statistics.median(self._file_elapsed)
            sorted_e = sorted(self._file_elapsed)
            p95_idx = max(0, min(len(sorted_e) - 1,
                                 int(round(len(sorted_e) * 0.95)) - 1))
            p95 = sorted_e[p95_idx]
            print(f"  📊 per-file: mean={mean:.1f}s · P50={p50:.1f}s · "
                  f"P95={p95:.1f}s · n={len(self._file_elapsed)}")
        batch_breakdown = self._timer.batch_done()
        if batch_breakdown:
            print(batch_breakdown)

    def file_start(self, idx, total, name):
        self._file_t0 = time.time()
        self._last_page_print_pct = -1
        self._timer.file_start()
        print(f"\n[{idx}/{total}] 📄 {name}")

    def file_done(self, name, ok, elapsed):
        breakdown = self._timer.file_done()
        icon = "✅" if ok else "❌"
        print(f"  {icon} {elapsed:.1f}s")
        if breakdown:
            print(breakdown)
        # P5：累積 elapsed，每 eta_every_n 檔印 ETA 估算
        self._file_elapsed.append(elapsed)
        n_done = len(self._file_elapsed)
        if (self._batch_total > 0 and n_done < self._batch_total
                and n_done % self._eta_every_n == 0):
            mean = statistics.mean(self._file_elapsed)
            eta_s = mean * (self._batch_total - n_done)
            eta_min = eta_s / 60.0
            print(f"  ⏳ ETA ~{eta_min:.1f} min "
                  f"({n_done}/{self._batch_total} done, mean {mean:.1f}s/file)")

    def phase(self, phase_name, detail=""):
        self._timer.phase(phase_name)
        self._last_page_print_pct = -1
        suffix = f" — {detail}" if detail else ""
        print(f"  ▶ {phase_name}{suffix}")

    def page_progress(self, curr, total, label=""):
        if total <= 0:
            return
        pct = int(curr * 100 / total)
        if self._is_tty:
            tail = f" {label}" if label else ""
            sys.stdout.write(f"\r    頁 {curr}/{total} ({pct}%){tail}   ")
            sys.stdout.flush()
            if curr == total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            if pct - self._last_page_print_pct >= self._page_step or curr == total:
                print(f"    頁 {curr}/{total} ({pct}%)")
                self._last_page_print_pct = pct

    def warning(self, msg):
        print(f"  ⚠️  {msg}")

    def file_stats(self, stats):
        print(f"  {stats.summary_line()}")


class TqdmReporter(ProgressReporter):
    """tqdm 三層 bar：batch / file_phase / page。需 `pip install tqdm`。

    若 tqdm 未裝，自動 fallback 成 ConsoleReporter（不中斷）。
    """
    def __init__(self):
        try:
            from tqdm import tqdm  # noqa: F401
            self._tqdm = tqdm
            self._fallback: ConsoleReporter | None = None
        except ImportError:
            self._tqdm = None
            self._fallback = ConsoleReporter()
        self._batch_bar = None
        self._page_bar = None
        self._current_file_name: str | None = None
        self._timer = _PhaseTimer()

    def _f(self) -> ConsoleReporter | None:
        return self._fallback

    def batch_start(self, total):
        if self._f(): return self._f().batch_start(total)
        self._batch_bar = self._tqdm(total=total, desc="batch", unit="file")

    def batch_done(self, ok, failed):
        if self._f(): return self._f().batch_done(ok, failed)
        if self._batch_bar:
            self._batch_bar.close()
            self._batch_bar = None
        print(f"🏁 成功 {ok} · 失敗 {failed}")
        breakdown = self._timer.batch_done()
        if breakdown:
            print(breakdown)

    def file_start(self, idx, total, name):
        if self._f(): return self._f().file_start(idx, total, name)
        self._current_file_name = name
        self._timer.file_start()
        if self._batch_bar:
            self._batch_bar.set_postfix_str(name[:40])

    def file_done(self, name, ok, elapsed):
        if self._f(): return self._f().file_done(name, ok, elapsed)
        if self._page_bar:
            self._page_bar.close()
            self._page_bar = None
        breakdown = self._timer.file_done()
        if self._batch_bar:
            self._batch_bar.update(1)
        if breakdown and self._tqdm:
            self._tqdm.write(breakdown)

    def phase(self, phase_name, detail=""):
        if self._f(): return self._f().phase(phase_name, detail)
        self._timer.phase(phase_name)
        # 切 phase 時關掉 page bar
        if self._page_bar:
            self._page_bar.close()
            self._page_bar = None
        if self._batch_bar:
            tag = f"{self._current_file_name or '?'}|{phase_name}"
            self._batch_bar.set_postfix_str(tag[:60])

    def page_progress(self, curr, total, label=""):
        if self._f(): return self._f().page_progress(curr, total, label)
        if self._page_bar is None:
            self._page_bar = self._tqdm(
                total=total, desc="  pages", unit="pg", leave=False,
            )
        # 更新 absolute 位置而非增量（呼叫端可能跳號）
        delta = curr - self._page_bar.n
        if delta > 0:
            self._page_bar.update(delta)
        if curr >= total:
            self._page_bar.close()
            self._page_bar = None

    def warning(self, msg):
        if self._f(): return self._f().warning(msg)
        if self._tqdm:
            self._tqdm.write(f"⚠️  {msg}")

    def file_stats(self, stats):
        if self._f(): return self._f().file_stats(stats)
        if self._tqdm:
            self._tqdm.write(stats.summary_line())
