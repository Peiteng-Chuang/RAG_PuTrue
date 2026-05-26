"""S3：Marker per-page 平行化（process pool）。

⚠️ **未在 4090 驗證**。寫好等 batch session 在 4090 跑 bench 對比效能 + VRAM 峰值才知道是否有用。

Marker `PdfConverter.__call__()` 接單檔，無原生 batch API（讀過 marker/converters/pdf.py 確認）。
這裡走 B1：每個 worker process 啟動時建一個 PdfConverter，持久在 process 生命週期內。
主程式只負責切單頁 PDF + submit job + 收結果。

成本估算（per 4090 24GB）：
- Marker 模型載入 ~5-8GB VRAM/process
- max_workers=2 → 10-16GB，安全
- max_workers=3 → 15-24GB，邊緣，可能 OOM
- 超過 2 之前一定要先用 bench_v5.py 量 VRAM 峰值

用法：
    from mkd_generator_v5.marker_pool import MarkerPool

    with MarkerPool(max_workers=2) as pool:
        results = pool.process_batch(["page1.pdf", "page2.pdf"])
        # results = [{"text": str, "images": {name: bytes}}, ...]

跟 sequential `marker_converter(path)` 的差異：images 從 PIL Image dict 變成 bytes dict
（process 間 pickle PIL 慢，先 PNG 序列化）；主程式要 deserialize 回 PIL。
"""
from __future__ import annotations

import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any


# Worker-globals（per-process），__main__ 不可用
_worker_converter: Any = None


def _worker_init() -> None:
    """Worker process 啟動：build PdfConverter once，存 process global。

    若 marker 未裝會 raise，主程式應該在 main 用 try/except 包 with-block。

    TODO (S5)：torch.compile 整合 — Marker `PdfConverter` 內部 model 結構複雜，
    沒有暴露單一 model attr 可直接 compile。實際 compile 入口要看 marker 套件版本，
    可能要在 builders 或 processors 層手動 compile。4090 batch session 再研究，
    這裡先預留位置。試行作法：
        # if os.environ.get("MARKER_COMPILE", "0") == "1":
        #     import torch
        #     for attr in ("layout_model", "ocr_model"):  # 看 marker 版本
        #         m = getattr(_worker_converter, attr, None)
        #         if m is not None:
        #             try:
        #                 setattr(_worker_converter, attr, torch.compile(m))
        #             except Exception:
        #                 pass
    """
    global _worker_converter
    from marker.converters.pdf import PdfConverter  # noqa: WPS433
    from marker.models import create_model_dict     # noqa: WPS433
    _worker_converter = PdfConverter(artifact_dict=create_model_dict())


def _worker_run(pdf_path: str) -> dict:
    """Worker：跑 marker single-page PDF，回 dict({text, images_bytes})。"""
    global _worker_converter
    from marker.output import text_from_rendered  # noqa: WPS433
    rendered = _worker_converter(pdf_path)
    page_text, _, images = text_from_rendered(rendered)
    images_bytes: dict[str, bytes] = {}
    for k, img in (images or {}).items():
        buf = io.BytesIO()
        try:
            img.save(buf, format="PNG")
            images_bytes[k] = buf.getvalue()
        except Exception:
            continue
    return {"text": page_text or "", "images": images_bytes}


def deserialize_images(images_bytes: dict[str, bytes]) -> dict[str, Any]:
    """把 bytes dict 還原成 PIL Image dict（給 pipeline.image_extractor.rewrite_marker_images 用）。"""
    from PIL import Image  # noqa: WPS433
    out: dict[str, Any] = {}
    for k, data in (images_bytes or {}).items():
        try:
            out[k] = Image.open(io.BytesIO(data))
        except Exception:
            continue
    return out


class MarkerPool:
    """Persistent process pool，每 worker 自己 build PdfConverter。

    Context manager：__enter__ spawn workers + 觸發 _worker_init（每個 worker import marker
    + 載模型，首次有十幾秒成本），__exit__ shutdown wait。
    """

    def __init__(self, max_workers: int = 2):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "MarkerPool":
        # Windows 強制 spawn（fork 不可用），其他平台統一用 spawn 確保跨平台一致
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_worker_init,
            mp_context=ctx,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def process_batch(self, pdf_paths: list[str]) -> list[dict]:
        """Submit 全部、依輸入順序回結果。任一 worker raise → 整批 raise。

        Failed result 是 `{"text": "", "images": {}, "error": "..."}` 形式（fail-soft）。
        """
        if self._executor is None:
            raise RuntimeError("MarkerPool 必須用 with-block，未進 __enter__")
        futures = [self._executor.submit(_worker_run, p) for p in pdf_paths]
        results: list[dict] = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"text": "", "images": {}, "error": str(e)})
        return results
