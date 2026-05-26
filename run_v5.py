"""mkd_generator_v5 launcher。

⚠️ 預設安全（[[feedback-data-dependency-audit-first]]）：
- 已標記檔（sidecar.review_status ∈ {processing/encoded/ingested}）會被 SAFE-SKIP，
  保護使用者標記成果跟下游 Qdrant chunks 對應關係。
- 強制覆寫已標記檔需要明確傳 --allow-overwrite-labeled（強烈不建議）。

常用：
    # 看會跑哪些檔、哪些被保護（dry-run；零 disk 寫入）
    python run_v5.py --preview

    # 跑單檔（仍會檢查該檔 sidecar，被保護就 abort）
    python run_v5.py --single "D:/.../xxx.pptx"

    # 跑批次（safe-skip + resume-skip，建議的標準跑法）
    python run_v5.py

    # 不要 marker（CAD 頁留 placeholder）
    python run_v5.py --no-marker
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 確保 stdout/stderr 用 UTF-8，避免 emoji 在 cp950 console 或 pipe 環境 crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 對齊 v4 (mkd_generator_v4.py:6)：torch first-time import 在這台機器 2-3 秒，
# 冷啟動或慢磁碟可能 30s-2min（PyTorch 2.11 載入 quantization → ao → fx.passes 等大量子模組）。
# 若 torch 不在程式啟動 top-level 載入，會延遲到 _build_marker_converter() 內
# 由 marker 內部觸發，發生在「📋 掃描結果」訊息之後，使用者誤判為 marker 卡住。
# 預先 import 將等待移到程式啟動瞬間並先報訊號，與 v4 行為對齊。
print("[啟動] 載入 PyTorch 中...（typically 2-15s，冷啟動可能更久，請耐心等候）",
      flush=True)
import torch  # noqa: F401（不直接使用，純為觸發模組載入）
print("[啟動] PyTorch 已就緒", flush=True)

# S2 註記：若想用 expandable_segments（PyTorch 2.1+），在 shell 自己 export：
#   PowerShell: $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
#   Bash:       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 預設不主動設，避免老 PyTorch / driver 組合 silent crash。


DEFAULT_DATA_ROOT = r"C:/project_file/RAG_PuTrue/璞真RAG_rawdata"
DEFAULT_OUTPUT_ROOT = "./mkdata"


def _build_marker_converter():
    """Marker 套件 import + 模型 weights 載入。

    torch 已在程式 top-level 預先 import，所以本函式 [1/2] 階段只剩 marker 自己的
    submodule（Surya / OCR / processors），通常 5-15 秒。[2/2] model weights 載入
    視磁碟 I/O 與 CUDA cache 狀態，30 秒至數分鐘。
    """
    import traceback
    print("[1/2] 載入 Marker 套件 submodule 中（torch 已預先載入，此段 5-15s）...",
          flush=True)
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as e:
        print(f"⚠️ Marker 套件未裝。CAD 頁將留 placeholder。")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"⚠️ Marker import 例外（非 ImportError）。CAD 頁將留 placeholder。")
        traceback.print_exc()
        return None
    print("[2/2] 載入 Marker 模型 weights（首次 30s-2min，後續較快；請勿中斷）...",
          flush=True)
    try:
        return PdfConverter(artifact_dict=create_model_dict())
    except Exception as e:
        print(f"⚠️ Marker 模型載入失敗。CAD 頁將留 placeholder。")
        traceback.print_exc()
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="mkd_generator_v5 batch runner")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--single", default=None,
                   help="只跑這一個檔；仍會 sidecar 檢查")
    p.add_argument("--no-marker", action="store_true")
    p.add_argument("--reporter", choices=["console", "tqdm", "silent"], default="console")
    p.add_argument("--preview", action="store_true",
                   help="只列分類（will_process / safe_skip / resume_skip），不跑 pipeline")
    p.add_argument("--no-resume-skip", action="store_true",
                   help="不要 skip tracker 標 SUCCESS 的（強制重跑，但仍會保護 labeled 檔）")
    p.add_argument("--allow-overwrite-labeled", action="store_true",
                   help="⚠️ 危險：覆寫已標記的檔（review_status != unprocessed）。"
                        "會破壞使用者標記資料 + 失效 Qdrant chunks 對應。"
                        "建議改用：先手動刪掉那幾個 .review.json 而非用此 flag。")
    p.add_argument("--marker-workers", type=int, default=0,
                   help="S3：>0 啟用 Marker process pool（每 worker 自己 build PdfConverter，"
                        "4090 上 max=2 安全）。預設 0 = sequential。**需 4090 驗證 VRAM 峰值**")
    p.add_argument("--file-workers", type=int, default=0,
                   help="S5：>1 啟用 file-level process pool（每 worker 自建 pipeline+marker）。"
                        "與 --marker-workers 互斥（VRAM 不夠）。預設 0 = 序列。"
                        "**需 4090 驗證**")
    args = p.parse_args()

    # S5 mutex：兩種平行化加總 VRAM 會炸
    if args.marker_workers > 0 and args.file_workers > 1:
        print("❌ --marker-workers 與 --file-workers 互斥（VRAM 不夠）。請二擇一。")
        return 1

    from mkd_generator_v5 import ConsoleReporter, TqdmReporter, SilentReporter
    reporter = {
        "console": ConsoleReporter(),
        "tqdm": TqdmReporter(),
        "silent": SilentReporter(),
    }[args.reporter]

    # --- 單檔模式 ---
    if args.single:
        from mkd_generator_v5 import default_pipeline
        from mkd_generator_v5.batch import (
            _derive_sidecar_path, _load_sidecar_status, _is_protected,
        )
        single_path = Path(args.single).absolute()
        output_root = Path(args.output_root).absolute()
        if not single_path.exists():
            print(f"❌ 找不到檔：{single_path}")
            return 1
        # 也檢查 sidecar
        sidecar = _derive_sidecar_path(output_root, single_path)
        status = _load_sidecar_status(sidecar)
        if _is_protected(status) and not args.allow_overwrite_labeled:
            print(f"🛡️ SAFE-SKIP：{single_path.name} 的 sidecar review_status={status}")
            print(f"   保護你的標記成果。如要強制覆寫，加 --allow-overwrite-labeled（不建議）")
            print(f"   或手動刪：{sidecar}")
            return 0
        marker = None if args.no_marker else _build_marker_converter()
        pl = default_pipeline(
            marker_converter=marker, reporter=reporter,
            marker_pool_workers=args.marker_workers,
        )
        print(f"📄 {single_path.name} → {args.output_root}/")
        ok = pl.run(single_path, args.output_root)
        return 0 if ok else 1

    # --- 批次模式 ---
    data_root = Path(args.data_root).absolute()
    output_root = Path(args.output_root).absolute()
    if not data_root.exists():
        print(f"❌ 找不到 data_root：{data_root}")
        return 1

    from mkd_generator_v5 import default_pipeline, BatchProcessor
    bp = BatchProcessor(
        data_root=data_root,
        output_root=output_root,
        pipeline=default_pipeline(marker_converter=None, reporter=reporter,
                                  marker_pool_workers=args.marker_workers),
        reporter=reporter,
    )

    # Preview 分類
    summary = bp.preview()
    print(f"\n📋 掃描結果：{summary['total']} 個檔")
    print(f"  🛡️ SAFE-SKIP（已標記，受保護）: {len(summary['safe_skip_labeled'])} 個")
    for fk, status in summary["safe_skip_labeled"][:10]:
        print(f"     · [{status}] {fk}")
    if len(summary["safe_skip_labeled"]) > 10:
        print(f"     · ... 還有 {len(summary['safe_skip_labeled']) - 10} 個")
    print(f"  ⏭ RESUME-SKIP（tracker 標 SUCCESS）: {len(summary['resume_skip_success'])} 個")
    print(f"  ▶ WILL-PROCESS: {len(summary['will_process'])} 個")
    for fk in summary["will_process"][:10]:
        print(f"     · {fk}")
    if len(summary["will_process"]) > 10:
        print(f"     · ... 還有 {len(summary['will_process']) - 10} 個")

    if args.preview:
        print("\n(--preview；不跑 pipeline)")
        return 0

    if args.allow_overwrite_labeled and summary["safe_skip_labeled"]:
        print(f"\n⚠️ --allow-overwrite-labeled 已開啟，"
              f"{len(summary['safe_skip_labeled'])} 個已標記檔會被覆寫！")
        try:
            ans = input("確定？[y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            print("已取消。")
            return 0

    if not summary["will_process"] and not args.allow_overwrite_labeled:
        print("\n(沒有檔要處理)")
        return 0

    # 真正跑
    if args.file_workers > 1:
        # S5 file-level pool：每 worker 自建 pipeline，主程式不需要 build marker
        print(f"🚀 file-level pool 啟用：{args.file_workers} workers")
        ok, failed = bp.process_all_parallel(
            workers=args.file_workers,
            no_marker=args.no_marker,
            skip_success=not args.no_resume_skip,
            allow_overwrite_labeled=args.allow_overwrite_labeled,
        )
    else:
        marker = None if args.no_marker else _build_marker_converter()
        bp.pipeline = default_pipeline(
            marker_converter=marker, reporter=reporter,
            marker_pool_workers=args.marker_workers,
        )
        ok, failed = bp.process_all(
            skip_success=not args.no_resume_skip,
            allow_overwrite_labeled=args.allow_overwrite_labeled,
        )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
