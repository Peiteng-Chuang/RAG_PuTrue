"""mkd_generator_v5 launcher。

用法：
    # 1. 跑整個 data_root（預設用 [[hardware-environment]] 內的路徑）
    python run_v5.py

    # 2. 跑單檔
    python run_v5.py --single "D:/璞真RAG資料夾/12.個案銷講資料/勤美之真/xxx.pptx"

    # 3. 換進度回報器
    python run_v5.py --reporter tqdm

    # 4. 跳過 marker（純 fast path，無 CAD 頁時可省 GPU）
    python run_v5.py --no-marker

    # 5. dry-run（只列要處理的檔，不跑 pipeline）
    python run_v5.py --dry-run

⚠️ 安全提醒：v5 跑完會**覆寫 mkdata/ 內對應 MD 檔**，跟之前 v4 災難同機制。
   建議跑前先 `git add mkdata && git commit -m "snapshot before v5 run"` 把當前狀態存進 git。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# === 預設路徑（可由 CLI 覆寫）===
DEFAULT_DATA_ROOT = r"D:/璞真RAG資料夾/12.個案銷講資料"
DEFAULT_OUTPUT_ROOT = "./mkdata"


def _build_marker_converter():
    """嘗試載入 marker；失敗時回 None 並印警告（pipeline 在 marker page 會留 placeholder）。"""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        print("🔧 載入 Marker 模型（首次需數十秒）...")
        return PdfConverter(artifact_dict=create_model_dict())
    except ImportError as e:
        print(f"⚠️ Marker 套件未裝（{e}）。 CAD 頁將留 placeholder。")
        return None
    except Exception as e:
        print(f"⚠️ Marker 模型載入失敗（{e}）。 CAD 頁將留 placeholder。")
        return None


def _list_inputs(data_root: Path, exts=("*.pdf", "*.ppt", "*.pptx")) -> list[Path]:
    files: list[Path] = []
    for ext in exts:
        files.extend(data_root.rglob(ext))
    return sorted(files)


def main() -> int:
    p = argparse.ArgumentParser(description="mkd_generator_v5 batch runner")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help=f"輸入資料夾（預設：{DEFAULT_DATA_ROOT}）")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                   help=f"輸出資料夾（預設：{DEFAULT_OUTPUT_ROOT}）")
    p.add_argument("--single", default=None,
                   help="只跑這一個檔（覆蓋 --data-root 行為，輸出仍進 --output-root）")
    p.add_argument("--no-marker", action="store_true",
                   help="跳過 Marker，純 fast path（無 GPU / 速度優先）")
    p.add_argument("--reporter", choices=["console", "tqdm", "silent"], default="console",
                   help="進度回報器（預設 console）")
    p.add_argument("--dry-run", action="store_true",
                   help="只列要處理的檔，不跑 pipeline")
    p.add_argument("--no-skip", action="store_true",
                   help="不要 skip tracker 標 SUCCESS 的檔（強制重跑）")
    p.add_argument("--yes", action="store_true",
                   help="跳過覆寫確認（自動化用）")
    args = p.parse_args()

    # reporter
    from mkd_generator_v5 import ConsoleReporter, TqdmReporter, SilentReporter
    reporter = {
        "console": ConsoleReporter(),
        "tqdm": TqdmReporter(),
        "silent": SilentReporter(),
    }[args.reporter]

    # 單檔模式
    if args.single:
        from mkd_generator_v5 import default_pipeline
        single_path = Path(args.single).absolute()
        if not single_path.exists():
            print(f"❌ 找不到檔：{single_path}")
            return 1
        marker = None if args.no_marker else _build_marker_converter()
        pl = default_pipeline(marker_converter=marker, reporter=reporter)
        print(f"📄 {single_path.name} → {args.output_root}/")
        ok = pl.run(single_path, args.output_root)
        return 0 if ok else 1

    # 批次模式
    data_root = Path(args.data_root).absolute()
    output_root = Path(args.output_root).absolute()
    if not data_root.exists():
        print(f"❌ 找不到 data_root：{data_root}")
        return 1

    inputs = _list_inputs(data_root)
    print(f"🔍 找到 {len(inputs)} 個檔")
    if args.dry_run:
        for f in inputs:
            print(f"  {f.relative_to(data_root)}")
        print("\n(dry-run；不跑 pipeline)")
        return 0

    if not inputs:
        print("(沒有可處理的檔)")
        return 0

    # ⚠️ 覆寫提醒
    if not args.yes:
        mkdata_md_count = len(list(output_root.glob("*.md"))) if output_root.exists() else 0
        if mkdata_md_count > 0:
            print(f"\n⚠️  {output_root} 內已有 {mkdata_md_count} 個 .md 檔。")
            print(f"   v5 跑完會 **覆寫** 對應檔的 .md（手動編輯會丟）。")
            print(f"   建議先 `git add {args.output_root} && git commit` 存檔。\n")
            try:
                ans = input("繼續？[y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans not in ("y", "yes"):
                print("已取消。")
                return 0

    # 跑
    from mkd_generator_v5 import default_pipeline, BatchProcessor
    marker = None if args.no_marker else _build_marker_converter()
    pipeline = default_pipeline(marker_converter=marker, reporter=reporter)
    bp = BatchProcessor(
        data_root=data_root,
        output_root=output_root,
        pipeline=pipeline,
        reporter=reporter,
    )
    ok, failed = bp.process_all(skip_success=not args.no_skip)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
