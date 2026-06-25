"""rebuild_tracker — 依「目前檔案樹結構」重建 process_tracker.json 的 file_key，並偵測檔案樹變動。

用途
----
v5 的 file_key = `relative_to(data_root)`。當你把 raw 檔重新整理（例如搬進「類別/建案/」
結構）或改了 data_root，file_key 就跟著變，但你**不需要重跑 v5 重產 .md**——.md / 向量
都是 stem-based、與路徑無關。這支工具只做兩件事：
  1. 掃描目前 data_root，把 process_tracker.json 的 key 重建成「目前樹」的相對路徑。
  2. 對照上次快照（file_tree_snapshot.json），回報 搬移 / 新增 / 消失。

之後到 md_review_ui 的 fast pipeline「♻️ 全部重來 → 更新資料庫」即可把新 file_key 帶出的
project_name / doc_type 重新 upsert（同 point_id 原地覆蓋，向量走快取、不重算）。

安全（[[feedback-data-dependency-audit-first]]）
----
- **預設 dry-run**：只列差異與重建結果，不寫任何檔。確認後加 `--apply` 才實寫。
- `--apply` 會先**備份**舊 tracker 成 `process_tracker.json.bak.<時間戳>`，再原子寫入。
- **stem 衝突**（不同路徑、相同檔名）會破壞 stem-based 的 .md/sidecar/向量 → 預設**拒絕 apply**，
  除非加 `--force`（強烈不建議）。

常用
----
    # 看會怎麼重建 + 檔案樹變了什麼（dry-run，零寫入）
    python rebuild_tracker.py --data-root "C:/project_file/RAG_PuTrue/璞真RAG資料夾"

    # 確認無誤後實寫（會備份舊 tracker）
    python rebuild_tracker.py --data-root "C:/project_file/RAG_PuTrue/璞真RAG資料夾" --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 stdout，避免 emoji 在 cp950 console / pipe crash（對齊 run_v5.py）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_DATA_ROOT = r"C:/project_file/RAG_PuTrue/璞真RAG資料夾"
DEFAULT_OUTPUT_ROOT = "./mkdata"
EXTENSIONS = ("*.pdf", "*.ppt", "*.pptx")
TRACKER_NAME = "process_tracker.json"
SNAPSHOT_NAME = "file_tree_snapshot.json"


def scan_tree(data_root: Path) -> dict[str, dict]:
    """掃描 data_root 下所有 PDF/PPT/PPTX，回 {file_key: {stem, size, mtime}}。

    file_key 用 `str(relative_to(data_root))`，與 v5 BatchProcessor 完全一致（Windows 為反斜線），
    確保 md_review_ui 讀得到、且不影響 stem-based 的下游檔名對應。
    """
    files: dict[str, dict] = {}
    for ext in EXTENSIONS:
        for p in data_root.rglob(ext):
            if not p.is_file():
                continue
            file_key = str(p.relative_to(data_root))
            try:
                st = p.stat()
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                size, mtime = 0, 0.0
            files[file_key] = {"stem": p.stem, "size": size, "mtime": mtime}
    return files


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def stem_of(file_key: str) -> str:
    """file_key（可能含 / 或 \\）→ 檔名 stem。"""
    return Path(file_key.replace("\\", "/")).stem


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 process_tracker.json 的 file_key + 偵測檔案樹變動")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="要掃描並作為 file_key 相對基底的根目錄")
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="mkdata（tracker / .md / 快照所在）")
    ap.add_argument("--apply", action="store_true", help="實際寫入 tracker + 快照（預設只 dry-run）")
    ap.add_argument("--force", action="store_true", help="即使有 stem 衝突也照寫（不建議）")
    args = ap.parse_args()

    data_root = Path(args.data_root).absolute()
    output_root = Path(args.output_root).absolute()
    tracker_path = output_root / TRACKER_NAME
    snapshot_path = output_root / SNAPSHOT_NAME

    if not data_root.exists():
        print(f"❌ 找不到 data_root：{data_root}")
        return 1

    # 1) 掃目前樹
    current = scan_tree(data_root)
    print(f"=== 掃描 {data_root} ===")
    print(f"找到 {len(current)} 個檔（{' / '.join(EXTENSIONS)}）\n")

    # 2) stem 衝突偵測（不同路徑、同檔名 → .md/sidecar/向量會互相覆蓋）
    stem_to_keys: dict[str, list[str]] = {}
    for fk, info in current.items():
        stem_to_keys.setdefault(info["stem"], []).append(fk)
    collisions = {s: ks for s, ks in stem_to_keys.items() if len(ks) > 1}
    if collisions:
        print(f"🟥 stem 衝突 {len(collisions)} 組（同檔名不同路徑 → 會破壞 stem-based 對應）：")
        for s, ks in list(collisions.items())[:20]:
            print(f"   · {s}：")
            for k in ks:
                print(f"        {k}")
        if len(collisions) > 20:
            print(f"   · …另 {len(collisions) - 20} 組")
        print()

    # 3) 對照上次狀態（優先用快照；無快照則退回舊 tracker 的 keys）
    snapshot = load_json(snapshot_path)
    old_tracker = load_json(tracker_path)
    if snapshot.get("files"):
        prior_keys = list(snapshot["files"].keys())
        prior_src = "快照"
    else:
        prior_keys = list(old_tracker.keys())
        prior_src = "舊 tracker"
    prior_stems = {stem_of(k): k for k in prior_keys}
    cur_stems = {info["stem"]: fk for fk, info in current.items()}

    moved, new_with_md, new_without_md, disappeared, unchanged = [], [], [], [], []
    for stem, fk in cur_stems.items():
        has_md = (output_root / f"{stem}.md").exists()
        if stem in prior_stems:
            old_fk = prior_stems[stem]
            if old_fk != fk:
                moved.append((old_fk, fk))
            else:
                unchanged.append(fk)
        else:
            (new_with_md if has_md else new_without_md).append(fk)
    for stem, old_fk in prior_stems.items():
        if stem not in cur_stems:
            disappeared.append(old_fk)

    # 4) 回報差異（vs {prior_src}）
    print(f"--- 檔案樹變動（對照 {prior_src}）---")
    print(f"🔀 搬移／改路徑 (re-keyed): {len(moved)}")
    for old_fk, fk in moved[:20]:
        print(f"   · {old_fk}  →  {fk}")
    if len(moved) > 20:
        print(f"   · …另 {len(moved) - 20} 筆")
    print(f"🆕 新增: {len(new_with_md) + len(new_without_md)}"
          f"（有 .md: {len(new_with_md)}／無 .md 需跑 v5: {len(new_without_md)}）")
    for fk in (new_with_md + new_without_md)[:20]:
        print(f"   · {fk}")
    print(f"❌ 消失（在{prior_src}、現已不在樹中）: {len(disappeared)}")
    for fk in disappeared[:20]:
        print(f"   · {fk}")
    if len(disappeared) > 20:
        print(f"   · …另 {len(disappeared) - 20} 筆")
    print(f"✅ 不變: {len(unchanged)}\n")

    # 5) 重建 tracker：只納入「有 .md（已處理、md_review_ui 開得起來）」的檔 → SUCCESS。
    #    無 .md 的新檔不納入（避免 md_review_ui 選到開不了的項目），列在上方「需跑 v5」。
    new_tracker = {
        fk: "SUCCESS"
        for fk, info in current.items()
        if (output_root / f"{info['stem']}.md").exists()
    }
    print("--- tracker 重建結果 ---")
    print(f"新 tracker：{len(new_tracker)} 筆（有 .md → SUCCESS）")
    print(f"未納入（無 .md，需先跑 v5）：{len(new_without_md)} 筆")
    print(f"舊 tracker：{len(old_tracker)} 筆 → 新 tracker：{len(new_tracker)} 筆\n")

    # 6) 寫入（或 dry-run）
    if not args.apply:
        print("📋 dry-run：未寫入任何檔。確認無誤後加 --apply 實寫（會先備份舊 tracker）。")
        return 0

    if collisions and not args.force:
        print("🟥 偵測到 stem 衝突，為保護 stem-based 的 .md/sidecar/向量，拒絕寫入。")
        print("   請先處理同名檔（改名或分開），或確知風險後加 --force。")
        return 1

    # 備份舊 tracker
    if tracker_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = tracker_path.with_name(f"{TRACKER_NAME}.bak.{ts}")
        backup.write_text(tracker_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"🗄 已備份舊 tracker → {backup.name}")

    save_json_atomic(tracker_path, new_tracker)
    save_json_atomic(snapshot_path, {
        "data_root": str(data_root),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "files": current,
    })
    print(f"✅ 已寫入 {tracker_path.name}（{len(new_tracker)} 筆）+ {snapshot_path.name}")
    print("👉 下一步：md_review_ui → fast pipeline「♻️ 全部重來 → 更新資料庫」重 upsert payload；"
          "並重啟 app 清 load_tracker 快取。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
