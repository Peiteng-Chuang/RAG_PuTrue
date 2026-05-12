"""
一次性清理：把 process_tracker.json 內所有 .ppt/.pptx 條目移除，
並刪除對應的 .md 檔（若存在），讓 v4 重新處理這些檔案。
不動 PDF 條目。
"""
import json
from pathlib import Path

MKDATA = Path("./mkdata")
TRACKER = MKDATA / "process_tracker.json"

with open(TRACKER, "r", encoding="utf-8") as f:
    tracker = json.load(f)

removed = []
for key in list(tracker.keys()):
    suffix = Path(key.replace("\\", "/")).suffix.lower()
    if suffix in (".ppt", ".pptx"):
        removed.append(key)
        del tracker[key]
        # 刪 md 檔
        stem = Path(key.replace("\\", "/")).stem
        md = MKDATA / f"{stem}.md"
        if md.exists():
            md.unlink()
            print(f"🗑️  removed md: {md.name}")
        else:
            print(f"   (md not found): {md.name}")

with open(TRACKER, "w", encoding="utf-8") as f:
    json.dump(tracker, f, ensure_ascii=False, indent=4)

print(f"\n✅ 從 tracker 移除 {len(removed)} 筆 PPT/PPTX 紀錄")
print(f"剩餘 PDF 紀錄 {len(tracker)} 筆")
