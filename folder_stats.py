"""指定資料夾 → 檔案類型統計（簡易版）。

用法：改下面的 TARGET_FOLDER，直接執行 `python folder_stats.py`。
統計底下所有檔案（遞迴）的「文件種類 / 副檔名」分佈：數量、佔用空間、
百分比、平均大小，並列出最大的幾個檔。純標準庫、不依賴 pandas。
"""
import os
import sys
from collections import defaultdict

# ========= 設定：改這裡 =========
TARGET_FOLDER = r"C:\project_file\RAG_PuTrue"   # 要分析的資料夾
TOP_N = 10                                       # 列出最大的 N 個檔（0=不列）
SHOW_BY_EXT = True                               # 是否額外列「依副檔名」明細
# ================================

# Windows console 用 UTF-8，避免中文在 cp950 crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 副檔名 → 文件種類（沿用 knowledge_base.py 的分類）
TAG_MAP = {
    ".pdf": "PDF 文件",
    ".docx": "Word 格式", ".doc": "Word 格式",
    ".txt": "純文字", ".md": "Markdown 筆記",
    ".pptx": "PowerPoint 簡報", ".ppt": "PowerPoint 簡報",
    ".csv": "CSV",
    ".xlsx": "Excel", ".xls": "Excel",
    ".json": "JSON", ".html": "網頁", ".htm": "網頁",
    ".jpg": "圖片", ".jpeg": "圖片", ".png": "圖片",
    ".gif": "圖片", ".bmp": "圖片", ".tif": "圖片", ".tiff": "圖片",
    ".dwg": "CAD 圖", ".dxf": "CAD 圖",
    ".zip": "壓縮檔", ".rar": "壓縮檔", ".7z": "壓縮檔",
}


def document_tag(ext):
    return TAG_MAP.get(ext.lower(), "其他")


def human_size(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} EB"


def scan(root):
    """遞迴掃描，回傳統計。用 os.walk，記憶體與檔案數無關。"""
    by_ext = defaultdict(lambda: [0, 0])   # ext -> [count, bytes]
    by_tag = defaultdict(lambda: [0, 0])   # tag -> [count, bytes]
    total_files = total_bytes = total_dirs = 0
    largest = []   # [(size, path), ...] 只留最大 TOP_N（掃完再排序，簡單版直接全收再截）
    errors = []

    for dirpath, dirnames, filenames in os.walk(root):
        total_dirs += len(dirnames)
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError as e:
                errors.append(f"{path} :: {e}")
                continue
            ext = os.path.splitext(name)[1].lower() or "(無副檔名)"
            by_ext[ext][0] += 1
            by_ext[ext][1] += size
            tag = document_tag(ext)
            by_tag[tag][0] += 1
            by_tag[tag][1] += size
            total_files += 1
            total_bytes += size
            if TOP_N > 0:
                largest.append((size, path))

    if TOP_N > 0:
        largest = sorted(largest, reverse=True)[:TOP_N]
    return by_ext, by_tag, total_files, total_bytes, total_dirs, largest, errors


def print_table(title, rows, total_files, total_bytes, label):
    print(f"\n{title}")
    print("-" * 72)
    print(f"{label:<20}{'數量':>8}{'數量%':>8}{'大小':>14}{'大小%':>8}{'平均':>12}")
    print("-" * 72)
    for key, (cnt, size) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        cpct = (cnt / total_files * 100) if total_files else 0
        spct = (size / total_bytes * 100) if total_bytes else 0
        avg = human_size(size // cnt) if cnt else "-"
        print(f"{key:<20}{cnt:>8}{cpct:>7.1f}%{human_size(size):>14}{spct:>7.1f}%{avg:>12}")
    print("-" * 72)
    print(f"{'合計':<20}{total_files:>8}{'100.0%':>8}{human_size(total_bytes):>14}{'100.0%':>8}")


def main():
    if not os.path.isdir(TARGET_FOLDER):
        print(f"找不到資料夾：{TARGET_FOLDER}")
        return
    print(f"掃描中：{os.path.abspath(TARGET_FOLDER)}")
    by_ext, by_tag, total_files, total_bytes, total_dirs, largest, errors = scan(TARGET_FOLDER)

    if total_files == 0:
        print("（此資料夾底下沒有檔案）")
        return

    print("\n" + "=" * 72)
    print(f"總覽：{total_files:,} 檔 · {total_dirs:,} 資料夾 · {human_size(total_bytes)}")
    print("=" * 72)

    print_table("依文件種類", by_tag, total_files, total_bytes, "文件種類")
    if SHOW_BY_EXT:
        print_table("依副檔名", by_ext, total_files, total_bytes, "副檔名")

    if TOP_N > 0 and largest:
        print(f"\n最大的 {len(largest)} 個檔")
        print("-" * 72)
        for size, path in largest:
            rel = os.path.relpath(path, TARGET_FOLDER)
            print(f"{human_size(size):>12}  {rel}")

    if errors:
        print(f"\n{len(errors)} 個檔無法讀取（權限等），前 5 筆：")
        for e in errors[:5]:
            print(f"   {e}")


if __name__ == "__main__":
    main()
