import argparse
import sys
import subprocess
from pathlib import Path

# 定義支援的副檔名
PPT_EXTS = {".ppt", ".pptx", ".pptm"}

def convert_one(ppt_path: Path, out_dir: Path) -> bool:
    """
    使用 LibreOffice Headless 模式將單一 PPT 轉為 PDF
    """
    try:
        # LibreOffice 轉換指令
        # --headless: 不啟動圖形介面
        # --convert-to pdf: 轉檔格式
        # --outdir: 輸出的資料夾位置
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to", "pdf",
            str(ppt_path.resolve()),
            "--outdir", str(out_dir.resolve())
        ]
        
        # 執行指令並捕獲錯誤
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL {ppt_path.name}: 指令執行錯誤 - {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"FAIL {ppt_path.name}: 未知錯誤 - {e}", file=sys.stderr)
        return False

def collect_inputs(src: Path) -> list[Path]:
    """
    收集所有符合條件的 PPT 檔案
    """
    if src.is_file():
        if src.suffix.lower() not in PPT_EXTS:
            raise ValueError(f"非支援的 PPT 格式: {src}")
        return [src]
    if src.is_dir():
        # 使用 rglob 遞迴搜尋所有子資料夾
        return sorted(p for p in src.rglob("*") if p.suffix.lower() in PPT_EXTS)
    raise FileNotFoundError(f"找不到路徑: {src}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Ubuntu 版 PPT/PPTX → PDF 轉換工具 (基於 LibreOffice)")
    parser.add_argument("input", help="輸入單一 PPT 檔案或資料夾路徑")
    parser.add_argument("-o", "--output", help="輸出 PDF 的資料夾（預設與來源同層）")
    args = parser.parse_args()

    src = Path(args.input)
    
    # 檢查輸入是否存在
    if not src.exists():
        print(f"錯誤：找不到輸入路徑 {src}")
        return 1

    files = collect_inputs(src)
    if not files:
        print("找不到可轉檔的 PPT 檔案。")
        return 1

    print(f"找到 {len(files)} 個檔案，準備開始轉檔...")

    fail_count = 0
    for ppt in files:
        # 決定輸出目錄：如果有指定 -o 則用之，否則用檔案所在目錄
        target_out_dir = Path(args.output) if args.output else ppt.parent
        target_out_dir.mkdir(parents=True, exist_ok=True)

        # 執行轉換
        success = convert_one(ppt, target_out_dir)
        
        if success:
            expected_pdf = target_out_dir / (ppt.stem + ".pdf")
            print(f"OK   {ppt.name} -> {expected_pdf.name}")
        else:
            fail_count += 1

    print("-" * 30)
    print(f"完成！成功: {len(files) - fail_count}, 失敗: {fail_count}")
    
    return 0 if fail_count == 0 else 2

if __name__ == "__main__":
    sys.exit(main())