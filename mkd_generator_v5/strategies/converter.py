"""格式轉換：PPT/PPTX → PDF。"""
from __future__ import annotations

import platform
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class FormatConverter(ABC):
    """把非 PDF 輸入轉成 PDF（若已是 PDF 則 supports() 回 False，pipeline 直接用原檔）。"""

    @abstractmethod
    def supports(self, ext: str) -> bool: ...

    @abstractmethod
    def convert(self, src: Path, tmp_dir: Path) -> Path: ...


class LibreOfficeConverter(FormatConverter):
    """v4 等價：用 LibreOffice headless 轉檔。

    Windows 預設找 `C:\\Program Files\\LibreOffice\\program\\soffice.exe`，
    其他平台用 `libreoffice` PATH。"""

    SUPPORTED_EXTS = {".ppt", ".pptx"}

    def __init__(self, soffice_path: str | None = None):
        self._soffice = soffice_path or self._find_soffice()

    @staticmethod
    def _find_soffice() -> str:
        if platform.system() == "Windows":
            p = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
            return str(p) if p.exists() else "soffice.exe"
        return "libreoffice"

    def supports(self, ext: str) -> bool:
        return ext.lower() in self.SUPPORTED_EXTS

    def convert(self, src: Path, tmp_dir: Path) -> Path:
        subprocess.run(
            [self._soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(tmp_dir), str(src)],
            check=True, capture_output=True,
        )
        return tmp_dir / f"{src.stem}.pdf"
