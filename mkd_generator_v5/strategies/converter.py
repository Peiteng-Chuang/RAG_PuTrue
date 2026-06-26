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

    def __init__(self, soffice_path: str | None = None, timeout: int = 180):
        self._soffice = soffice_path or self._find_soffice()
        self._timeout = timeout

    @staticmethod
    def _find_soffice() -> str:
        if platform.system() == "Windows":
            p = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
            return str(p) if p.exists() else "soffice.exe"
        return "libreoffice"

    def supports(self, ext: str) -> bool:
        return ext.lower() in self.SUPPORTED_EXTS

    def convert(self, src: Path, tmp_dir: Path) -> Path:
        # 每檔獨立的 UserInstallation profile，避免 (1) 殘留 lock 卡住 headless 轉檔、
        # (2) 多 worker 並行時共用預設 profile 互相鎖死。
        profile_dir = (tmp_dir / f"_lo_profile_{src.stem}").absolute()
        profile_uri = profile_dir.as_uri()
        try:
            subprocess.run(
                [self._soffice, "--headless",
                 f"-env:UserInstallation={profile_uri}",
                 "--convert-to", "pdf",
                 "--outdir", str(tmp_dir), str(src)],
                check=True, capture_output=True, timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as e:
            # 不讓 LibreOffice hang 卡死整檔；轉成可被 pipeline 捕捉的錯誤
            raise RuntimeError(
                f"LibreOffice 轉檔逾時（>{self._timeout}s）：{src.name}"
            ) from e
        return tmp_dir / f"{src.stem}.pdf"
