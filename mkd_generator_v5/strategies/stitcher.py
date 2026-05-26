"""最終 MD 組裝。"""
from __future__ import annotations

import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

from ..types import PageFragment


class Stitcher(ABC):
    @abstractmethod
    def stitch(self, fragments: list[PageFragment], meta: dict, out: Path) -> None: ...


class PageFragmentStitcher(Stitcher):
    """v4 等價：
    ```
    # {filename}

    ---
    extracted_main_title: "{title}"
    file_hash: "{md5}"
    ---

    ## 第 N 頁
    ### {title}

    {body}

    ---
    (重複每頁)
    ```
    meta keys: filename_stem / main_title / file_hash
    """

    def stitch(self, fragments, meta, out):
        """R4: 包 try/except，failed 印明確 warning 含 path + error 後 raise
        讓 outer pipeline.run 仍標 FAILED。"""
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(f"# {meta['filename_stem']}\n\n")
                f.write("---\n")
                f.write(f"extracted_main_title: \"{meta.get('main_title', 'Unknown')}\"\n")
                f.write(f"file_hash: \"{meta.get('file_hash', '')}\"\n")
                f.write("---\n\n")
                for frag in fragments:
                    f.write(frag.render())
        except Exception as e:
            print(
                f"[ERROR] stitch 寫 .md 失敗 (path={out}, "
                f"{type(e).__name__}: {e})",
                file=sys.stderr, flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            raise
