"""最終 MD 組裝。"""
from __future__ import annotations

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
        """R4 + P1：寫檔失敗 raise chained exception，outer pipeline.run 抓到後
        reporter.warning 會帶完整 path 與原始錯誤，不再額外印 stderr。"""
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
            raise RuntimeError(
                f"stitch 寫 .md 失敗 (path={out}): {type(e).__name__}: {e}"
            ) from e
