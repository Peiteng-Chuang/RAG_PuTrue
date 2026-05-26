"""Marker 輸出後處理工具。

A1：從 pipeline.py 抽出來，讓 orchestrator 不含 markdown 演算法。
目前只有 normalize_marker_headings；未來其他 Marker post-process 可放這。
"""
from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```")
_PIPE_RE = re.compile(r"^\s*\|")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def normalize_marker_headings(text: str) -> str:
    """把 Marker 輸出的 heading 壓平到 `### / ####`，跟 fast path 對齊。

    階層約定（見 [[etl-heading-hierarchy-convention]]）：
    - `##` = page anchor（stitcher 自動加）
    - `###` = page_title
    - `####` = subtitle / inline_heading

    Marker body 不應出現 `#` / `##`；`###` / `####` / `#####+` 都壓到 `####`。

    保護區段（不動其 `#`）：
    - fenced code（``` 圍住區塊）
    - pipe table（行首為 `|`）
    """
    lines = text.split("\n")
    out_lines: list[str] = []
    in_fence = False
    for ln in lines:
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            out_lines.append(ln)
            continue
        if in_fence or _PIPE_RE.match(ln):
            out_lines.append(ln)
            continue
        m = _HEADING_RE.match(ln)
        if m:
            hashes, rest = m.group(1), m.group(2)
            n = len(hashes)
            if n <= 2:           # #, ## → ###
                new = "###"
            elif n == 3:         # ### → ####
                new = "####"
            else:                # ####, #####, ###### → ####
                new = "####"
            out_lines.append(f"{new} {rest}")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)
