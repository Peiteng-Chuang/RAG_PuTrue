"""圖檔抽取：fast (fitz) + slow (marker) 兩路徑。

W5/C — **統一命名 `{stem}_h{md5[:8]}.{ext}`**：fast 跟 slow 都用 hash 命名，
跨路徑 dedup（saved_image_hashes / hash_to_filename）。
"""
from __future__ import annotations

import hashlib
import io
import re
from abc import ABC, abstractmethod
from typing import Any

from ..types import ExtractContext, ImageRef
from .template import _visible_image_xrefs_on_page


class ImageExtractor(ABC):
    """fast 與 slow 路徑共用同一個 strategy；slow 多一個 marker_images 參數。"""

    @abstractmethod
    def extract_fast(self, page, ctx: ExtractContext) -> list[ImageRef]:
        """從 fitz page 直接抽圖（fast path）。"""

    @abstractmethod
    def rewrite_marker_images(
        self,
        marker_text: str,
        marker_images: dict[str, Any],
        ctx: ExtractContext,
    ) -> str:
        """處理 marker 輸出的 image dict，存檔 + 改寫 marker_text 內的 ![]() 引用。

        回改寫後的 markdown text。"""


def _bytes_of(img_obj: Any, fallback_ext: str = "png") -> tuple[bytes, str] | None:
    """marker 圖片可能是 bytes / PIL Image / 別的。統一轉成 (bytes, ext)。

    回 None 表示這張圖不能轉成 bytes（跳過）。"""
    if isinstance(img_obj, (bytes, bytearray)):
        return bytes(img_obj), fallback_ext
    if hasattr(img_obj, "save"):  # PIL.Image.Image
        try:
            fmt = (getattr(img_obj, "format", None) or fallback_ext).lower()
            # PIL format → 副檔名（jpeg → jpg）
            ext_map = {"jpeg": "jpg"}
            ext = ext_map.get(fmt, fmt)
            buf = io.BytesIO()
            img_obj.save(buf, format=fmt.upper() if fmt != "jpg" else "JPEG")
            return buf.getvalue(), ext
        except Exception:
            return None
    return None


class HashNamedImageExtractor(ImageExtractor):
    """v5/C 主實作：fast 跟 slow 路徑都用 `{stem}_h{hash[:8]}.{ext}` 命名。"""

    def extract_fast(self, page, ctx: ExtractContext) -> list[ImageRef]:
        refs: list[ImageRef] = []
        visible = _visible_image_xrefs_on_page(page)
        for img_info in page.get_images(full=True):
            try:
                xref = img_info[0]
                if xref not in visible:
                    continue

                # 取 hash（優先吃 template scan 算過的 cache）
                img_hash = ctx.filter_state.xref_to_hash_cache.get(xref)
                if not img_hash:
                    base_img = ctx.doc.extract_image(xref)
                    img_hash = hashlib.md5(base_img["image"]).hexdigest()
                    ctx.filter_state.xref_to_hash_cache[xref] = img_hash

                # banned
                if img_hash in ctx.filter_state.banned_image_hashes:
                    continue

                # 已存過 → 直接引用，不重存
                if img_hash in ctx.saved_image_hashes:
                    name = ctx.hash_to_filename[img_hash]
                    refs.append(ImageRef(
                        md_path=f"{ctx.folder_name}/{name}",
                        abs_path=ctx.img_dir / name,
                        img_hash=img_hash,
                    ))
                    continue

                base_img = ctx.doc.extract_image(xref)
                if base_img["width"] < ctx.min_image_dim or base_img["height"] < ctx.min_image_dim:
                    continue

                ext = base_img["ext"]
                img_name = f"{ctx.stem}_h{img_hash[:8]}.{ext}"
                (ctx.img_dir / img_name).write_bytes(base_img["image"])
                ctx.saved_image_hashes.add(img_hash)
                ctx.hash_to_filename[img_hash] = img_name
                refs.append(ImageRef(
                    md_path=f"{ctx.folder_name}/{img_name}",
                    abs_path=ctx.img_dir / img_name,
                    img_hash=img_hash,
                ))
            except Exception:
                # 單張失敗不擋其他；reporter 由呼叫端負責
                continue
        return refs

    def rewrite_marker_images(
        self,
        marker_text: str,
        marker_images: dict[str, Any],
        ctx: ExtractContext,
    ) -> str:
        """Marker 輸出的每張圖：算 hash → dedup → 存檔（_h{hash}.{ext}）→ 改寫 marker_text 內引用。"""
        new_text = marker_text
        for orig_name, img_obj in marker_images.items():
            # fallback ext 取自 orig_name 副檔名
            fallback_ext = orig_name.rsplit(".", 1)[-1].lower() if "." in orig_name else "png"
            converted = _bytes_of(img_obj, fallback_ext=fallback_ext)
            if converted is None:
                continue
            img_bytes, ext = converted

            img_hash = hashlib.md5(img_bytes).hexdigest()

            # 已存過（含 fast 路徑也算過）→ 直接 reuse，不重寫檔
            if img_hash in ctx.saved_image_hashes:
                safe_name = ctx.hash_to_filename[img_hash]
            else:
                safe_name = f"{ctx.stem}_h{img_hash[:8]}.{ext}"
                target = ctx.img_dir / safe_name
                try:
                    target.write_bytes(img_bytes)
                except Exception:
                    continue
                ctx.saved_image_hashes.add(img_hash)
                ctx.hash_to_filename[img_hash] = safe_name

            # 改寫 marker_text 內 ![..](orig_name) → ![safe](folder/safe)
            new_ref = f"![{safe_name}]({ctx.folder_name}/{safe_name})"
            new_text = re.sub(
                r"!\[[^\]]*\]\(" + re.escape(orig_name) + r"\)",
                lambda _m, ref=new_ref: ref,
                new_text,
            )
        return new_text
