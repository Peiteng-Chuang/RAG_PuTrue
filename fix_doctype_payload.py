"""fix_doctype_payload — 批次修正 Qdrant 既有 points 的 doc_type / project_name / file_key。

問題：若當初 ingest 時 file_key 首段是「建案」（data_root 設在 根/類別 之下），doc_type 會
fallback 成建案名 → chat「文件種類」下拉冒出一堆建案。本工具把它們收斂成「類別/建案」結構。

做法（不動向量、point_id 不變、只改 payload 的 metadata.source）：
  1. 掃 DATA_ROOT 下的實體檔 → 學到每檔的新 file_key = `類別/建案/檔`（相對 DATA_ROOT）。
  2. 掃 Qdrant，依「舊 file_key」把 points 分組（每檔一組）。
  3. 用「舊 file_key 的尾段 == 新 file_key 去掉類別」配對（精準），配不到再退而用 stem。
  4. 重算 doc_type = 新file_key[0]（套 doc_type_map 覆寫）、project_name = 新file_key[1]，
     用 set_payload(key="metadata.source") 整檔批次更新。

安全（[[feedback-data-dependency-audit-first]]）：預設 dry-run，列出「doc_type 從什麼→什麼、
幾檔幾 points」；確認後加 --apply 才寫。set_payload 只改 metadata.source，不碰向量/其他欄位。

用法：
    python fix_doctype_payload.py --data-root "D:/璞真RAG資料夾"            # dry-run
    python fix_doctype_payload.py --data-root "D:/璞真RAG資料夾" --apply    # 實寫
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATA_ROOT_ENV = (os.getenv("DATA_ROOT", "") or "").strip()
DEFAULT_QDRANT = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = "putrue_rag_text_v1"
MKDATA = Path("./mkdata")
EXTS = ("*.pdf", "*.ppt", "*.pptx")


def norm(s: str | None) -> str:
    return (s or "").replace("\\", "/")


def parts_of(fk: str | None) -> list[str]:
    return [p for p in norm(fk).split("/") if p]


def derive_project_name(fk: str) -> str:
    """建案 = file_key[1]（類別/建案/檔）；無建案層退回首段；只有檔名退回未分類。"""
    p = parts_of(fk)
    if len(p) >= 3:
        return p[1]
    if len(p) == 2:
        return p[0]
    return "未分類"


def load_doc_type_map() -> dict:
    f = MKDATA / "doc_type_map.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def resolve_doc_type(fk: str, dtm: dict) -> str:
    """doc_type = doc_type_map[首段] or 首段（類別資料夾名）or 未分類。對齊 md_review_ui。"""
    p = parts_of(fk)
    folder = p[0] if len(p) > 1 else ""
    return str(dtm.get(folder) or folder or "未分類")


def scan_disk(data_root: Path):
    """掃實體檔 → (by_tail, by_stem, collisions_tail)。

    by_tail[建案/檔] = 新file_key（首段=類別）；by_stem[stem] = [新file_key, ...]。
    """
    by_tail: dict[str, str] = {}
    by_stem: dict[str, list[str]] = defaultdict(list)
    collisions_tail: set[str] = set()
    for ext in EXTS:
        for p in data_root.rglob(ext):
            if not p.is_file():
                continue
            new_fk = norm(str(p.relative_to(data_root)))
            seg = parts_of(new_fk)
            tail = "/".join(seg[1:]) if len(seg) >= 2 else new_fk
            if tail in by_tail and by_tail[tail] != new_fk:
                collisions_tail.add(tail)
            by_tail[tail] = new_fk
            by_stem[p.stem].append(new_fk)
    return by_tail, by_stem, collisions_tail


def main() -> int:
    ap = argparse.ArgumentParser(description="批次修正 Qdrant 既有 points 的 doc_type/project_name/file_key")
    ap.add_argument("--data-root", default=DATA_ROOT_ENV, help="原始資料根目錄（類別資料夾的父層）")
    ap.add_argument("--qdrant-url", default=DEFAULT_QDRANT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY", ""))
    ap.add_argument("--apply", action="store_true", help="實際寫入（預設 dry-run）")
    args = ap.parse_args()

    if not args.data_root:
        print("❌ 需要 --data-root（或在 .env 設 DATA_ROOT）")
        return 1
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ data_root 不存在/不可達：{data_root}")
        return 1

    from qdrant_client import QdrantClient
    cli = QdrantClient(url=args.qdrant_url, api_key=args.api_key or None, timeout=60)
    if not cli.collection_exists(args.collection):
        print(f"❌ collection 不存在：{args.collection}")
        return 1

    dtm = load_doc_type_map()
    by_tail, by_stem, coll_tail = scan_disk(data_root)
    print(f"=== 掃描 {data_root} ===")
    print(f"實體檔：{sum(len(v) for v in by_stem.values())}；唯一 tail（建案/檔）：{len(by_tail)}")
    if coll_tail:
        print(f"⚠️ tail 衝突 {len(coll_tail)} 組（同『建案/檔』出現在多個類別）→ 退用 stem 或跳過")

    # 掃 Qdrant，依舊 file_key 分組（每檔一組 → set_payload 整檔批次改 source）
    groups: dict[str, list] = defaultdict(list)
    sources: dict[str, dict] = {}
    total = 0
    offset = None
    while True:
        pts, offset = cli.scroll(
            args.collection, limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for pt in pts:
            total += 1
            src = ((pt.payload or {}).get("metadata") or {}).get("source") or {}
            old_fk = norm(src.get("file_key") or "")
            key = old_fk or f"__nokey__{pt.id}"
            groups[key].append(pt.id)
            sources.setdefault(key, src)
        if offset is None:
            break

    print(f"=== Qdrant {args.collection} ===")
    print(f"points：{total}；檔（依舊 file_key 分組）：{len(groups)}\n")

    def match_new_fk(old_fk: str, src: dict):
        ofk = norm(old_fk)
        if ofk in by_tail:
            return by_tail[ofk], "tail"
        stem = Path(norm(src.get("file_name") or ofk)).stem
        cands = by_stem.get(stem, [])
        if len(cands) == 1:
            return cands[0], "stem"
        return None, ("ambiguous" if len(cands) > 1 else "orphan")

    transitions: dict[tuple[str, str], int] = defaultdict(int)
    updates: list[tuple[str, str, dict, list]] = []  # (old_fk, new_fk, new_src, ids)
    n_orphan = n_ambig = n_same = 0
    for key, ids in groups.items():
        src = dict(sources[key])
        new_fk, how = match_new_fk(key, src)
        if new_fk is None:
            if how == "ambiguous":
                n_ambig += 1
            else:
                n_orphan += 1
            continue
        new_dt = resolve_doc_type(new_fk, dtm)
        new_pj = derive_project_name(new_fk)
        if (src.get("doc_type") == new_dt and src.get("project_name") == new_pj
                and norm(src.get("file_key")) == new_fk):
            n_same += 1
            continue
        transitions[(src.get("doc_type", "(無)"), new_dt)] += 1
        new_src = dict(src)
        new_src.update(doc_type=new_dt, project_name=new_pj, file_key=new_fk, file_path=new_fk)
        updates.append((key, new_fk, new_src, ids))

    print("=== doc_type 轉換（檔數）===")
    for (o, n), c in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"  {o}  →  {n}   ({c} 檔)")
    print("\n=== 範例（前 10 檔 old→new file_key）===")
    for old_fk, new_fk, _, ids in updates[:10]:
        print(f"  {old_fk}  →  {new_fk}   ({len(ids)} points)")
    print(f"\n待更新：{len(updates)} 檔 / {sum(len(i) for *_, i in updates)} points")
    print(f"已正確跳過：{n_same}；orphan（disk 找不到對應）：{n_orphan}；stem 模糊跳過：{n_ambig}")

    if not args.apply:
        print("\n📋 dry-run：未寫入任何資料。確認無誤後加 --apply。")
        return 0

    done = 0
    fails = 0
    for old_fk, new_fk, new_src, ids in updates:
        try:
            cli.set_payload(args.collection, payload=new_src, points=ids, key="metadata.source")
            done += 1
        except Exception as e:
            fails += 1
            print(f"⚠️ set_payload 失敗 {new_fk}: {e}")
    print(f"\n✅ 已更新 {done}/{len(updates)} 檔的 metadata.source（doc_type/project_name/file_key）。"
          + (f" 失敗 {fails}。" if fails else ""))
    print("👉 chat「文件種類」facet 有 5 分鐘 cache；重啟 app 或稍候即刷新。向量與 point_id 未動。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
