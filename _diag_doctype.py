"""唯讀診斷：列出 Qdrant 各 doc_type 的 points 數、每個 doc_type 底下的建案與檔數。
專查「20.個案結案報告資料夾 是否還在、file_key/doc_type 變成什麼」。不寫任何資料。

用法：
    python _diag_doctype.py                       # 用 .env 的 QDRANT_URL
    python _diag_doctype.py --collection putrue_rag_text_v1
"""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
ap.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY", ""))
ap.add_argument("--collection", default="putrue_rag_text_v1")
ap.add_argument("--focus", default="20.個案結案報告資料夾", help="特別細看的 doc_type")
args = ap.parse_args()

from qdrant_client import QdrantClient
cli = QdrantClient(url=args.qdrant_url, api_key=args.api_key or None, timeout=60)
if not cli.collection_exists(args.collection):
    print(f"❌ collection 不存在：{args.collection}"); sys.exit(1)

# doc_type facet
print(f"=== collection: {args.collection} ===")
resp = cli.facet(collection_name=args.collection, key="metadata.source.doc_type", limit=1000)
print("\n[doc_type 分布]（points 數）")
for h in sorted(resp.hits, key=lambda x: str(x.value)):
    print(f"  {h.count:>7}  {h.value!r}")

# 全量掃一次，依 doc_type 統計建案/檔，並收集 focus 的明細
by_dt_proj = defaultdict(lambda: defaultdict(set))   # doc_type -> project -> set(file_key)
focus_keys = defaultdict(int)                          # file_key -> points（focus 用）
focus_hash_to_fk = defaultdict(set)                    # file_hash -> {file_key}
total = 0; offset = None
while True:
    pts, offset = cli.scroll(args.collection, limit=512, offset=offset,
                             with_payload=True, with_vectors=False)
    for pt in pts:
        total += 1
        src = ((pt.payload or {}).get("metadata") or {}).get("source") or {}
        dt = src.get("doc_type") or "(無)"
        pj = src.get("project_name") or "(無)"
        fk = src.get("file_key") or "(無)"
        by_dt_proj[dt][pj].add(fk)
        if dt == args.focus:
            focus_keys[fk] += 1
            h = (src.get("file_hash") or "").strip()
            if h:
                focus_hash_to_fk[h].add(fk)
    if offset is None:
        break

print(f"\n總 points：{total}")
print(f"\n[各 doc_type 的 建案數 / 檔數]")
for dt in sorted(by_dt_proj):
    nproj = len(by_dt_proj[dt])
    nfile = sum(len(s) for s in by_dt_proj[dt].values())
    print(f"  {dt!r}: {nproj} 建案 / {nfile} 檔")

print(f"\n=== 細看 focus = {args.focus!r} ===")
if not focus_keys:
    print("  ⚠️ 此 doc_type 底下 0 個 points → 可能被改名/重 upsert 覆蓋掉，或從未入庫。")
else:
    print(f"  {len(focus_keys)} 檔 / {sum(focus_keys.values())} points。前 30 檔：")
    for fk in sorted(focus_keys)[:30]:
        print(f"     {focus_keys[fk]:>4} pts  {fk}")
    cross = {h: ks for h, ks in focus_hash_to_fk.items() if len(ks) > 1}
    if cross:
        print(f"\n  ⚠️ 同 file_hash 對到多個 file_key {len(cross)} 組（內容撞號跡象）：")
        for h, ks in list(cross.items())[:10]:
            print(f"     hash {h[:10]}… → {sorted(ks)}")
