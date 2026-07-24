"""LLM_fill_sheet.py — 透過詢問 RAG 自動填/更新「建案屬性表」（upstream 工具）。

流程：對每個 (建案 × 屬性) 空格 → bge-m3 hybrid 檢索該建案的文件片段（Qdrant，
project_name 過濾）→ 交給 LLM 嚴格抽值 → 找到才填、找不到留空（絕不臆測）。
每格的抽取值/來源/片段另寫一份審查報告，供人工複核。

**範圍**：純上游（檢索 + LLM 抽取 + 寫 CSV），**不 import md_review_ui / streamlit**
（守工具範圍隔離）。檢索與 dify_retrieval_api 同一套 bge-m3 hybrid（named vectors
text_dense/text_sparse、RRF），額外加 project_name 過濾。

**LLM 自動篩值**：每格不只抽值，還讓 LLM 自評信心（high/medium/low）+ 附佐證原文。
`--min-confidence`（預設 medium）當閘門：達門檻才自動採用；低於門檻**不填、只記報告待複核**。
所有嘗試（採用/未採用/NOT_FOUND）都寫進 `*.fill_report.csv` 供人工複核。

**資料安全（預設）**：
- 預設 **dry-run**（只印會填什麼、不動檔）；加 `--write` 才真的寫，且**先備份原檔**。
- 預設**只補空格**；加 `--overwrite` 才會覆蓋已填的格子。
- LLM 回 NOT_FOUND / 空 / 低信心 → **留空**（不塞假值、不 fall through）。

依賴（venv312 已有）：qdrant-client + FlagEmbedding + numpy + openai。

用法：
    # 先看會填什麼（不動檔）
    python LLM_fill_sheet.py
    # 只跑特定建案、每格看幾筆片段
    python LLM_fill_sheet.py --projects 文林中正,洲美河堤 --top-k 6
    # 確認後真的寫入（會先備份 + 出審查報告）
    python LLM_fill_sheet.py --write

環境變數（全有預設）：
    QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION
    EMBED_MODEL / EMBED_DEVICE(cpu)
    VLLM_BASE_URL / LLM_MODEL_NAME / VLLM_API_KEY（與 md_review_ui/.env 統一命名）
    ATTR_TABLE_PATH（預設 structured_data/建案屬性表.csv）
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import structured_table as stbl  # 只借用 _decode / _RESERVED_COLS，不進 ETL

# 讀 .env（與 md_review_ui / dify_retrieval_api 一致）→ 金鑰/URL/模型寫 .env 即可，不必 shell set
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---- 設定（環境變數，全有預設；對齊 dify_retrieval_api / md_review_ui）----
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "putrue_rag_text_v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.201.66:8000/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "") or "EMPTY"  # 統一名稱(同 md_review_ui/.env)；vLLM 通常不驗但 client 需非空
DEFAULT_TABLE = str(Path(__file__).resolve().parent / "structured_data" / "建案屬性表.csv")

# Qdrant payload 內 project_name 的路徑（見 dify_retrieval_api.hit_to_record）
PROJECT_KEY_PATH = "metadata.source.project_name"

SYSTEM_PROMPT = (
    "你是建築工程文件的資料抽取助手。只能根據提供的文件片段回答，"
    "嚴禁臆測、嚴禁用常識或訓練知識補充。找不到就明講 NOT_FOUND。"
)


# ---------------- 檢索（bge-m3 hybrid + project 過濾）----------------
def load_model():
    from FlagEmbedding import BGEM3FlagModel
    if EMBED_DEVICE == "cpu":
        return BGEM3FlagModel(EMBED_MODEL, use_fp16=False, devices="cpu")
    return BGEM3FlagModel(EMBED_MODEL, use_fp16=True)


def encode_query(model, text: str):
    out = model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)[0]
    sparse = {str(k): float(v) for k, v in out["lexical_weights"][0].items()}
    return dense, sparse


def search(client, qm, dense, sparse, project: str, top_k: int, prefetch_k: int = 40):
    """RRF 融合 + project_name 過濾。回 points。"""
    flt = qm.Filter(must=[qm.FieldCondition(
        key=PROJECT_KEY_PATH, match=qm.MatchValue(value=project))])
    sparse_vec = qm.SparseVector(
        indices=[int(k) for k in sparse.keys()],
        values=[float(v) for v in sparse.values()],
    )
    res = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            qm.Prefetch(query=dense.tolist(), using="text_dense", limit=prefetch_k, filter=flt),
            qm.Prefetch(query=sparse_vec, using="text_sparse", limit=prefetch_k, filter=flt),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        query_filter=flt,
    )
    return res.points


def resolve_project_names(client, qm, csv_keys: list[str]) -> tuple[dict, list]:
    """把 CSV 純建案名對映到 Qdrant 實際的 project_name（常帶建案代號前綴，如 'B65文林中正'）。
    Qdrant 過濾用完全相等，故必須用「全名」才 match。回 (mapping: csv_key→qdrant_name|None, 全名清單)。"""
    qnames: list[str] = []
    try:
        r = client.facet(collection_name=QDRANT_COLLECTION, key=PROJECT_KEY_PATH, limit=1000)
        qnames = [h.value for h in r.hits if isinstance(h.value, str) and h.value]
    except Exception:  # noqa: BLE001 — facet 不可用（欄位未建 keyword 索引）→ scroll 兜底
        seen, offset = set(), None
        for _ in range(30):
            pts, offset = client.scroll(QDRANT_COLLECTION, limit=1000, with_payload=True, offset=offset)
            for p in pts:
                v = ((p.payload or {}).get("metadata", {}) or {}).get("source", {}).get("project_name")
                if isinstance(v, str) and v:
                    seen.add(v)
            if offset is None:
                break
        qnames = sorted(seen)
    mapping: dict = {}
    for k in csv_keys:
        kk = k.strip()
        if not kk:
            mapping[k] = None
        elif kk in qnames:                      # 完全相同
            mapping[k] = kk
        else:
            cands = [n for n in qnames if kk in n]     # Qdrant 全名「包含」CSV 純名
            if len(cands) == 1:
                mapping[k] = cands[0]
            elif len(cands) > 1:                       # 多個候選 → 取唯一 endswith，否則視為不明確
                ends = [n for n in cands if n.endswith(kk)]
                mapping[k] = ends[0] if len(ends) == 1 else None
            else:
                mapping[k] = None
    return mapping, qnames


def point_text_and_src(point) -> tuple[str, str, str, float]:
    """回 (content, 來源檔, 來源頁, score)。對映同 dify_retrieval_api.hit_to_record。"""
    payload = getattr(point, "payload", None) or {}
    meta = payload.get("metadata", {}) or {}
    source = meta.get("source", {}) or {}
    location = meta.get("location", {}) or {}
    content_obj = payload.get("content", {}) or {}
    content = (content_obj.get("md_content") or content_obj.get("text_with_prefix")
               or content_obj.get("text") or "")
    src_file = source.get("file_name") or source.get("file_key") or ""
    src_page = str(location.get("page") or "")
    return content, src_file, src_page, float(getattr(point, "score", 0.0) or 0.0)


# ---------------- LLM 抽值 ----------------
def attr_to_query(project: str, attr: str) -> str:
    """把屬性欄名轉成自然查詢；去掉 _單位 後綴（基地面積_m2 → 基地面積）。"""
    base = attr.split("_")[0] if "_" in attr else attr
    return f"{project} 的 {base}"


CONF_RANK = {"low": 1, "medium": 2, "high": 3}


def _parse_json_obj(text: str) -> dict | None:
    """從 LLM 回覆抽第一個 JSON 物件（容忍 ```json 圍欄與前後雜訊）。"""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:  # noqa: BLE001
        return None


def extract_judged(llm, model_name: str, project: str, attr: str, chunks: list[str]) -> dict:
    """LLM 抽值 + 自評信心 + 佐證原文。
    回 {value, confidence(high/medium/low/none), evidence}；找不到 value="" confidence="none"。"""
    joined = "\n\n---\n\n".join(chunks)
    user = (
        f"建案：{project}\n"
        f"要抽取的屬性：{attr}\n\n"
        f"文件片段：\n---\n{joined}\n---\n\n"
        f"請從片段中抽出「{attr}」這個屬性的值，並自評可信度。只輸出一個 JSON 物件：\n"
        f'{{"value":"值本身(簡短，含單位照原文)","confidence":"high|medium|low","evidence":"直接支持該值的原文片段(照抄)"}}\n'
        f"規則：\n"
        f"- 只能依片段，嚴禁臆測；片段沒明確講到 → value 用空字串、confidence 用 \"none\"。\n"
        f"- confidence：片段明確直述且與此建案直接相關=high；需輕度推斷或表述模糊=medium；"
        f"僅間接暗示/可能張冠李戴=low。\n"
        f"- evidence 必須是片段裡的原文，不可自己造句。\n"
        f"只輸出 JSON："
    )
    resp = llm.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=400,
    )
    raw = (resp.choices[0].message.content or "").strip()
    obj = _parse_json_obj(raw)
    if not obj:
        # 解析失敗 → 不採用（安全），保留原文供複核
        return {"value": "", "confidence": "none", "evidence": f"[JSON解析失敗] {raw[:120]}"}
    val = str(obj.get("value", "") or "").strip().strip('「」"\'` ')
    conf = str(obj.get("confidence", "none") or "none").strip().lower()
    if conf not in CONF_RANK:
        conf = "none"
    if not val or "NOT_FOUND" in val.upper():
        return {"value": "", "confidence": "none", "evidence": str(obj.get("evidence", ""))[:200]}
    return {"value": val[:300], "confidence": conf, "evidence": str(obj.get("evidence", ""))[:200]}


# ---------------- 表格讀寫（矩陣格式）----------------
def read_matrix(path: str) -> tuple[list[str], list[list[str]], str]:
    raw = Path(path).read_bytes()
    text, enc = stbl._decode(raw)
    delim = stbl._sniff_delimiter(text)   # tab / 逗號自動偵測（Excel 繁中常存 tab）
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        raise SystemExit(f"空檔或無表頭：{path}")
    header = [h.strip() for h in rows[0]]
    body = [r + [""] * (len(header) - len(r)) for r in rows[1:] if any(c.strip() for c in r)]
    return header, body, enc


def write_matrix(path: str, header: list[str], body: list[list[str]]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in body:
        w.writerow(r)
    Path(path).write_text(buf.getvalue(), encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser(description="透過 RAG 自動填/更新建案屬性表")
    ap.add_argument("--table", default=os.environ.get("ATTR_TABLE_PATH", DEFAULT_TABLE))
    ap.add_argument("--write", action="store_true", help="真的寫入（預設 dry-run 不動檔）")
    ap.add_argument("--overwrite", action="store_true", help="連已填的格子也重抽（預設只補空格）")
    ap.add_argument("--projects", default="", help="只跑這些建案（逗號分隔；預設全部）")
    ap.add_argument("--attributes", default="", help="只跑這些屬性欄（逗號分隔；預設全部）")
    ap.add_argument("--top-k", type=int, default=5, help="每格檢索片段數")
    ap.add_argument("--min-confidence", choices=["high", "medium", "low"], default="medium",
                    help="LLM 自評信心達此門檻才自動採用；低於門檻→留空、只記報告待複核（預設 medium）")
    ap.add_argument("--limit", type=int, default=0, help="最多處理幾格（測試用；0=不限）")
    args = ap.parse_args()
    min_rank = CONF_RANK[args.min_confidence]

    header, body, enc = read_matrix(args.table)
    key_col = "建案key" if "建案key" in header else header[0]
    key_idx = header.index(key_col)
    reserved = stbl._RESERVED_COLS
    attr_cols = [(i, h) for i, h in enumerate(header)
                 if h and h != key_col and h not in reserved]

    only_projects = {p.strip() for p in args.projects.split(",") if p.strip()}
    only_attrs = {a.strip() for a in args.attributes.split(",") if a.strip()}

    print(f"[表] {args.table}  編碼={enc}  建案={len(body)}  屬性欄={len(attr_cols)}")
    print(f"[模式] {'WRITE' if args.write else 'DRY-RUN（不動檔）'} | "
          f"{'覆蓋已填' if args.overwrite else '只補空格'} | top_k={args.top_k} | "
          f"信心門檻≥{args.min_confidence}")
    print(f"[檢索] Qdrant {QDRANT_URL} / {QDRANT_COLLECTION} | embed {EMBED_MODEL}({EMBED_DEVICE})")
    print(f"[LLM] {VLLM_BASE_URL} model={LLM_MODEL_NAME or '(未設 LLM_MODEL_NAME)'}")

    # 連線 / 載入
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm
    from openai import OpenAI
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    model_name = LLM_MODEL_NAME
    llm = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    if not model_name:  # 未指定就抓 vLLM 第一個模型
        try:
            model_name = llm.models.list().data[0].id
            print(f"[LLM] 自動選用模型：{model_name}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ 未設 LLM_MODEL_NAME 且抓 /v1/models 失敗：{e}")
            return 2
    print("[embed] 載入 bge-m3 …（CPU 約 15-60s）")
    embed = load_model()

    # 建案名對映：CSV 純名 → Qdrant 實際 project_name（常帶代號前綴，如 B65文林中正）。
    # Qdrant 過濾用完全相等，對不上就 0 筆 → 全格「無相關片段」。
    csv_projects = [row[key_idx].strip() for row in body
                    if row[key_idx].strip() and (not only_projects or row[key_idx].strip() in only_projects)]
    proj_map, qnames = resolve_project_names(client, qm, csv_projects)
    print("[建案對映] CSV 建案key → Qdrant project_name：")
    for k in csv_projects:
        print(f"    {k!r} → {proj_map.get(k)!r}")
    _unresolved = [k for k in csv_projects if not proj_map.get(k)]
    if _unresolved:
        print(f"    ⚠️ 對不到 Qdrant 的建案（將整案跳過、不查）：{_unresolved}")
        print(f"       Qdrant 現有 project_name（前 15）：{qnames[:15]}")

    audit: list[dict] = []
    filled = skipped = notfound = low_conf = 0
    processed = 0

    for row in body:
        project = row[key_idx].strip()
        if not project or (only_projects and project not in only_projects):
            continue
        qproject = proj_map.get(project)   # Qdrant 過濾用的全名（帶前綴）
        if not qproject:
            continue                        # 對不到 Qdrant 建案 → 整案跳過（上面已警告）
        for ci, attr in attr_cols:
            if only_attrs and attr not in only_attrs:
                continue
            if row[ci].strip() and not args.overwrite:
                skipped += 1
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1

            q = attr_to_query(project, attr)
            try:
                dense, sparse = encode_query(embed, q)
                pts = search(client, qm, dense, sparse, qproject, args.top_k)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ [{project} / {attr}] 檢索失敗：{type(e).__name__}: {e}")
                continue
            if not pts:
                notfound += 1
                print(f"  · [{project} / {attr}] 無相關片段 → 留空")
                continue
            triples = [point_text_and_src(p) for p in pts]
            chunks = [t[0] for t in triples if t[0]]
            try:
                j = extract_judged(llm, model_name, project, attr, chunks)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ [{project} / {attr}] LLM 失敗：{type(e).__name__}: {e}")
                continue

            top = triples[0]
            val, conf, ev = j["value"], j["confidence"], j["evidence"]
            rec = {"建案": project, "屬性": attr, "抽取值": val, "信心": conf, "狀態": "",
                   "來源檔": top[1], "來源頁": top[2], "score": f"{top[3]:.4f}", "佐證": ev}
            if not val:
                notfound += 1
                rec["狀態"] = "留空(NOT_FOUND)"
                print(f"  · [{project} / {attr}] NOT_FOUND → 留空")
            elif CONF_RANK.get(conf, 0) >= min_rank:
                filled += 1
                row[ci] = val
                rec["狀態"] = "採用"
                print(f"  ✅ [{project} / {attr}] = {val}  [{conf}]  (來源 {top[1]} p.{top[2]})")
            else:
                low_conf += 1
                rec["狀態"] = f"未採用(信心 {conf} < {args.min_confidence})"
                print(f"  ⚠️ [{project} / {attr}] = {val}  [{conf}] < 門檻 → 留空、記報告待複核")
            audit.append(rec)
        if args.limit and processed >= args.limit:
            break

    print(f"\n[結果] 採用 {filled} | 未採用(低信心) {low_conf} | 找不到/留空 {notfound} | 跳過(已填) {skipped}")

    if not args.write:
        print("\nDRY-RUN：未寫入任何檔。確認上面結果後，加 --write 實際寫入"
              "（低於信心門檻的候選也會列進報告供複核）。")
        return 0

    tp = Path(args.table)
    # 審查報告（採用/未採用/NOT_FOUND 全紀錄，含信心與佐證）—— 即使沒採用任何值也寫，方便複核
    if audit:
        rep = tp.with_name(tp.stem + ".fill_report.csv")
        with open(rep, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["建案", "屬性", "抽取值", "信心", "狀態",
                                              "來源檔", "來源頁", "score", "佐證"])
            w.writeheader()
            w.writerows(audit)
        print(f"[報告] 每格抽取/信心/狀態/佐證 → {rep}（請人工複核，尤其『未採用』與數值/工法）")
    if filled == 0:
        print("沒有達信心門檻的新值，未改動屬性表（低信心候選見上方報告）。")
        return 0

    bak = tp.with_name(tp.stem + ".before_fill.bak.csv")
    if not bak.exists():
        shutil.copy2(tp, bak)
        print(f"[備份] 原檔 → {bak}")
    write_matrix(args.table, header, body)
    print(f"[寫入] {args.table}（已採用 {filled} 格）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
