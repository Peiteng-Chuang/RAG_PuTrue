"""Dify External Knowledge API 端點（方案 C，見 solution.txt 2026-06-23）。

讓 Dify 當「對話前端 + workflow 編排」，**完全不碰本專案 Qdrant**：Dify 只 POST /retrieval，
我方用既有 bge-m3 hybrid 檢索回結果。一個 payload 欄位都不用拔，前處理／來源追溯／hybrid 全保留。

**範圍**：純檢索（上游），獨立於 md_review_ui（下游標記工具）。本檔自帶 bge-m3 + Qdrant 檢索，
**不 import md_review_ui / streamlit**，以維持工具範圍隔離（見 feedback_tool_scope_isolation、solution.txt §7）。

依賴（venv312 已有）：starlette + uvicorn + qdrant-client + FlagEmbedding + numpy。

啟動：
    set QDRANT_URL=http://localhost:6333
    set QDRANT_COLLECTION=putrue_rag_text_v1
    set DIFY_API_KEY=<你在 Dify 填的同一把 Bearer 金鑰>
    set EMBED_DEVICE=cpu          # 2050 互動端安全；4090 可 cuda
    python dify_retrieval_api.py   # 預設 0.0.0.0:8100

Dify 設定：外部知識庫 API → API Endpoint 填本服務基底 URL（Dify 會自動補 /retrieval）+ 同一把 API Key。
⚠️ hybrid 用 RRF 融合，分數是「排名倒數和」（很小，非 0~1 相似度）。**請在 Dify 端關閉 Score Threshold**
   （或設 0），否則預設 0.5 會把結果全濾掉。本服務預設忽略 score_threshold（見 DIFY_APPLY_SCORE_THRESHOLD）。
"""
from __future__ import annotations

import contextlib
import os
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dify_retrieval")

# ---- 設定（環境變數，全有預設）----
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "putrue_rag_text_v1")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "")           # 空 = 停用驗證（僅開發用）
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
DEFAULT_TOP_K = int(os.environ.get("DIFY_DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.environ.get("DIFY_MAX_TOP_K", "20"))
# 預設忽略 Dify 的 score_threshold（RRF 分數非 0~1，套了會誤殺）；設 1 才啟用
APPLY_SCORE_THRESHOLD = os.environ.get("DIFY_APPLY_SCORE_THRESHOLD", "0") == "1"

# ---- 行程內單例（startup 載入）----
_model = None
_client: QdrantClient | None = None


def _load_model():
    from FlagEmbedding import BGEM3FlagModel
    if EMBED_DEVICE == "cpu":
        return BGEM3FlagModel(EMBED_MODEL, use_fp16=False, devices="cpu")
    return BGEM3FlagModel(EMBED_MODEL, use_fp16=True)


def _encode_query(model, text: str) -> tuple[np.ndarray, dict]:
    """bge-m3 對單條 query 取 dense + sparse（symmetric encoder，query/passage 同前綴）。"""
    out = model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)[0]
    sparse = {str(k): float(v) for k, v in out["lexical_weights"][0].items()}
    return dense, sparse


def _sparse_vec(sparse: dict) -> qm.SparseVector:
    return qm.SparseVector(
        indices=[int(k) for k in sparse.keys()],
        values=[float(v) for v in sparse.values()],
    )


def _search_hybrid(client, coll, dense, sparse, top_k, prefetch_k=40):
    """RRF 融合：dense + sparse 各撈 prefetch_k，伺服器端 Reciprocal Rank Fusion。
    與 md_review_ui.search_hybrid 同一套（named vectors text_dense/text_sparse）。"""
    res = client.query_points(
        collection_name=coll,
        prefetch=[
            qm.Prefetch(query=dense.tolist(), using="text_dense", limit=prefetch_k),
            qm.Prefetch(query=_sparse_vec(sparse), using="text_sparse", limit=prefetch_k),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return res.points


def hit_to_record(point: Any) -> dict:
    """把一個 Qdrant 檢索命中 → Dify record。**純函式**（不碰全域），方便單測。

    對映（見 solution.txt §6）：
      content  ← content.md_content / text_with_prefix / text（擇一非空）
      score    ← Qdrant hybrid 分數
      title    ← source.doc_title / file_name / file_key
      metadata ← source + location 整包（原封保留做來源追溯；**必為 object、非 null**）
    """
    payload = getattr(point, "payload", None) or {}
    meta = payload.get("metadata", {}) or {}
    source = meta.get("source", {}) or {}
    location = meta.get("location", {}) or {}
    content_obj = payload.get("content", {}) or {}

    content = (
        content_obj.get("md_content")
        or content_obj.get("text_with_prefix")
        or content_obj.get("text")
        or ""
    )
    title = (
        source.get("doc_title") or source.get("file_name")
        or source.get("file_key") or "未知來源"
    )
    # metadata：保留來源追溯關鍵欄（None 也保留成 key，但整體恆為 dict）
    md = {
        "file_key": source.get("file_key"),
        "file_name": source.get("file_name"),
        "project_name": source.get("project_name"),
        "doc_type": source.get("doc_type"),
        "file_path": source.get("file_path"),
        "file_hash": source.get("file_hash"),
        "page": location.get("page"),
        "breadcrumb": location.get("breadcrumb") or location.get("headings_flat") or [],
    }
    return {
        "content": content,
        "score": float(getattr(point, "score", 0.0) or 0.0),
        "title": title,
        "metadata": md,
    }


def _err(status: int, code: int, msg: str) -> JSONResponse:
    """Dify External Knowledge API 錯誤格式。"""
    return JSONResponse({"error_code": code, "error_msg": msg}, status_code=status)


def _check_auth(request: Request) -> JSONResponse | None:
    """Bearer 驗證。回 None = 通過；回 JSONResponse = 擋下。DIFY_API_KEY 空 → 停用（開發）。"""
    if not DIFY_API_KEY:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return _err(403, 1001,
                    "Invalid Authorization header format. Expected 'Bearer <api-key>' format.")
    if auth[len("Bearer "):].strip() != DIFY_API_KEY:
        return _err(403, 1002, "Authorization failed")
    return None


async def retrieval(request: Request) -> JSONResponse:
    denied = _check_auth(request)
    if denied is not None:
        return denied
    try:
        body = await request.json()
    except Exception:
        return _err(400, 400, "Request body is not valid JSON")

    query = (body.get("query") or "").strip()
    if not query:
        return _err(400, 400, "Missing 'query'")
    setting = body.get("retrieval_setting") or {}
    top_k = min(int(setting.get("top_k") or DEFAULT_TOP_K), MAX_TOP_K)
    threshold = setting.get("score_threshold")
    knowledge_id = body.get("knowledge_id", "")

    if _model is None or _client is None:
        return _err(500, 500, "Service not ready (model/client not loaded)")

    try:
        dense, sparse = _encode_query(_model, query)
        points = _search_hybrid(_client, QDRANT_COLLECTION, dense, sparse, top_k)
    except Exception as e:  # noqa: BLE001 — 回可讀錯誤給 Dify，不吞
        log.exception("retrieval failed")
        return _err(500, 500, f"Retrieval failed: {type(e).__name__}: {e}")

    records = [hit_to_record(p) for p in points]
    if APPLY_SCORE_THRESHOLD and threshold is not None:
        records = [r for r in records if r["score"] >= float(threshold)]
    log.info("kb=%s query=%r top_k=%d → %d records", knowledge_id, query[:50], top_k, len(records))
    return JSONResponse({"records": records})


async def health(_request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok" if (_model is not None and _client is not None) else "loading",
        "collection": QDRANT_COLLECTION,
        "auth": "on" if DIFY_API_KEY else "off",
    })


async def _startup() -> None:
    global _model, _client
    log.info("connecting Qdrant %s …", QDRANT_URL)
    _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=30)
    if not _client.collection_exists(QDRANT_COLLECTION):
        log.warning("collection %r 不存在（服務仍啟動，檢索會報錯）", QDRANT_COLLECTION)
    log.info("loading %s on %s …（首次約數秒）", EMBED_MODEL, EMBED_DEVICE)
    _model = _load_model()
    if not DIFY_API_KEY:
        log.warning("DIFY_API_KEY 未設 → 驗證停用（僅限開發，勿對外開放）")
    log.info("ready. POST /retrieval")


@contextlib.asynccontextmanager
async def _lifespan(_app):
    await _startup()
    yield


app = Starlette(
    debug=False,
    routes=[Route("/retrieval", retrieval, methods=["POST"]), Route("/health", health)],
    lifespan=_lifespan,
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("DIFY_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("DIFY_API_PORT", "8100")),
    )
