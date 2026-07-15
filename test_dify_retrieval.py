"""dify_retrieval_api 端點的專門測試程式。

兩種用法：
  1) 查詢探針（預設）——模擬 Dify 送一條 query，漂亮印出回傳的 records（rank/score/title/
     來源/正文預覽），順便就地檢查每筆是否符合 Dify 合約。日常用來看「這條問題檢索品質好不好」。
         python test_dify_retrieval.py "各案公設比" --top-k 5
  2) 合約測試套件（--suite）——對執行中的端點跑一組斷言：/health、缺 auth→1001、
     錯 key→1002、正常查詢→200 且 records 形狀正確、缺 query→400。全過 exit 0，否則 exit 1（可進 CI）。
         python test_dify_retrieval.py --suite

設定：
  --url   端點基底 URL（預設 http://localhost:8100）；本程式會自己補 /retrieval。
  --key   Bearer 金鑰（預設讀環境變數 DIFY_API_KEY）。端點若停用驗證（auth:off）自動跳過 auth 檢查。

依賴：httpx（venv312 已有）。
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import httpx

sys.stdout.reconfigure(encoding="utf-8")  # Windows 主控台中文防亂碼


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""


# ---------- 合約檢查用的底層請求 ----------

def _post(client: httpx.Client, body: dict, key: str | None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
    return client.post("/retrieval", json=body, headers=headers)


def _validate_records(payload: dict) -> list[str]:
    """驗 Dify External Knowledge API 回應形狀，回問題清單（空 = 合格）。"""
    problems: list[str] = []
    if not isinstance(payload, dict) or "records" not in payload:
        return ["回應缺 'records' 欄"]
    recs = payload["records"]
    if not isinstance(recs, list):
        return ["'records' 不是 list"]
    for i, r in enumerate(recs):
        p = f"records[{i}]"
        if not isinstance(r, dict):
            problems.append(f"{p} 不是 object"); continue
        if not isinstance(r.get("content"), str):
            problems.append(f"{p}.content 非字串")
        elif not r["content"].strip():
            problems.append(f"{p}.content 為空（可用但無意義）")
        if not isinstance(r.get("score"), (int, float)):
            problems.append(f"{p}.score 非數字")
        if not isinstance(r.get("title"), str):
            problems.append(f"{p}.title 非字串")
        md = r.get("metadata", None)
        if md is None or not isinstance(md, dict):
            problems.append(f"{p}.metadata 必須是 object 且非 null（Dify 硬性要求）")
    return problems


# ---------- 查詢探針 ----------

def query_and_print(client: httpx.Client, query: str, top_k: int, key: str | None) -> int:
    body = {"knowledge_id": "test", "query": query, "retrieval_setting": {"top_k": top_k}}
    print(f"→ POST /retrieval  query={query!r}  top_k={top_k}")
    try:
        resp = _post(client, body, key)
    except httpx.ConnectError:
        print("✗ 連不到端點。先啟動：python dify_retrieval_api.py，並確認 --url 正確。")
        return 1
    if resp.status_code != 200:
        print(f"✗ HTTP {resp.status_code}：{resp.text[:300]}")
        return 1

    payload = resp.json()
    recs = payload.get("records", [])
    print(f"← 200  回 {len(recs)} 筆\n" + "=" * 72)
    if not recs:
        print("（0 筆。若確定該有結果：檢查 Qdrant collection 名、以及 Dify 端 Score Threshold 是否誤開。）")
    for i, r in enumerate(recs, 1):
        md = r.get("metadata") or {}
        proj = md.get("project_name") or "?"
        page = md.get("page")
        src = md.get("file_name") or md.get("file_key") or ""
        content = (r.get("content") or "").replace("\n", " ")
        preview = content[:200] + ("…" if len(content) > 200 else "")
        print(f"[{i}] score={r.get('score'):.4f}  建案={proj}  頁={page}")
        print(f"    title: {r.get('title')}")
        print(f"    來源 : {src}")
        print(f"    正文 : {preview}")
        print("-" * 72)

    problems = _validate_records(payload)
    if problems:
        print("⚠️ 合約檢查發現問題：")
        for p in problems:
            print("   -", p)
        return 1
    print("✓ 回應符合 Dify 合約。")
    return 0


# ---------- 合約測試套件 ----------

def run_suite(client: httpx.Client, key: str | None) -> int:
    results: list[Result] = []

    # /health
    try:
        h = client.get("/health")
        hj = h.json()
        auth_on = hj.get("auth") == "on"
        results.append(Result("/health 可達且回 ok/loading",
                              h.status_code == 200 and hj.get("status") in ("ok", "loading"),
                              str(hj)))
    except httpx.ConnectError:
        print("✗ 連不到端點。先啟動 dify_retrieval_api.py 再跑 --suite。")
        return 1

    # auth（僅端點啟用驗證時才測）
    if auth_on:
        r = _post(client, {"query": "x"}, key=None)
        results.append(Result("缺 auth → 403/1001",
                              r.status_code == 403 and r.json().get("error_code") == 1001,
                              f"{r.status_code} {r.json()}"))
        r = _post(client, {"query": "x"}, key="definitely-wrong-key")
        results.append(Result("錯 key → 403/1002",
                              r.status_code == 403 and r.json().get("error_code") == 1002,
                              f"{r.status_code} {r.json()}"))
    else:
        results.append(Result("auth 檢查", True, "端點 auth:off（開發模式）→ 跳過"))

    # 缺 query → 400
    r = _post(client, {"knowledge_id": "t"}, key=key)
    results.append(Result("缺 query → 400", r.status_code == 400, f"{r.status_code} {r.text[:120]}"))

    # 正常查詢 → 200 + 形狀
    r = _post(client, {"knowledge_id": "t", "query": "公設比", "retrieval_setting": {"top_k": 3}}, key=key)
    if r.status_code == 200:
        problems = _validate_records(r.json())
        results.append(Result("正常查詢 → 200 且形狀合格", not problems,
                              "OK" if not problems else "；".join(problems)))
    else:
        results.append(Result("正常查詢 → 200 且形狀合格", False, f"HTTP {r.status_code}: {r.text[:200]}"))

    # 報告
    print("=" * 72)
    passed = 0
    for res in results:
        mark = "✓ PASS" if res.passed else "✗ FAIL"
        print(f"{mark}  {res.name}")
        if res.detail:
            print(f"        {res.detail}")
        passed += res.passed
    print("=" * 72)
    print(f"{passed}/{len(results)} 通過")
    return 0 if passed == len(results) else 1


def build_client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url.rstrip("/"), timeout=60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dify_retrieval_api 端點測試")
    ap.add_argument("query", nargs="?", help="要檢索的問題（不給且未 --suite 時用預設題）")
    ap.add_argument("--url", default=os.environ.get("DIFY_API_URL", "http://localhost:8100"))
    ap.add_argument("--key", default=os.environ.get("DIFY_API_KEY", ""))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--suite", action="store_true", help="跑合約測試套件")
    args = ap.parse_args(argv)

    key = args.key or None
    with build_client(args.url) as client:
        if args.suite:
            return run_suite(client, key)
        query = args.query or "各案公設比排序"
        return query_and_print(client, query, args.top_k, key)


if __name__ == "__main__":
    raise SystemExit(main())
