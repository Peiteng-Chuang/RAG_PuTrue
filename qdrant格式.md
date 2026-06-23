# Qdrant 儲存格式規範 v2

> 本檔是 RAG 系統向量資料庫的 single source of truth。
> 任何寫入 / 檢索程式都應依此 schema 實作，schema 變動須 bump `pipeline_version`。

---

## 0. 架構決策（locked）

| # | 項目 | 決議 |
|---|---|---|
| 1 | Embedding 模型 | **BAAI/bge-m3**（dense 1024 維 + learned sparse；中文多語 + hybrid 原生） |
| 2 | Qdrant 部署 | **Docker server 模式**（支援 10-12 人團隊並發） |
| 3 | Review 狀態持久化 | **Sidecar JSON 一檔一份**：`mkdata/{stem}.review.json` |
| 4 | Hybrid 檢索 | **Qdrant Named Vectors**（同一 point 同時帶 dense + sparse，Qdrant 端 RRF fusion） |
| 5 | 多模態 | **分兩個 collection**（text 與 images），靠 `file_hash + page` join |

---

## 1. 環境參數

```
QDRANT_URL = http://<host>:6333          # Docker server
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM_DENSE = 1024
EMBEDDING_DISTANCE_DENSE = COSINE
EMBEDDING_DISTANCE_SPARSE = DOT          # Qdrant sparse 預設
PIPELINE_VERSION = "2.1.0"               # 2.1.0：新增 metadata.source.doc_type
EMBEDDING_VERSION = "v1"                 # 換模型 / 換 prefix 策略時 bump
```

---

## 2. Collection: `putrue_rag_text_v1`

### 2.1 Named Vectors

```
vectors_config = {
  "text_dense":  VectorParams(size=1024, distance=COSINE)
}
sparse_vectors_config = {
  "text_sparse": SparseVectorParams()
}
```

### 2.2 Payload Indexes（建立時就建好，避免事後 reindex）

| 欄位 | 型別 | 用途 |
|---|---|---|
| `metadata.source.project_name` | keyword | 篩建案 |
| `metadata.source.doc_type` | keyword | 篩文件種類（專案報告／教育訓練／法規規範／部門規則…） |
| `metadata.source.file_hash` | keyword | 重 ingest 偵測、文件 join |
| `metadata.source.file_key` | keyword | 回 review UI 的錨點 |
| `metadata.location.page` | integer | 範圍查詢、跨頁擴張 |
| `metadata.location.headings_flat` | keyword[] | section heading 篩選 |
| `chunking.strategy` | keyword | 舊版 chunk 標記為 stale |
| `visuals.has_image` | bool | 只查含圖 chunk |
| `visuals.image_label` | keyword | 「拆屋前現況照片」這類精確類別查詢 |
| `labels.label_keys` | keyword[] | 自訂標籤 key 篩選 |
| `sys_info.review_status` | keyword | 只 ingest `approved` |
| `sys_info.embedding_model` | keyword | 多模型並存時切換 |

### 2.3 Point ID 策略

```
point_id = uuid5(NAMESPACE_DNS, f"{file_hash}|{page}|{section_idx}|{chunk_idx}")
```
- Deterministic → 重 embed 自動覆蓋同一 point
- 跨檔案 / 跨頁不衝突
- `delete_collection` 不再是必要動作

### 2.4 Payload 完整範例

```json
{
  "metadata": {
    "source": {
      "project_name": "勤美璞真城仰",
      "doc_type": "專案報告",
      "file_key": "勤美璞真城仰\\1120510_勤美璞真城仰_機電銷講簡報.pptx",
      "file_name": "1120510_勤美璞真城仰_機電銷講簡報.pptx",
      "file_path": "D:/璞真RAG資料夾/12.個案銷講資料/勤美璞真城仰/1120510_勤美璞真城仰_機電銷講簡報.pptx",
      "file_hash": "e99a18c428cb38d5f260853678922e03",
      "doc_title": "勤美璞真城仰機電銷講簡報"
    },
    "location": {
      "page": 13,
      "page_label": "第 13 頁",
      "headings": {
        "3": "(二)、規劃設計報告與建議",
        "4": "2、拆屋前現況照片"
      },
      "headings_flat": ["(二)、規劃設計報告與建議", "2、拆屋前現況照片"],
      "breadcrumb": ["勤美璞真城仰機電銷講簡報", "第 13 頁", "(二)、規劃設計報告與建議", "2、拆屋前現況照片"],
      "current_header": "2、拆屋前現況照片",
      "section_idx": 1,
      "chunk_idx": 0,
      "chunk_idx_global": 47
    }
  },
  "content": {
    "text": "2.1 日式屋瓦  2.2 基地原貌  本因坊結案報告",
    "text_with_prefix": "[勤美璞真城仰 | 1120510_勤美璞真城仰_機電銷講簡報 | P13 | (二)、規劃設計報告與建議 > 2、拆屋前現況照片] 2.1 日式屋瓦  2.2 基地原貌  本因坊結案報告",
    "md_content": "2.1 日式屋瓦\n\n2.2 基地原貌\n\n本因坊結案報告",
    "token_count": 28,
    "char_count": 32
  },
  "visuals": {
    "has_image": true,
    "image_label": "2、拆屋前現況照片",
    "image_count": 5,
    "images": [
      {
        "file_name": "page13_img1.jpeg",
        "local_path": "./mkdata/1120510_勤美璞真城仰_機電銷講簡報_image/page13_img1.jpeg",
        "md_ref": "![page13_img1.jpeg](images/page13_img1.jpeg)",
        "alt_text": "page13_img1.jpeg"
      }
    ]
  },
  "labels": {
    "document": [
      { "key": "context", "value": "璞真建設股份有限公司台北案場機電工法簡報" }
    ],
    "page": [],
    "label_keys": ["context"]
  },
  "chunking": {
    "strategy": "heading+token-v1",
    "parent_section_id": "550e8400-e29b-41d4-a716-446655440000",
    "prev_chunk_id": "660e8400-e29b-41d4-a716-446655440001",
    "next_chunk_id": "770e8400-e29b-41d4-a716-446655440002"
  },
  "sys_info": {
    "ingestion_time": "2026-05-11T09:15:57Z",
    "pipeline_version": "2.0.0",
    "embedding_model": "BAAI/bge-m3",
    "embedding_version": "v1",
    "review_status": "done",
    "reviewer": "ai.team@cmp.com.tw",
    "reviewed_at": "2026-05-10T14:20:00Z"
  }
}
```

### 2.5 關鍵欄位語意

| 欄位 | 用途 |
|---|---|
| `metadata.source.doc_type` | 文件種類維度，與 `project_name` 正交。決定優先序：**sidecar 逐檔 `doc_type` 覆寫 > 資料夾對照表 `mkdata/doc_type_map.json`（`{資料夾: doc_type}`）> 預設 `未分類`**。對照表一個資料夾一筆，避免逐檔 bulk 寫 sidecar |
| `content.text` | 純文字，給 LLM 上下文用、給 sparse 編碼用 |
| `content.text_with_prefix` | **dense embedding 的輸入**，prefix 注入 project/檔名/heading 提升命中率 |
| `content.md_content` | 帶 markdown 原樣，給 UI 顯示 |
| `location.headings_flat` | 扁平化 headings.values()，用於 Qdrant payload index（Qdrant 不支援 dict-typed index） |
| `labels.label_keys` | 扁平化 doc_labels + page_labels 所有 keys，用於 filter |
| `chunking.{prev,next}_chunk_id` | parent-doc retrieval / 鄰近塊擴張 |

---

## 3. Collection: `putrue_rag_images_v1`

### 3.1 Vectors

```
vectors_config = {
  "image_clip": VectorParams(size=768, distance=COSINE)
}
```
> 預設用 `jina-clip-v2`（多語、768 維）。P6 階段再決定，留位即可。

### 3.2 Payload Indexes

| 欄位 | 型別 |
|---|---|
| `source.project_name` | keyword |
| `source.file_hash` | keyword |
| `source.file_key` | keyword |
| `location.page` | integer |
| `image_label` | keyword |
| `linked_text_chunk_ids` | keyword[] |

### 3.3 Point ID 策略

```
point_id = uuid5(NAMESPACE_DNS, f"{file_hash}|{page}|{image_filename}")
```

### 3.4 Payload 範例

```json
{
  "source": {
    "project_name": "勤美璞真城仰",
    "file_key": "勤美璞真城仰\\1120510_勤美璞真城仰_機電銷講簡報.pptx",
    "file_hash": "e99a18c428cb38d5f260853678922e03"
  },
  "location": {
    "page": 13,
    "headings_flat": ["(二)、規劃設計報告與建議", "2、拆屋前現況照片"]
  },
  "image": {
    "file_name": "page13_img1.jpeg",
    "local_path": "./mkdata/1120510_勤美璞真城仰_機電銷講簡報_image/page13_img1.jpeg",
    "image_hash": "f3a1...",
    "width": 1920,
    "height": 1080,
    "format": "jpeg"
  },
  "image_label": "2、拆屋前現況照片",
  "linked_text_chunk_ids": [
    "uuid-of-text-chunk-that-references-this-image"
  ],
  "sys_info": {
    "ingestion_time": "2026-05-11T09:15:57Z",
    "image_model": "jinaai/jina-clip-v2",
    "pipeline_version": "2.0.0"
  }
}
```

---

## 4. Sidecar JSON 格式：`mkdata/{stem}.review.json`

review UI 編輯狀態的持久化檔，與 MD 平行存放。ingest pipeline 讀這份檔生成 chunks。

```json
{
  "file_key": "勤美璞真城仰\\1120510_勤美璞真城仰_機電銷講簡報.pptx",
  "file_hash": "e99a18c428cb38d5f260853678922e03",
  "review_status": "done",
  "reviewer": "ai.team@cmp.com.tw",
  "reviewed_at": "2026-05-10T14:20:00Z",
  "custom_labels": {
    "document": [
      { "key": "context", "value": "璞真建設股份有限公司台北案場機電工法簡報" }
    ],
    "pages": {
      "12": [{ "key": "section", "value": "拆屋階段" }]
    }
  },
  "split_settings": {
    "mode": "delimiter",
    "delim": "\\n",
    "max_tokens": 500,
    "overlap_tokens": 80
  },
  "delete_history": [
    {
      "page_idx": 12,
      "page_num": 13,
      "md_path": "images/page13_img7.jpeg",
      "removed_line": "![page13_img7.jpeg](images/page13_img7.jpeg)\n",
      "line_offset": 245,
      "abs_path": "./mkdata/.../page13_img7.jpeg",
      "trash_path": "./_md_trash/.../page13_img7.jpeg",
      "deleted_at": "2026-05-10T13:55:00Z"
    }
  ]
}
```

---

## 4.1 `review_status` 工作流（自動轉換，UI 不可手動）

**四態系統**，由系統依「使用者動作」自動切換：

| 狀態 | 顏色 | 觸發條件 |
|---|---|---|
| `unprocessed` | 🔴 紅 | 預設值；sidecar 尚未存在或從未編輯 |
| `processing` | 🟡 黃 | 任何 mutation：加/刪 label、刪/復原圖、改 split、儲存 MD、改 reviewer |
| `encoded` | 🟢 綠 | **Tab 4 全檔批次 encode 成功**（必須在最後一頁），本地向量已建 |
| `ingested` | 🔵 淡藍 | **Tab 5 Qdrant 上傳成功**（必須在最後一頁），Qdrant 已收下 vectors+payload；蓋 `reviewed_at` 時間戳 |

### 轉換規則

- `unprocessed` → `processing`：任何 mutation
- `processing` → `processing`：no-op（避免多餘磁碟寫入，但仍 persist 其他欄位）
- 任意態 → `processing`：mutation 一定降回 processing（代表本地快取或 Qdrant 內容已 stale）
- `processing` → `encoded`：**僅** Tab 4 全檔 encode 成功時觸發 `mark_encoded`
- `encoded` → `ingested`：**僅** Tab 5 Qdrant upsert 全部成功時觸發 `mark_ingested`
- `unprocessed`/`encoded`/`ingested` → `processing`：任何後續 mutation

### UI 規則

- 狀態 badge 自動依當前狀態著色，**不開放下拉手動指定**
- Tab 4「全檔批次 encode」與 Tab 5「上傳到 Qdrant」按鈕**只在最後一頁 enabled**
  - 強制 reviewer 至少切到末頁才能 finalise
- Sidebar 檔案 selectbox 每個項目前綴 emoji（🔴🟡🟢🔵），下方顯示色票圖例 + 各狀態檔案數
- 舊版 sidecar 自動遷移：
  - `unreviewed` → `unprocessed`
  - `in_progress` / `needs_rework` → `processing`
  - `approved` → `ingested`
  - `done`（舊 3 態末態）→ `ingested`

## 5. 重 ingest 條件

下列任一觸發時，該檔案的所有 points 應重新 embed 並 upsert（point_id deterministic，會自動覆蓋）：

| 觸發 | 處理 |
|---|---|
| `file_hash` 變動（原檔被修改） | 全檔重 chunk + 重 embed |
| `chunking.strategy` 升版 | 全檔重 chunk + 重 embed |
| `sys_info.embedding_model` 或 `embedding_version` 變動 | 重 embed（chunks 不變） |
| `labels` 或 `review_status` 變動 | 只重寫 payload，向量不動（Qdrant `set_payload`） |
| `delete_history` 新增 | 只重寫 payload，向量不動 |

---

## 6. 版本歷史

| Version | Date | Notes |
|---|---|---|
| 1.0 | 2026-05-? | 初稿，single dense vector，無 hybrid，payload 欄位最小 |
| 2.0.0 | 2026-05-11 | bge-m3 hybrid（dense+sparse named vectors）、雙 collection（text + images）、sidecar JSON 持久化、point_id deterministic、payload 對齊 review UI metadata、加入 chunking 鄰接欄位 |
| 2.1.0 | 2026-06-23 | 新增 `metadata.source.doc_type`（文件種類）keyword 欄位 + 索引；來源為資料夾對照表 `doc_type_map.json` + sidecar 逐檔覆寫；既有 collection 由 `ensure_payload_indexes` 補索引，舊 points 需重 ingest 或 `set_payload` 回填 |
