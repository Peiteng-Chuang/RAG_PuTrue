# Claude Code Memory Export — c--project-file-RAG-PuTrue

**Exported**: 2026-05-21
**Source PC**: 原 PC（含 `.claude/projects/c--project-file-RAG-PuTrue/memory/`）
**Total memories**: 11 + 1 index

---

## 如何在另一台 PC 匯入

### 方法 A（手動分檔，最直接）

每個 `<!-- ===== FILE: xxx.md ===== -->` 區段就是一個獨立 memory 檔。在目標 PC 上：

1. 確認目錄存在：`~/.claude/projects/c--project-file-RAG-PuTrue/memory/`
   （Windows: `C:\Users\<你>\.claude\projects\c--project-file-RAG-PuTrue\memory\`）
2. 把每段內容（不含 `<!-- ===== FILE ... ===== -->` marker 那一行）存成同名檔案

### 方法 B（用底下提供的 Python script 自動分檔）

```python
# split_memory_export.py
from pathlib import Path
import re

EXPORT_FILE = "CLAUDE_MEMORY_EXPORT.md"
OUT_DIR = Path.home() / ".claude" / "projects" / "c--project-file-RAG-PuTrue" / "memory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

content = Path(EXPORT_FILE).read_text(encoding="utf-8")

# 分段：每個 <!-- ===== FILE: xxx ===== --> 開始一段
parts = re.split(r"^<!-- ===== FILE: (.+?) ===== -->\s*\n", content, flags=re.MULTILINE)
# parts = [前言, filename1, body1, filename2, body2, ...]
for i in range(1, len(parts), 2):
    filename = parts[i].strip()
    body = parts[i + 1].rstrip()
    # 移除 trailing 區段分隔符（如 "---\n" 在最末段）
    body = re.sub(r"\n+---\n*$", "", body)
    (OUT_DIR / filename).write_text(body + "\n", encoding="utf-8")
    print(f"✓ {filename}")
print(f"\n寫入 {OUT_DIR}")
```

跑：`python split_memory_export.py`

### 驗收

匯入後跑：
```bash
ls ~/.claude/projects/c--project-file-RAG-PuTrue/memory/
```
應該看到 12 個 .md 檔（11 memory + 1 MEMORY.md）。

---

## ⚠️ 兩條最重要的行為規則（請優先確認載入）

以下兩條是 2026-05-21 教訓後的最新 feedback memory，下次 session 必須**立刻生效**：

1. **`feedback_data_dependency_audit_first.md`** — 寫/讀任何會 disk-write 的程式都必須先稽核下游依賴
2. **`feedback_tool_scope_isolation.md`** — `md_review_ui` 跟 `mkd_generator` 絕對不可同 session 變動

---

<!-- ===== FILE: MEMORY.md ===== -->
- [Project Overview — Construction Engineering RAG](project_overview.md) — 1TB 建築工程 PDF（CAD 圖、表格、術語）→ Markdown → RAG，重點在前處理品質與來源追溯
- [ETL Pipeline Design — Asymmetric ETL Engine](etl_pipeline_design.md) — 三支柱：自動分流／結構化還原（`## 第[n]頁` Citation 錨點）／多模態強關聯
- [Tech Stack — Hybrid Asymmetric ETL](tech_stack.md) — PyMuPDF（快路徑）+ Marker/Surya/Nougat（慢路徑 GPU）+ Py3.12 venv + PyTorch CUDA + Regex/Multiprocessing
- [RAG Storage Stack — Qdrant + bge-m3](rag_storage_stack.md) — bge-m3 hybrid（dense+sparse named vectors）/ Qdrant Docker server / 雙 collection（text+images）/ sidecar JSON 一檔一份，詳見 qdrant格式.md
- [Implementation Progress](implementation_progress.md) — P0-P3 已完成（含 4 態狀態機、ace 編輯器、Tab 4 embedding、Tab 5 Qdrant 寫入）；P4+ 待做；開機後從 Docker container 跑 Qdrant 開始
- [Triage Threshold — Why 200 Vector Paths](triage_threshold_rationale.md) — `page.get_drawings() > 200` 切 Marker；CAD 結構符號每個就上百向量，閾值是成本/精度交界
- [Hardware Environment](hardware_environment.md) — RTX 4090 24GB / Win11 / 單機單卡，吞吐與 batch 規劃以此為基準
- [User Role — Solo Builder](user_role.md) — 獨立開發者，無團隊規矩，可直接給技術判斷不需中性化
- [Feedback — Strict Structure Anchors](feedback_strict_structure_anchors.md) — 結構性 regex（頁錨點等）優先嚴守原樣 + UI 防呆，而非 regex 放寬容錯
- [⚠️ Feedback — Data Dependency Audit FIRST](feedback_data_dependency_audit_first.md) — 遇到**任何**會 disk-write/delete/overwrite 的程式（自建 OR 既有）都必須主動稽核下游依賴 + 預設安全 + 警告危險工具。2026-05-21 `reset_pptx_for_rerun.py` + 自建 v5 雙重事件教訓。已知危險程式清單見內文
- [⚠️ Feedback — Tool Scope Isolation](feedback_tool_scope_isolation.md) — `md_review_ui`（標記工具，downstream）跟 `mkd_generator`（ETL，upstream）**絕對不可同 session 同時變動**。任務開始必須先標明範圍；跨範圍要拆兩段 + 各自 commit

---

<!-- ===== FILE: project_overview.md ===== -->
---
name: Project Overview — Construction Engineering RAG
description: Final goal, scale, data characteristics, and success criteria for the industrial-grade construction-engineering RAG system
type: project
originSessionId: 325c85d0-e0f0-4176-aee2-87600c769db5
---
最終目標：建立一個「工業級建築工程知識 RAG 系統」。

**資料規模與特性：**
- 約 1TB 非結構化資料：建築結案報告、施工圖說等
- 與一般商業文件差異：包含大量 CAD 向量圖、複雜工程表格、高密度專業術語
- PDF 多為「掃描不出來」或「結構混亂」的類型

**核心挑戰（成功關鍵）：**
將上述 PDF 轉化為 LLM 可理解的高質量 Markdown 數據庫，作為 RAG 索引基礎。

**最終驗收標準：**
精準的知識檢索 + 可追溯性（traceability，能回指原始文件出處）。

**Why:** 使用者明確指出此專案不同於一般 RAG，瓶頸在於「前處理／資料轉換」品質，而非檢索演算法本身。

**How to apply:**
- 提出技術方案時，優先考慮對 CAD 向量圖、工程表格、專業術語的處理能力
- 評估 OCR / PDF parser / Markdown 轉換工具時，把「結構混亂的工程 PDF」當主要測試案例，而非一般文書
- 任何檢索優化建議都要保留來源追溯（chunk → 原始頁／圖／表）
- 規模 1TB 級代表 pipeline 必須考慮批次處理、增量更新、儲存成本

---

<!-- ===== FILE: etl_pipeline_design.md ===== -->
---
name: ETL Pipeline Design — Asymmetric ETL Engine
description: Core design philosophy and three pillars of the PDF→Markdown preprocessing engine; constrains all pipeline-related suggestions
type: project
originSessionId: 325c85d0-e0f0-4176-aee2-87600c769db5
---
本程式定位為整個 RAG 系統的**前處理引擎（ETL Pipeline）**，設計理念是
**「效能與精度的非對稱平衡」（Asymmetric ETL）**。

## 三大設計支柱

### 1. 自動化分流（Triage）
- 自動判別「簡單頁面」 vs 「複雜向量頁面」
- 目的：避免對所有頁面都跑昂貴的 AI 運算
- 簡單頁走快路徑，複雜向量頁才走 AI / 多模態路徑

### 2. 結構化還原（Structural Restoration）
- 將雜亂的 PDF 重新編排
- **每頁必須有清晰的 `## 第[n]頁` 標籤**
- 原因：對後續 RAG 的 **Citation（引用來源標註）** 至關重要

### 3. 多模態準備（Multimodal Readiness）
- 提取文字同時，**精確裁切並命名圖片**
- 文字與圖像「強關聯」存儲
- 為後續多模態 RAG 預留接口

**Why:** 1TB 規模下，全頁過 AI 不可行（成本 + 時間）；同時「結構混亂」與「向量圖密集」是品質瓶頸。非對稱策略是在成本/精度之間找到工程實務的平衡點。

**How to apply:**
- 任何頁面處理建議都要先問「這是簡單頁還是複雜向量頁？」走哪條分支？
- **絕不能破壞 `## 第[n]頁` 標籤格式** — 它是 Citation 的錨點
- 提取圖片時必須保留與所在頁／所在文字段落的關聯（檔名命名規則 + metadata）
- 不要建議「全部用 GPT-4V/Vision 處理」這類一刀切方案，違反非對稱原則
- 評估新工具時，優先看它能否塞進「分流 → 對應路徑」的架構，而非取代整個 pipeline

---

<!-- ===== FILE: tech_stack.md ===== -->
---
name: Tech Stack — Hybrid Asymmetric ETL Pipeline
description: Component-level technology choices for the PDF→Markdown ETL pipeline, with each tool's specific role
type: project
originSessionId: 325c85d0-e0f0-4176-aee2-87600c769db5
---
架構:**混合式、非對稱式 ETL**，目標是 1TB 資料處理效率最大化。

## 組件對應表

| 角色 | 技術 | 功能 |
|---|---|---|
| 基礎解析（PDF Parser） | **PyMuPDF (Fitz)** | 全文預掃描、文字座標提取、點陣圖（Bitmap）抽離、分頁 |
| 深度解析（OCR / VLM） | **Marker**（基於 Surya & Nougat） | 複雜佈局、表格、向量圖頁面的 Markdown 重構與視覺渲染 |
| 環境管理 | **Python 3.12 + Venv** | 獨立虛擬環境，解決 Windows 路徑與相依性衝突（如 opencv 優先級） |
| 運算控制 | **PyTorch (CUDA)** | GPU 加速 Marker 推論與佈局偵測 |
| 邏輯控制 | **Regex + Multiprocessing** | Markdown 語法校正、錨點縫合、多進程保護機制 |

## 角色分工的設計意圖

- **PyMuPDF = 快路徑** → 用於分流階段的「全文預掃描」，便宜快速取得文字與座標
- **Marker = 慢路徑** → 只對被分流判定為「複雜向量頁」的頁面執行，因為它要過 GPU
- 兩者是「先 Fitz 後 Marker」的串接，不是並聯競爭

**Why:**
- PyMuPDF 純 CPU、無模型，適合做 1TB 規模的初篩
- Marker 走 Surya（layout / OCR）+ Nougat（公式 / 學術佈局），是目前對工程 PDF 結構還原較強的開源組合
- Windows + Python 3.12 + opencv 的相依衝突是已知踩過的坑，所以鎖死 venv 隔離

**How to apply:**
- 建議新套件前，先確認它在「快路徑」還是「慢路徑」，不要把 GPU 工具塞進預掃描階段
- 動到 opencv / numpy / torch 版本時，務必警告 venv 衝突風險（Windows 環境特別敏感）
- Markdown 後處理一律走 Regex + Multiprocessing 既有架構，不要引入新框架（如 LangChain document loaders）取代
- 「錨點縫合」指的是 PyMuPDF 與 Marker 兩條路徑輸出合併時，靠 `## 第[n]頁` 對齊 — 改任何分頁邏輯前先想清楚對縫合的影響
- Multiprocessing 是「保護機制」 — 暗示 Marker 可能會 crash／OOM，子進程隔離避免拖垮整批

---

<!-- ===== FILE: rag_storage_stack.md ===== -->
---
name: RAG Storage Stack — Qdrant + bge-m3
description: 向量資料庫層的技術選型與 schema 架構決策（不含 ETL，那是 tech_stack.md）
type: project
originSessionId: 52f80103-abe9-4ad7-910b-5fb23f35cdf7
---
ETL 下游、RAG 上游的儲存與檢索層決策。完整 schema 規範在 `qdrant格式.md`（v2.0.0）。

## 五項決策（2026-05-11 拍板）

| # | 項目 | 決議 | Why |
|---|---|---|---|
| 1 | Embedding 模型 | `BAAI/bge-m3` | 中文多語強、原生支援 dense+sparse hybrid、一次推理出兩種向量、4090 24GB 跑 batch=32-64 沒壓力 |
| 2 | Qdrant 部署 | Docker server | 團隊 10-12 人共用，本地 `path=` 模式不支援並發 |
| 3 | Review 持久化 | Sidecar JSON 一檔一份 `mkdata/{stem}.review.json` | git diff 看得到變更、無 lock 衝突、跟 MD 平行 |
| 4 | Hybrid 實作 | Qdrant Named Vectors（dense + sparse 同 point） | bge-m3 一次出兩種、Qdrant 1.10+ 原生 fusion、運維只一套系統 |
| 5 | 多模態 | 分兩個 collection（text / images） | 不同實體不該共用 collection；換 vision model 不會動到 1TB 文字 |

## How to apply

- **Collection 命名**：`putrue_rag_text_v1` + `putrue_rag_images_v1`，bump version suffix 而不是 mutate schema
- **Point ID**：必須 deterministic（`uuid5(file_hash|page|section_idx|chunk_idx)`），不要再用 pipeline.ipynb 的整數索引，那是踩過的坑
- **Embedding 輸入**：用 `content.text_with_prefix`（含 project/檔名/heading prefix），不要用裸 `content.text`，prefix 對命中率提升大
- **Sparse 維護**：bge-m3 推理時一次拿 dense + sparse，不要分兩次跑模型
- **重 ingest 條件**：只有 `file_hash` / `chunking.strategy` / `embedding_model` 變動才重 embed；單純 labels 編輯走 `set_payload` 不動向量
- **review_status filter**：批次 ingest 只吃 `approved` 的，避免半成品污染檢索結果

## review_status 工作流（locked 2026-05-12，從 3 態擴 4 態）

四態自動轉換，UI 不開放手動下拉：

- `unprocessed`（🔴 紅）：預設；sidecar 不存在或未編輯
- `processing`（🟡 黃）：任何 mutation（label / image / split / MD save / reviewer）自動觸發
- `encoded`（🟢 綠）：**Tab 4 全檔 encode 成功**（必須在最後一頁）時 `mark_encoded` 觸發，本地向量建好
- `ingested`（🔵 淡藍）：**Tab 5 Qdrant 上傳成功**（必須在最後一頁）時 `mark_ingested` 觸發，蓋 `reviewed_at`

**Why 拆 encoded / ingested：** 本地建向量跟實際進 DB 是兩個不同的承諾。拆開後 sidebar 可以一眼看出「哪些檔處理完但還沒進庫」。

**Why 任何 mutation 都降回 processing：** 強制以「資料同步狀態」為真相，避免人為手動 mark 但內容已改。

**How to apply:**
- 任何會改 sidecar 或 chunks 的程式碼必須呼叫 `mark_processing(file_key)`
- Tab 4 全檔 encode 成功必須呼叫 `mark_encoded(file_key)`
- Tab 5 Qdrant upsert 全部成功必須呼叫 `mark_ingested(file_key)`
- 沒呼叫 = 狀態卡住、sidebar 顏色錯
- Tab 4 / Tab 5 的 commit 按鈕 UI gating 在最後一頁，是強迫 reviewer 通讀的軟約束
- 舊 3 態的 `done` → 自動遷移為 `ingested`（`LEGACY_STATUS_MIGRATION` 表）

## 反模式（不要做）

- 不要把 dense + sparse + image_clip 全塞同一 collection 的 named vectors —— 那是「同實體多面向」設計，文字 chunk 跟圖片是不同實體
- 不要每次 ingest 都 `delete_collection`（pipeline.ipynb 的舊習慣），用 deterministic point_id + upsert
- 不要用外部 BM25（Elasticsearch）取代 sparse —— bge-m3 的 learned sparse 在中文同義詞處理上常打贏純 BM25，且不用維護兩套系統
- 不要在 chunk 文字裡保留 `![](...)` 圖片引用 —— 用 `strip_image_refs`，圖片靠 `visuals.images[]` 結構化存

---

<!-- ===== FILE: implementation_progress.md ===== -->
---
name: Implementation Progress — md_review_ui.py + Qdrant Pipeline
description: P0-P7 升級計畫進度盤點，已完成階段、未做階段、開機後從哪繼續
type: project
originSessionId: 52f80103-abe9-4ad7-910b-5fb23f35cdf7
---
`md_review_ui.py` 是 review UI + ETL → Qdrant 寫入的一條龍工具。升級計畫骨架在 `qdrant格式.md`。

## 2026-05-14 session（UI 體感調校）

- **Sidebar selectbox 折疊顯示不刷新色球**：encode/ingest 完 rerun 後，下拉**展開**的選項列表色球會更新，但**折疊狀態的 selected label** 還是舊色。原因是 Streamlit 在 widget `key` 不變時，會快取已選項目的折疊 label。**修法**：sidebar 計算 `_status_sig = tuple(_file_status_map[fk] for fk in file_keys)`；簽章一變就 bump `_sb_key_version`、selectbox key 從 `sidebar_file_selectbox` 改成 `sidebar_file_selectbox_v{ver}`。Migration block 把舊 key 的選擇值搬到新 key、避免使用者選擇被重置。Tab 6 「跳回該檔該頁」按鈕同步改寫到當前版本 key。
- **編輯器隱藏 `## 第 N 頁` 行**：使用者反映不小心在頁標頭後加字，PDF 預覽跳到別張投影片。原因是 `parse_md` regex `^## 第 (\d+) 頁\n` 嚴格匹配，頁標頭壞掉 → 該頁錨點失效 → page_idx 對映漂移。配合 [[feedback-strict-structure-anchors]]，**選 UI 防呆而非放寬 regex**：Tab 1 編輯器 value 改成 `strip_page_header(page_md)`，上方放藍底唯讀資訊條（頁碼 + 「請從 `###` 開始標」hint）。`commit_editor_if_dirty` commit 時 `full_edited = f"## 第 {page_num} 頁\n{edited}"` 黏回頁標頭再 `replace_page`，下游 parse 結構不變。
- **發現未修的潛在 bug — 跨頁編輯不會 commit 到 `current_md`**：`commit_editor_if_dirty(selected_key, st.session_state.page_idx)` 在 line 1481 跑，但翻頁鈕已把 `page_idx` 改成新值，函式查的是「新頁」的 editor key，**舊頁編輯內容停留在 widget session_state 裡卻沒進 `current_md`**。視覺上翻回去仍看得到（widget state 還在），但「儲存 .md」/「Embedding」走 `current_md` 就會丟。修法：翻頁鈕在改 `page_idx` 之前先 `commit_editor_if_dirty(selected_key, 舊 page_idx)`。**暫未動，記下備修**。

## 已完成（截至 2026-05-12）

| 階段 | 內容 | 狀態 |
|---|---|---|
| **P0** | Sidecar 持久化 (`mkdata/{stem}.review.json`)、review_status 自動轉換、reviewer 欄位（後已拔）| ✓ |
| **P0e** | Review 狀態機從 3 態擴 4 態（unprocessed/processing/encoded/ingested），各態自動轉換、舊 sidecar 自動遷移 | ✓ |
| **P0f** | streamlit-ace 整合（keybinding="vscode"、Alt+↑↓、多游標）、commit_editor_if_dirty 取代 on_change | ✓ |
| **P0g/h** | 文字暫存區：嘗試 chip + click-to-insert → 撤回 → 改回純 textarea → 改成 st_ace（min_lines/max_lines auto-grow） | ✓ |
| **P0i** | Sidebar 檔案 selectbox 加 emoji 狀態前綴 + 色票圖例（含各狀態檔案數）| ✓ |
| **P1** | `build_chunk_payload` + `build_all_chunks_for_doc` 對齊 qdrant格式.md v2 schema | ✓ |
| **P2** | Tab 4 Embedding（bge-m3 hybrid dense+sparse、本地快取、cache 失效偵測）| ✓ |
| **P2c** | Tab 4 全檔 encode 按鈕 gating 在最後一頁；成功觸發 `mark_encoded` | ✓ |
| **P3** | Tab 5 Qdrant 寫入（連線測試、collection 管理、本檔 ingest 預覽、批次 upsert、smoke query 驗證）| ✓ **E2E 已跑通** |
| **P5** | Tab 6 檢索測試（dense / sparse / hybrid 三欄並排 + project_name multiselect filter + 結果跳回原檔頁）| ✓ **使用者已測試通過** |

## 未完成

| 階段 | 內容 |
|---|---|
| **P4** | 批次處理：掃 process_tracker.json 內所有 sidecar，挑非 ingested 的批次處理 |
| **P6** | 多模態圖片（分開的 `putrue_rag_images_v1` collection，CLIP 或 jina-clip-v2）|
| **P7** | 版本管理 + 觀測儀表板（eval set + recall@K 數字化評估）|

## 本次 session 順手做的優化（除了 P3/P5）

- **PDF 預覽 cache**：`render_pdf_page_png` 包 `@st.cache_data(max_entries=256)`，key 含 `mtime`。原本每次 streamlit rerun 都重 open PDF + render，後段 CAD 頁尤其卡；現在第一次開頁付一次成本，後續編輯/切 tab/打字都瞬間回。
- **捨棄編輯狀態還原**：sidecar 加 `prior_status` + `disk_dirty` 兩欄。已入庫檔案編輯（未存檔）→ 捨棄編輯 → 自動還原 ingested。若中途按過「儲存 .md」（disk_dirty=True）則留在 processing 不謊報。`mark_encoded` / `mark_ingested` 完成時會清掉這兩個欄位。
- **sidebar selectbox 跳檔 bug**：原因是沒指定 `key`，format_func 又 closure 依賴每 rerun 重建的 `_file_status_map`，Tab 4 mark_encoded / Tab 5 mark_ingested 後 rerun 會讓 Streamlit 對 widget 識別失去穩定性。修法：加 `key="sidebar_file_selectbox"`。Tab 4/5 都覆蓋到。

## Qdrant 環境

- Docker Desktop 29.4.3（已更新），daemon 開機後要手動把 Docker Desktop GUI 打開
- Container：`qdrant`（`--restart unless-stopped`），port 6333 (HTTP) + 6334 (gRPC)
- Volume：`./qdrant_storage`（持久化）
- Qdrant version: 1.18.0
- Dashboard：http://localhost:6333/dashboard
- Collection：`putrue_rag_text_v1`（vectors: `text_dense` cosine 1024d + sparse `text_sparse`，11 個 payload index）

## 開機後第一件事（下次 session）

1. **啟動 Docker Desktop**（GUI 開啟讓 daemon 跑起來；container 因 `--restart unless-stopped` 會自動拉回）
2. **`streamlit run md_review_ui.py`** via `./venv312/Scripts/streamlit.exe`
3. 接著走 P4（批次）或 P6（圖片多模態）或 P7（eval set）—— 看當下優先級

## 踩過的坑（避免再踩）

- **`np.save` 副檔名自動補 `.npy`**：若 path 不以 `.npy` 結尾，numpy 會 append。tmp 檔命名 `xxx.npy.tmp` 寫入後實際變成 `xxx.npy.tmp.npy`，後續 rename 找不到原 tmp 檔。**修法**：用 `with open(...) as f: np.save(f, arr)` 餵 file object 繞過。已修在 `save_vectors`。
- **streamlit-ace 首次掛載回 None**：Tab 3 / Tab 4 / Tab 5 三處讀 `st.session_state[editor_key]` 都要 `isinstance(x, str)` 守門。
- **chip system 點擊插入 ace**：streamlit-ace 沒有「在 cursor 位置插入」的 API，只能 bump_widget_version remount ace，cursor 跳頁首。最後撤回 chip 整套，回到純 textarea→ace 暫存區。
- **Streamlit selectbox 沒辦法給 dropdown options 上真背景色**：CSS 沒有 content-matching selector，只能用 emoji 前綴 + 圖例補強。
- **Streamlit widget 沒指定 `key` 又用 closure-based format_func**：lambda 依賴 per-rerun 變動的資料時，widget 識別會跳。Solution: 顯式 `key=`。
- **Streamlit 沒有 programmatic tab 切換 API**：Tab 6 「跳回該檔該頁」只能切換 sidebar 檔案 + 設 page_idx，使用者要手動點 Tab 1 看結果。
- **bge-m3 是 symmetric encoder**：query 跟 passage 共用同一個 encoder，不需要 E5 那種 `query: ` / `passage: ` 前綴。`encode_query` 不套 `build_text_with_prefix`。

## 依賴清單

| 套件 | 用途 | 安裝 |
|---|---|---|
| `streamlit-ace` | VSCode 級編輯器 | `pip install streamlit-ace` ✓ |
| `FlagEmbedding` | bge-m3 推理 | `pip install FlagEmbedding` ✓ |
| `qdrant-client` | Qdrant SDK | 已裝 1.17.1 |
| Docker Desktop | 跑 Qdrant container | 已裝 29.4.3 ✓ |

## 未提交的檔案

`md_review_ui.py` 整個還沒進 git（`??` untracked）。沿用其他相關產物：sidecars (`.review.json`)、vector cache (`.vectors.*`)、垃圾桶 (`_md_trash/`)、PDF cache (`_compare_cache/`)、`qdrant_storage/` 全是 generated artifacts，建議都加入 `.gitignore`。

---

<!-- ===== FILE: triage_threshold_rationale.md ===== -->
---
name: Triage Threshold — Why 200 Vector Paths
description: Domain rationale for the page-triage threshold that routes pages to PyMuPDF fast-path vs Marker slow-path
type: project
originSessionId: 325c85d0-e0f0-4176-aee2-87600c769db5
---
分流判別 = 用 PyMuPDF 的 `page.get_drawings()` 計算向量繪圖物件數，閾值 **200**。

```
if len(page.get_drawings()) > 200:
    # 走 Marker 慢路徑
```

## 為什麼是 200（CAD 領域知識）

- **簡單頁面**：純文字或含掃描圖的頁，向量路徑極少 — 通常只有底線、表格框線等少量裝飾向量
- **複雜向量頁面**：CAD 轉 PDF 的建築圖紙，**一條樑、一扇門、一個結構符號就是由數百條向量點組成**
- 超過 200 條時，PyMuPDF 的傳統文字提取會失效（抓到無數碎片），此時必須切到 Marker

## 觸發路徑

- ≤ 200：PyMuPDF 直接提文字 → 進入後處理
- \> 200：判定為 CAD 圖紙 → 交給 Marker（GPU + Surya/Nougat）做視覺重構

**Why:** 這個閾值是「成本／精度」非對稱平衡的具體交界點。它不是憑空抓的數字，是用 CAD 結構符號的向量複雜度當依據 — 一頁圖紙的向量數很容易破千，純文字頁則遠低於 200，中間有很大 buffer。

**How to apply:**
- **動到 200 這個閾值前先警告** — 改高會讓 CAD 頁漏進快路徑（產生碎片），改低會讓單純有表格的文字頁誤判進慢路徑（浪費 GPU）
- 如果使用者抱怨「某種頁面分錯邊」，先問是哪類型 PDF，再判斷該調閾值還是改判別邏輯（例如改成複合條件：向量數 + 文字密度）
- `page.get_drawings()` 是 Fitz 較底層 API，升級 PyMuPDF 版本時要回歸測試這條判別還準不準
- 不要把這個閾值換成「過 ML 模型判斷頁面類型」 — 違反非對稱原則，預掃描階段必須便宜

---

<!-- ===== FILE: hardware_environment.md ===== -->
---
name: Hardware Environment
description: GPU and OS environment for the ETL pipeline; bounds Marker batch sizing and throughput estimates
type: project
originSessionId: 325c85d0-e0f0-4176-aee2-87600c769db5
---
- **GPU：RTX 4090（24GB VRAM）**
- **OS：Windows 11 Pro**
- **Python 3.12 + Venv**
- **CUDA via PyTorch**

**Why:** 1TB 級資料量 + Marker 走 GPU，VRAM 容量直接決定 batch size 與單頁吞吐。4090 是單卡上限級的消費級 GPU。

**How to apply:**
- 估算 Marker 跑完 1TB 的時間時，以單卡 4090 為基準（沒有多卡 / 沒有叢集）
- Batch size 建議考慮 24GB VRAM 上限，但要為 Surya layout + Nougat 推論留 buffer
- 不要建議需要 A100 / H100 / 80GB 才合理的方案
- Windows 環境 + 單機 → CUDA 驅動、torch 版本、opencv 衝突都要在 venv 內處理（沒有 Linux 容器化退路）
- 若日後吞吐不夠，方向是「優化分流（少送 Marker 一點頁）」或「升級到雲端多卡」，而非調 batch

---

<!-- ===== FILE: user_role.md ===== -->
---
name: User Role — Solo Builder
description: User builds this project alone with no team; affects communication style and the kinds of suggestions that are useful
type: user
originSessionId: 325c85d0-e0f0-4176-aee2-87600c769db5
---
使用者是**獨立開發者（solo build）**，整個專案沒有團隊。

## 影響協作方式

- **沒有團隊規矩** — 不需要套用「團隊慣例」、code review SOP、PR template 之類的建議
- 決策權在使用者一人 → 不需要說「先和團隊討論」、「等 review」
- 可以直接給技術判斷和強烈建議，不需要過度中性化
- 引入新依賴 / 改架構的決定也由他單獨拍板，溝通可以直接

## 仍要注意

- 「自己 build」不代表沒經驗 — 從技術選型（Marker + Surya + Nougat、非對稱 ETL、Multiprocessing 保護機制）看得出有實戰判斷力
- 仍要維持解釋深度（不是新手），但可以省略基礎概念鋪陳
- 不需要過度補充「團隊維護性」「他人接手」這類顧慮，除非他主動問

---

<!-- ===== FILE: feedback_strict_structure_anchors.md ===== -->
---
name: feedback-strict-structure-anchors
description: "對「結構性 regex（如 `## 第 N 頁` 頁錨點）」優先選擇嚴守原樣 + 靠 UI 防呆，而非放寬 regex 容錯"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4149cc23-1d39-444d-ac42-973be45ed406
---

碰到「user 不小心改到結構性 anchor 行（如頁標頭）導致下游 parse 失準」時，**優先用 UI 層防呆**（隱藏該行不可編輯、上方另開唯讀資訊條、加 hint 提示安全的編輯起點），**而不是把 regex 放寬成 `\s*.*\n` 容錯**。

**Why**：使用者在 [[etl_pipeline_design]] 把 `## 第 N 頁` 當 Citation 錨點是整套 RAG 來源追溯的命脈，明確說「先保留這個模式，我喜歡這樣」。嚴格 regex 換來的是「壞掉就直接壞」的快速失敗訊號 — 比起「regex 容錯但語意悄悄漂移」可預測得多。

**How to apply**：
- 提議改 regex 容錯前先停一下 — 預設答案是「不放寬」。
- 改 UI 層即可：例如 [[implementation_progress]] 裡 Tab 1 編輯器把 `## 第 N 頁` 行藏起來、改在上方放唯讀頁碼條 + 「請從 `###` 開始標」hint。
- 同理推廣：其他 Citation/錨點性 regex（heading hierarchy、image ref 等）也走同一邏輯。

---

<!-- ===== FILE: feedback_data_dependency_audit_first.md ===== -->
---
name: feedback-data-dependency-audit-first
description: 對「會寫/刪/覆寫 disk 檔」的程式（自建 OR 既有），必須主動稽核下游依賴 + 預設安全 + 警告破壞性工具；不可只當旁觀者
metadata:
  type: feedback
---

# 任何會寫/刪/覆寫 disk 檔的程式，都必須主動做資料依賴稽核

## 規則

當任務涉及**任何**會寫入、覆寫、刪除磁碟檔案的程式時（不論是我要新建的、還是 repo 內既存的、還是使用者要執行的），必須**主動**完成以下三件事：

### 1. 遇到既存程式 → 立刻稽核 + 標記
碰到 repo 內任何「會寫/刪/覆寫 disk 檔」的程式時（透過 grep、read、user 提及），**不可只把它當資訊**，必須：
- 讀它的副作用範圍（會碰哪些檔/路徑）
- 對照 [[implementation-progress-md-review-ui-py-qdrant-pipeline]] 看那些檔是否有 sidecar / 標記系統 / 被下游依賴
- 若有風險，**主動向 user 說明**：「這檔目前的設計會破壞 X，建議改成 Y / 退役 / 加 safeguard」
- **不能只是「我注意到了」然後跳過**

### 2. 要新建程式 → 預設安全是強制前提
寫**任何**會 disk-write 的程式時：
- **預設行為必須是「不破壞既有資料」**（如 sidecar-aware skip / 寫前備份 / 寫前 user 確認）
- 破壞性操作要走明確 opt-in flag（如 `--allow-overwrite-labeled`），不可預設覆寫加 `--yes` 跳過確認
- safe-by-default 在程式碼層面強制，不是寫在 docstring 跟使用者「自己小心」
- 若使用者選的改動清單裡沒含資料安全項，**主動提出「這項必須加為前提」**，不要默默做完其他項

### 3. 提示使用者執行任何程式時 → 必須先 audit
使用者問「怎麼跑 X」時，回答前必須先看 X 會碰哪些 disk 檔、有沒有破壞既有標記的風險。
不可以直接給命令叫使用者執行，未經 audit 的 `python xxx.py` 就是危險指令。

---

## Why（為何訂這條，2026-05-21 雙重事件）

### 事件 A：既存工具沒主動標記
Repo 內有 `reset_pptx_for_rerun.py`：把 process_tracker.json 內 PPT 條目移除 + **刪掉對應 .md 檔**，「讓 v4 重新處理」。

調查圖檔遺失時我**讀過這檔的內容、貼出 docstring 給 user 看**，清楚知道它刪 MD 的行為。但我只把它當「可能元兇」分析（確認今天沒跑），就放過了。

**沒做**的事：
- 沒指出「這檔的設計違反資料依賴原則，應退役或加 sidecar-aware skip」
- 沒提醒「未來不要再跑它，否則會重演今天的事」
- 沒提議改寫成安全版本

### 事件 B：自建程式重蹈覆轍
User 選 C/D/F 三項升級（圖檔命名統一 / progress reporter / strategy 重構），**沒選 A**（sidecar-aware skip）。我**默默做完 C/D/F**，沒主動指出「無論你選什麼，A 必須是前提」。

結果建出 `mkd_generator_v5/` + `run_v5.py`，預設行為跟事件 A 的 `reset_pptx_for_rerun.py` 一樣 — **會覆寫已標記 MD**。launcher 還加了 `--yes` 跳過確認、`--no-skip` 強制重跑。

事件 A 跟事件 B 的根因相同：**沒把資料依賴當設計約束**。

我**完全知道**這些事實：
- `md_review_ui.py` 是 review/labeling 系統
- sidecar JSON 記錄 status / custom_labels / delete_history
- embedding 流程從 MD 算 chunks
- Qdrant ingestion 用 chunks 寫入向量庫
- 今早 `git reset --hard HEAD` 才剛把 user 5/12 之後的編輯全砍光

但**沒把這些事實連起來做出推論**：「MD 是整條 RAG pipeline 的單一資料源，任何會寫 MD 的程式（既存的、自建的、user 要執行的）都必須先保證不會破壞已標記檔」。

User 反饋原文：「**這是不可接受的疏失**」。對。

---

## How to apply

### 觸發時機（出現任一即觸發）：
- 我要寫新程式，副作用會碰 disk（write/delete/overwrite/rename/move）
- 我在 grep / read 看到既有程式做這些事
- User 提到要執行某個程式 / notebook / cell
- User 描述 disk 內容異常（檔案少了、被改了、status 不對）

### 觸發後，必須完成這個 checklist 並 **write down 給 user 看**：

```
- [ ] 這程式會 write/delete/overwrite 哪些 disk 檔？路徑模式是什麼？
- [ ] 那些檔現在 disk 上有沒有？是 user 加工過的還是程式自動產出的？
- [ ] 有沒有 sidecar / metadata 標記哪些檔被 user 加工過（review_status、labels、delete_history）？
- [ ] 哪些下游程式/資料庫吃這些檔案？被破壞會影響什麼？
      （本 codebase 預設鏈：MD → chunks → embedding → Qdrant points → 檢索 → LLM RAG）
- [ ] default 行為是「不動現有檔」還是「覆寫現有檔」？必須是前者。
- [ ] 破壞性操作（覆寫、刪除）需要哪個 flag？預設關閉。
- [ ] 若是既存程式且不安全：建議退役/quarantine/重寫成安全版本。
- [ ] 若 user 選的功能清單裡沒含資料安全項：主動提出「這必須加為前提」。
```

### 已知本 codebase 的高敏感檔案模式：
- `mkdata/*.md` — review/labeling 的核心，**有對應 sidecar 標 review_status 的不可覆寫**
- `mkdata/*.review.json` — sidecar 本體，記錄 status / labels / delete_history
- `mkdata/*_image/*.{png,jpeg}` — MD 引用的圖檔；刪除前先檢查 MD 還有沒有引用
- `mkdata/*.vectors.{dense.npy,sparse.json,manifest.json}` — embedding 快取
- `mkdata/process_tracker.json` — ETL pipeline tracker；改動會影響 ETL skip 邏輯
- `qdrant_storage/` — Qdrant 持久化，**絕對不能直接動**

### 已知本 codebase 的危險既存程式（看到要警告，不可放過）：
- `reset_pptx_for_rerun.py` — 刪 PPT 對應 MD + 清 tracker。**已造成 2026-05-21 災難，建議退役或加 sidecar-aware skip**

### 設計優先順序對照表：

| 任務類型 | 設計優先順序 |
|---|---|
| 純程式重構（無 disk 寫入）| API 等價 → 效能 → 可讀性 |
| 純讀取 / 分析 | 正確性 → 效能 → 可讀性 |
| **會 disk-write 的程式** | **不破壞既有資料 → 正確性 → 其他** |
| 會寫資料庫的程式 | 不污染既有資料 → 冪等性 → 其他 |

對 [[user-role]] 這種獨立開發者環境：沒 code review、沒 staging、disk 就是 production、出事沒救援。**每個我經手的程式都默認是「會在 production 跑」**，必須以最高安全標準設計或警告。

---

## 相關記憶

- [[implementation-progress-md-review-ui-py-qdrant-pipeline]] — 標記系統 + sidecar schema + 下游 RAG pipeline 依賴鏈
- [[user-role]] — 獨立開發者環境風險
- [[feedback-strict-structure-anchors]] — 結構性資料優先嚴守原則（同精神不同面向）

---

<!-- ===== FILE: feedback_tool_scope_isolation.md ===== -->
---
name: feedback-tool-scope-isolation
description: md_review_ui（標記工具）跟 mkd_generator（ETL 工具）絕對不可在同一個 task / session 同時變動。每個任務開始必須先標明範圍
metadata:
  type: feedback
---

# md_review_ui ↔ mkd_generator 工具範圍隔離

## 規則

本 codebase 有兩個職責截然不同的核心工具，**絕對不可**在同一個任務 / 同一個 session 同時變動：

| 工具 | 職責 | Lifecycle 角色 |
|---|---|---|
| `md_review_ui.py` | **資料標記主要工具** | **下游 / consumer**：讀 MD + 加 user-curated metadata（labels / delete_history / review_status）→ embedding → Qdrant |
| `mkd_generator_v*` | **資料正規格式化工具**（ETL）| **上游 / producer**：原始 PDF/PPT → 標準化 MD + 圖庫 |

任務開始時，**必須**先在對話內 write down「這個 task 屬於哪個工具」。發現範圍跨越時，**停下來**跟 user 確認是否該拆兩個 session，不可默默動兩邊。

## Why

兩工具混改的具體危害：

1. **責任歸屬不清** — 若標記資料異常，無法判斷是 md_review_ui 的 bug 還是 mkd_generator 改變 MD 格式造成的
2. **資料依賴破壞** — mkd_generator 改變 MD 格式 → md_review_ui 的 parse_md / regex / sidecar schema 假設失效
3. **review 跟 ETL 同時動** — user 無法測試任一邊，因為兩邊都在不穩定狀態
4. **commit 邊界混亂** — 一個 commit 含兩個 tool 的改動 → 不能單獨 revert / cherry-pick / bisect
5. **跨工具測試成本高** — 需要 staging 兩條 pipeline 才能驗證，獨立開發者環境 ([[user-role]]) 沒這資源

2026-05-21 session 我就違反了這條：
- 開頭做 `md_review_ui.py` 的 W4/W1/W2 效能優化
- 中間 user 提「開發新的 v4 版本」，我**直接跳去建 `mkd_generator_v5/`**
- 沒有明確「我們現在切換工具」的 handoff
- 結果 v5 出問題後，難以追蹤是 v5 設計問題還是 md_review_ui 那邊 W2 GC / shared chunks 改動引起的副作用

## How to apply

### 觸發時機（任務開始時）
- 我要動程式碼前 → 先標明屬於哪個工具
- User 描述任務時 → 我要 echo back「這是 md_review_ui 的事？還是 mkd_generator 的事？」
- 任何 grep / read / write 動作前 → 先確認檔案屬於哪個工具

### 工具歸屬判定
- 路徑含 `md_review_ui.py` / `llm_chat.py` → md_review_ui 工具範圍
- 路徑含 `mkd_generator_v*.py` / `mkd_generator_v*/` / `pdf_image_extractor*.py` / `ppt_to_pdf*.py` / `extract_*.py` / `reset_pptx_*` / `simulate_v4_*` / `pipeline.ipynb` → mkd_generator 工具範圍
- `mkdata/*` 是兩者交界的**資料**（不是工具）：mkd_generator 寫入，md_review_ui 讀取 + 加 sidecar

### 跨範圍情況的處理
- 若 user 明確要在同 session 動兩邊（例如「v5 要保護 md_review_ui 的 sidecar」），可以做，但**必須**：
  - 把任務拆成「先改 A，commit，再改 B，commit」
  - 兩者間的 interface（如 sidecar schema）改動需要單獨討論
  - 不可以同時 in-flight 改兩邊的內部實作
- 若 user 沒明說但任務看起來會擴散到兩邊，**停下來確認**：「這部分屬於 X 工具範圍，要不要先收尾 Y 再開始 X？」

### Commit 邊界
- 一個 commit 內**不可**同時包含兩個工具的改動
- 例外：跨工具的 schema 同步（如新增一個 sidecar 欄位兩邊都要識別），但需在 commit message 明寫「cross-tool schema sync: ...」並列出兩邊改動清單

## 相關記憶

- [[user-role]] — 獨立開發者環境，無 staging 不能跨工具同時動
- [[feedback-data-dependency-audit-first]] — 資料安全是另一條獨立規則，跟工具隔離互補（資料安全＝寫前稽核，工具隔離＝改前明界）
- [[implementation-progress-md-review-ui-py-qdrant-pipeline]] — md_review_ui 的範圍 + 跟 mkd_generator 的交界
- [[etl-pipeline-design]] — mkd_generator 的範圍 + 跟 md_review_ui 的交界
