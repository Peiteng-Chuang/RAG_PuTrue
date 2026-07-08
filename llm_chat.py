"""LLM 回答模組：system prompt + session 記憶（+ 可選 RAG context）

設計原則：
- system prompt 用 role:system（不要塞在 user content）
- 對話歷史用 messages 列表累積（不要 string concat）
- 滑動視窗管 token，預設保留最近 5 輪
- RAG context（chunks）只放在「當輪」user message，不寫進 history（避免記憶爆炸）
"""
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = "你是一個專業的助理。請用繁體中文回答，仔細地思考後回答，語氣專業簡潔。"

RAG_SYSTEM_PROMPT = """你是璞真建設的專業文檔助手，根據檢索到的工程簡報資料回答問題。

【回答規則】
1. 優先根據「參考資料」回答；資料中沒有的請明確說「資料中未提及」，不憑空推測。
2. 引用必須標明來源（檔名、頁碼、章節），例如：「根據《本因坊結案報告》第13頁《2、拆屋前現況照片》…」。
3. 涉及數字、面積、樓層、金額時，列出依據與計算過程。
4. 若用戶追問先前提過的內容，可結合「對話歷史」回答，但事實依據仍以「參考資料」為準。
5. 用繁體中文回答，語氣專業簡潔。"""


HYDE_SYSTEM_PROMPT = """你是璞真建設的工程簡報撰寫助手。使用者會給你一個問題，請你扮演「這份簡報資料的撰寫者」，寫一段**像是從簡報資料中節錄出來**的文字片段，假設這份資料能完整回答這個問題。

【撰寫規則】
1. 直接寫出答案內容片段，**不要加任何「假設」、「假如」、「可能」之類的不確定語氣**，也不要說「以下是回答」之類的引導語。
2. 主動帶入建築工程領域的同義詞與相關術語，讓檢索向量能涵蓋多種詞彙。常見對映：
   - 結構 ↔ 構架 ↔ RC 結構 ↔ 鋼骨結構 ↔ 主結構 ↔ 結構體
   - 機電 ↔ 水電 ↔ 弱電 ↔ 強電 ↔ 機水電
   - 建照 ↔ 施工執照 ↔ 使用執照 ↔ 建築執照
   - 銷講 ↔ 銷售簡報 ↔ 公設介紹 ↔ 個案介紹
3. 模仿簡報資料的口吻：條列、編號、附帶具體數字／面積／樓層／案名／廠商名。若不確定具體值，使用合理的範例值（標註「約」、「例如」即可）。
4. 用繁體中文，3-6 句，控制在 200 字內。

這段文字**不會直接給使用者看**，純粹用來把向量推進到正確的語意鄰域，提升 dense retrieval 對使用者口語詞彙的回收率。"""


def generate_hyde(
    client: OpenAI,
    model_name: str,
    query: str,
    system_prompt: str = HYDE_SYSTEM_PROMPT,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """HyDE: 用 LLM 把使用者 query 改寫成「像是從文件節錄的段落」，回傳該段落字串。

    後續用該段落（而非原 query）去做 dense embedding，可有效縮短使用者口語 ↔ 文件用詞
    的詞彙鴻溝（例如使用者問「結構」、文件寫「構架」）。

    本函式不寫入任何 history、不修改 caller 的 ChatBot 狀態，是純 stateless 的一次性呼叫。
    """
    opts = options or {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **opts,
    )
    return (response.choices[0].message.content or "").strip()


# ============================================================================
# 意圖路由（三路：fact / compare / rank）—— 第 1 層
# 見 memory: rag_comparison_routing_design
# ============================================================================

ROUTER_SYSTEM_PROMPT = """你是璞真建設文檔問答系統的「查詢意圖分析器」。使用者會問一個關於工程簡報／建案資料的問題，你要輸出一段 JSON，供後端決定檢索策略。

【intent 三選一】
- "fact"：針對單一主題／單一建案的事實查詢（是什麼、在哪、多少、如何、某案的某項內容）。這是預設值，拿不準時選它。
- "compare"：要「跨多個建案／實體」做質性比較、對照、找異同（哪個比較好、A 和 B 差在哪、各案的做法差異）。
- "rank"：要「數值排序、極值、聚合」（最大／最高／最多／最貴／前三名／總共幾個／平均是多少）。這類題需要完整且正確的數值序，語意相似度檢索答不了。

【entity_mentions】使用者「明確點名或明顯指涉的建案／實體名稱」字串陣列。使用者只說「這幾個案子」「各個建案」但沒點名時，回空陣列。**不要杜撰不存在的名字。**

【rewrites】拿去做向量檢索的改寫查詢，模仿工程簡報的用詞（主動帶入建築領域同義詞：結構↔構架↔RC、機電↔水電、建照↔施工執照 等）：
- intent=fact：輸出 1 段，涵蓋問題核心。
- intent=compare：**每個 entity_mention 各一段，順序與 entity_mentions 對齊**；若無具名實體，輸出 1-3 段代表不同比較面向。
- intent=rank：輸出 1 段（僅供參考檢索，不保證用得上）。
每段 40-120 字，直述句、帶具體名詞，不要「假設」「可能」之類不確定語氣。

【attributes】比較／排序涉及的屬性關鍵詞（如「面積」「樓層」「總戶數」「單價」「公設比」），字串陣列；沒有就空陣列。

【輸出規則】**只輸出 JSON 本體**，不要 markdown code fence、不要任何說明或思考文字。嚴格格式：
{"intent": "fact|compare|rank", "entity_mentions": [], "rewrites": [], "attributes": []}"""


COMPARE_SYSTEM_PROMPT = """你是璞真建設的專業文檔助手，正在處理一個「跨建案比較」問題。檢索結果已**依建案分組**放在「參考資料」中（每組標「===== 建案：XXX =====」）。

【回答規則】
1. 逐一針對每個建案作答，再做橫向比較；適合時用表格或分段清楚對照。
2. **只能根據「參考資料」中實際出現的內容比較。** 某建案在資料中缺少某屬性時，明確標「資料中未提及」，**不要拿其他建案的值去推測、也不要猜。**
3. 「參考資料」只涵蓋被檢索到的建案。務必在開頭提醒「以下僅涵蓋檢索到的 N 個建案：…」，不要讓使用者誤以為已窮舉全部建案。
4. 引用必須標明來源（檔名、頁碼、章節）。
5. 用繁體中文，語氣專業簡潔。"""


RANK_SYSTEM_PROMPT = """你是璞真建設的專業文檔助手。使用者問的是「排序／極值／數值聚合」類問題（例如最大、最高、最貴、前幾名、總數、平均）。

【重要限制 — 務必遵守】
本系統負責精確排序的「結構化資料表（第 2 層）」**尚未接入**。語意檢索只能撈到相似片段，**無法保證涵蓋所有建案、也無法保證數值大小正確**。因此：
1. **開頭先明確告知使用者：排序／極值查詢功能尚未接入，以下內容未經完整排序，僅供參考。**
2. **絕對不要**根據「參考資料」的片段自行排出名次、宣稱「最大／最高／第一」，也不要把片段裡的數字加總或平均後當成結論。
3. 你可以「不帶排名地」列出檢索到的相關建案及其在資料中出現的相關數值，逐筆標明來源（檔名、頁碼）。
4. 用繁體中文，語氣專業誠實；寧可說「無法完整排序」，也不要為了給答案而編造完整性。"""


@dataclass
class QueryPlan:
    """analyze_query 的結構化輸出。intent ∈ {fact, compare, rank}。"""
    intent: str = "fact"
    entity_mentions: List[str] = field(default_factory=list)
    rewrites: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """從 LLM 回應中抽出第一個 JSON 物件。

    容錯：剝除 qwen3 等模型的 <think>…</think> 思考區塊、markdown code fence，
    再抓第一個 `{` 到最後一個 `}`。解析失敗回 None（交由 caller fallback）。
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001 — 解析失敗屬預期，回 None fallback
        return None


def analyze_query(
    client: OpenAI,
    model_name: str,
    query: str,
    system_prompt: str = ROUTER_SYSTEM_PROMPT,
    options: Optional[Dict[str, Any]] = None,
) -> QueryPlan:
    """單一 LLM 呼叫做意圖分析，回 QueryPlan。

    純 stateless，不寫 history。JSON 解析失敗會 raise ValueError，由 caller 決定 fallback
    （設計上：router 失敗 → 退回現行單發 fact 檢索，見 rag_comparison_routing_design）。

    注意：max_tokens 給得寬（含 qwen3 思考預算）；真正的 JSON 很短，思考區塊會被剝除。
    """
    opts = options or {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **opts,
    )
    raw = (response.choices[0].message.content or "").strip()
    data = _extract_json_object(raw)
    if data is None:
        raise ValueError(f"router 回傳無法解析為 JSON：{raw[:200]!r}")
    intent = str(data.get("intent", "fact")).strip().lower()
    if intent not in ("fact", "compare", "rank"):
        intent = "fact"
    plan = QueryPlan(
        intent=intent,
        entity_mentions=[str(x).strip() for x in (data.get("entity_mentions") or []) if str(x).strip()],
        rewrites=[str(x).strip() for x in (data.get("rewrites") or []) if str(x).strip()],
        attributes=[str(x).strip() for x in (data.get("attributes") or []) if str(x).strip()],
        raw=data,
    )
    if not plan.rewrites:  # 模型沒給改寫 → 至少用原 query，保證 fact 路徑有東西可 encode
        plan.rewrites = [query]
    return plan


TITLE_SYSTEM_PROMPT = """你是對話標題產生器。使用者會給你他在一場對話中問的「第一個問題」，請你擷取其**核心重點**，產生一個精煉的繁體中文短標題。

【規則】
1. 6 ～ 15 個字，越精準越好；抓住主題名詞（案名、工項、文件類型等），去掉「請問」「我想知道」之類的贅語。
2. 只輸出標題本身，**不要**加引號、句號、冒號、編號或任何說明文字。
3. 不要輸出「標題：」之類的前綴，也不要換行。
4. 若問題本身就很短，直接精煉它即可。"""


def generate_title(
    client: OpenAI,
    model_name: str,
    first_question: str,
    system_prompt: str = TITLE_SYSTEM_PROMPT,
    options: Optional[Dict[str, Any]] = None,
    max_chars: int = 20,
) -> str:
    """用 LLM 把使用者「第一個問題」濃縮成一個對話室標題（繁中、6-15 字）。

    純 stateless 的一次性呼叫，不寫入任何 history。回傳已清洗（去引號／換行／前綴、
    截長）的標題字串；若 LLM 回空字串，回傳空字串交由 caller 決定 fallback。
    """
    opts = options or {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 48,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_question},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **opts,
    )
    raw = (response.choices[0].message.content or "").strip()
    # 清洗：取首行、去常見前綴與包裹引號
    title = raw.splitlines()[0].strip() if raw else ""
    for prefix in ("標題：", "標題:", "Title:", "title:"):
        if title.startswith(prefix):
            title = title[len(prefix):].strip()
    title = title.strip().strip("「」『』\"'《》").strip()
    if len(title) > max_chars:
        title = title[:max_chars].rstrip() + "…"
    return title


def format_chunks(
    chunks: List[Dict[str, Any]],
    group_by_entity: bool = False,
) -> str:
    """把檢索結果格式化成模型友善的 context 區塊。

    支援兩種 chunk 結構：
    A. qdrant格式.md v2 規格（md_review_ui.py 內部用）：
       {"score": float, "payload": {"metadata": {...}, "content": {...}}}
    B. 簡易結構（pipeline.ipynb 舊版相容）：
       {"score": float, "content": str, "page_name": str}

    group_by_entity=True（compare/rank fan-out 用）：依 chunk 的 `_entity` 欄位分組，
    每組冠上「===== 建案：XXX =====」標頭，讓 LLM 能逐建案比較。
    """
    if not chunks:
        return "（本輪未檢索到相關資料）"
    if group_by_entity:
        return _format_chunks_grouped(chunks)
    parts = []
    for i, c in enumerate(chunks, 1):
        score = c.get("score", 0.0)

        # 結構 A：qdrant 規格
        payload = c.get("payload")
        if isinstance(payload, dict):
            meta = payload.get("metadata", {}) or {}
            source = meta.get("source", {}) or {}
            location = meta.get("location", {}) or {}
            content_obj = payload.get("content", {}) or {}

            file_name = source.get("file_name") or source.get("file_key", "未知檔案")
            page = location.get("page", "?")
            breadcrumb = location.get("breadcrumb") or location.get("headings_flat") or []
            heading_str = " > ".join(breadcrumb) if breadcrumb else "（無章節）"
            text = (
                content_obj.get("text")
                or content_obj.get("md_content")
                or ""
            )
            header = f"[資料 {i} | 相關度 {score:.3f}] {file_name} / 第{page}頁 / {heading_str}"
            parts.append(f"{header}\n{text}")
            continue

        # 結構 B：舊版
        page_name = c.get("page_name", "未知來源")
        text = c.get("content", "")
        parts.append(f"[資料 {i} | 來源 {page_name} | 相關度 {score:.3f}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _format_chunks_grouped(chunks: List[Dict[str, Any]]) -> str:
    """依 chunk 的 `_entity` 欄位分組格式化（保留輸入順序 = fan-out 建案順序）。"""
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for c in chunks:
        ent = c.get("_entity") or "（未分組）"
        groups.setdefault(ent, []).append(c)
    blocks = []
    for ent, cs in groups.items():
        blocks.append(f"===== 建案：{ent} =====\n{format_chunks(cs)}")
    return "\n\n".join(blocks)


class ChatBot:
    """system prompt + session 記憶的對話機器人，可選擇性帶入 RAG context。

    - system_prompt：固定身份與規則（role:system）
    - history：session 內的多輪對話記憶（role:user / role:assistant），只存原始 query
    - chat(query, rag_chunks=...)：rag_chunks 給定時，當輪 user message 會包進 context；
      history 仍只存原始 query（避免 context 隨對話累積爆 token）
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 5,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.model = model_name
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
        self.options = options or {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
        }

    def _build_messages(
        self,
        query: str,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        recent = self.history[-self.max_turns * 2 :]
        messages.extend(recent)
        if rag_chunks:
            context_text = format_chunks(rag_chunks)
            user_msg = f"【參考資料】\n{context_text}\n\n【問題】\n{query}"
        else:
            user_msg = query
        messages.append({"role": "user", "content": user_msg})
        return messages

    def chat(
        self,
        query: str,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """單輪對話。傳入 rag_chunks 會把檢索結果注入當輪 user message。

        注意：history 只存原始 query，不存 RAG context；多輪對話的 context 由呼叫端
        每次重新檢索注入。
        """
        messages = self._build_messages(query, rag_chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.options,
        )
        answer = response.choices[0].message.content or ""
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self) -> None:
        """清空對話歷史（system prompt 保留）。"""
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def set_system_prompt(self, prompt: str) -> None:
        """執行期更換 system prompt（測試不同 prompt 用）。同時清空歷史。"""
        self.system_prompt = prompt
        self.history.clear()
