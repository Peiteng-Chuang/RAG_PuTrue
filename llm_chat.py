"""LLM 回答模組：system prompt + session 記憶（+ 可選 RAG context）

設計原則：
- system prompt 用 role:system（不要塞在 user content）
- 對話歷史用 messages 列表累積（不要 string concat）
- 滑動視窗管 token，預設保留最近 5 輪
- RAG context（chunks）只放在「當輪」user message，不寫進 history（避免記憶爆炸）
"""
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


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """把檢索結果格式化成模型友善的 context 區塊。

    支援兩種 chunk 結構：
    A. qdrant格式.md v2 規格（md_review_ui.py 內部用）：
       {"score": float, "payload": {"metadata": {...}, "content": {...}}}
    B. 簡易結構（pipeline.ipynb 舊版相容）：
       {"score": float, "content": str, "page_name": str}
    """
    if not chunks:
        return "（本輪未檢索到相關資料）"
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
