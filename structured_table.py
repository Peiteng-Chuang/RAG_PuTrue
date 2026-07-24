"""第 2 層「結構化屬性表」：長格式 CSV 的載入 + 查詢（downstream-only，不進 ETL）。

設計見 rag_comparison_routing_design：人工維護、隨時可改、熱載入。

**主格式＝矩陣（建案 × 屬性）**：一列一建案、每欄一屬性，好填、加屬性＝加一欄（零改碼）。
    建案key,   地上樓層, 基地面積_m2, 總戶數, 公設比_pct, 帶車位, 特殊工法, 來源檔, 來源頁
    勤美之真,  29,       1200,        120,    33,         是,     免模工法,  xxx.pptx, 12
- 第一欄（或名為「建案key」的欄）＝建案 key，需 == Qdrant payload `metadata.source.project_name`。
- 保留欄（逐列共用、非屬性）：來源檔 / 來源頁 / 來源 / 備註。其餘每欄都是一個屬性。
- 單位建議寫進欄名（`基地面積_m2`、`公設比_pct`）；數值型填純數字（"地上29層"/"約33%" 也能抽出數字排序）。
- 空白格 = 未收錄（排序/聚合時排除並回報，**不可當 0，也不可 fall through 回向量猜**）。

**自動相容舊長格式**：表頭同時含「屬性」「值」欄時，改走一列一 (建案,屬性) 的長格式解析。
內部一律熔成 `TableRow` 長表 → `lookup()` 與下游完全不變。

本模組**純函式、不依賴 streamlit**（方便單元測試）；熱載入的 mtime 快取由呼叫端（UI）包。
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Excel 在繁中環境「另存 CSV」預設 cp950/big5，故 UTF-8 之外也試 cp950/big5
_ENCODINGS = ["utf-8-sig", "utf-8", "cp950", "big5"]


@dataclass
class TableRow:
    project: str
    attr: str
    value: str
    number: Optional[float]
    unit: str
    src_file: str
    src_page: str


@dataclass
class TableData:
    rows: list[TableRow] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)   # distinct 屬性（保序）
    projects: list[str] = field(default_factory=list)      # distinct 建案key（保序）
    encoding: str = ""
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.rows)


def _norm(s: str) -> str:
    """比對用正規化：去所有空白 + 小寫。"""
    return re.sub(r"\s+", "", (s or "")).lower()


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_number(s: str) -> Optional[float]:
    """轉 float 供排序。先試整格純數字；否則**只有當整格幾乎就是這個數字**
    （扣掉數字後的殘料 ≤4 字，即單位/約/地上 之類）才採計。

    這條界線把「29層 / 約33% / 地下5層」（可排序）跟「採用順打工法配置3層水平鋼支撐」
    （敘述性，殘料很長 → 不採，數值=None）分開，避免從長句硬抽出假數字污染排序。
    空／無數字／敘述句 → None（= 該屬性非數值或未收錄，排序時排除）。"""
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = _NUM_RE.search(s)
    if not m:
        return None
    remainder = s[:m.start()] + s[m.end():]   # 數字以外的殘料
    return float(m.group()) if len(remainder) <= 4 else None


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def _sniff_delimiter(text: str) -> str:
    """看表頭那行：tab 數 > 逗號數 → tab，否則逗號。
    繁中 Excel「另存 CSV」有時會存成 tab 分隔（或使用者存成 Unicode 文字），
    自動偵測可避免 tab 檔被逗號 reader 讀成單一欄 → 靜默空表。"""
    first = text.splitlines()[0] if text else ""
    return "\t" if first.count("\t") > first.count(",") else ","


def load_table(path) -> TableData:
    """讀 CSV → TableData。編碼自動 fallback；任何硬錯誤都收進 `error`（呼叫端據此走 last-good）。"""
    p = Path(path)
    if not p.exists():
        return TableData(error=f"資料表不存在：{p}")
    try:
        raw = p.read_bytes()
    except Exception as e:  # noqa: BLE001
        return TableData(error=f"讀取失敗：{type(e).__name__}: {e}")

    text, enc = _decode(raw)
    delim = _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if not reader.fieldnames:
        return TableData(error="CSV 無表頭", encoding=enc)
    header = [(h or "").strip() for h in reader.fieldnames]

    td = TableData(encoding=enc)
    if enc not in ("utf-8-sig", "utf-8"):
        td.warnings.append(f"CSV 以 {enc} 解碼（非 UTF-8）；建議存成「CSV UTF-8」以免亂碼。")
    if delim == "\t":
        td.warnings.append("以 TAB 分隔讀取（檔案是分頁字元分隔）；建議 Excel 另存成「CSV 逗號分隔」較穩。")
    if len(header) == 1 and ("\t" in header[0] or "," in header[0]):
        # 分隔符判斷失敗才會發生（正常單欄表頭不含分隔符）→ 明講，別靜默空表
        td.error = "疑似分隔符不符：只解析出 1 欄且欄名含分隔字元，請確認檔案為逗號或 tab 分隔的 CSV。"
        return td

    rows = [
        {(k or "").strip(): (v.strip() if isinstance(v, str) else "") for k, v in rr.items()}
        for rr in reader
    ]
    # 格式自動偵測：長格式帶「屬性」「值」欄；否則視為矩陣（建案 × 屬性）
    if "屬性" in header and "值" in header:
        _parse_long(td, rows)
    else:
        if "建案key" not in header and not header:
            return TableData(error="CSV 無可用欄位", encoding=enc)
        _parse_matrix(td, header, rows)

    td.attributes = list(dict.fromkeys(r.attr for r in td.rows))
    td.projects = list(dict.fromkeys(r.project for r in td.rows))
    # 空表（表頭合法但無已填列）不算 error，只是 ok=False（尚未填）；error 專留給真正讀檔/格式失敗
    return td


# 矩陣格式中「非屬性」的保留欄名（其餘每欄都視為一個屬性）
_RESERVED_COLS = {"建案key", "來源檔", "來源頁", "來源", "備註", "note", "source"}


def _parse_long(td: TableData, rows: list[dict]) -> None:
    """舊長格式：一列一個 (建案, 屬性) 事實。"""
    for i, row in enumerate(rows, start=2):
        proj, attr, val = row.get("建案key", ""), row.get("屬性", ""), row.get("值", "")
        if not any((proj, attr, val)):
            continue
        if not proj or not attr:
            td.warnings.append(f"第 {i} 列缺 建案key 或 屬性 → 略過")
            continue
        if not val and not (row.get("數值", "") or "").strip():
            continue  # 佔位列（值與數值皆空）→ 尚未填
        td.rows.append(TableRow(
            project=proj, attr=attr, value=val,
            number=_to_number(row.get("數值", "") or val),
            unit=row.get("單位", ""),
            src_file=row.get("來源檔", ""), src_page=str(row.get("來源頁", "") or ""),
        ))


def _parse_matrix(td: TableData, header: list[str], rows: list[dict]) -> None:
    """矩陣格式：一列一建案、每欄一屬性 → 內部熔成長格式（單位建議寫進欄名）。

    建案key 欄＝第一欄或名為「建案key」的欄；來源檔／來源頁／備註 為保留欄（逐列共用），
    其餘每欄都當一個屬性。空白格 = 未收錄（略過）。加屬性 = 加一欄，零改碼。"""
    key_col = "建案key" if "建案key" in header else header[0]
    attr_cols = [h for h in header if h and h != key_col and h not in _RESERVED_COLS]
    for i, row in enumerate(rows, start=2):
        proj = row.get(key_col, "")
        if not proj:
            if any(row.get(c, "") for c in attr_cols):
                td.warnings.append(f"第 {i} 列缺 建案key → 略過")
            continue
        src_file = row.get("來源檔", "") or row.get("來源", "")
        src_page = str(row.get("來源頁", "") or "")
        for col in attr_cols:
            cell = row.get(col, "")
            if not cell:
                continue  # 空格 = 未收錄
            td.rows.append(TableRow(
                project=proj, attr=col, value=cell, number=_to_number(cell),
                unit="", src_file=src_file, src_page=src_page,
            ))


def _match_attributes(requested: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    """把 router 抽出的 attribute 詞對到表內實際屬性名（正規化 + 雙向子字串）。

    requested 空 → 回全部 available（沒指定就整案屬性都給）。
    回 (matched_canonical 保序去重, unmatched_requested)。"""
    if not requested:
        return list(available), []
    matched: list[str] = []
    unmatched: list[str] = []
    for r in requested:
        rn = _norm(r)
        hit = None
        for a in available:
            an = _norm(a)
            if rn and (rn == an or rn in an or an in rn):
                hit = a
                break
        if hit:
            if hit not in matched:
                matched.append(hit)
        else:
            unmatched.append(r)
    return matched, unmatched


@dataclass
class LookupResult:
    markdown: str                       # 餵給 answer LLM 的精簡表
    covered_projects: list[str]         # 表內實際涵蓋到的建案
    matched_attrs: list[str]            # 對到的屬性
    unmatched_attrs: list[str]          # router 提到但表內沒有的屬性
    notices: list[str] = field(default_factory=list)


def lookup(
    table: TableData,
    *,
    entities: Optional[list[str]] = None,
    attributes: Optional[list[str]] = None,
    restrict_projects: Optional[list[str]] = None,
) -> Optional[LookupResult]:
    """依 (建案, 屬性) 篩出子表並排序。無任何命中 → None（呼叫端據此走「未收錄」）。

    - restrict_projects：sidebar 硬約束（已是 project_name key），None = 不限。
    - entities：router 點名的建案；空 → 該約束內全部建案。
    - attributes：router 抽的屬性詞；空 → 涵蓋到的建案的全部屬性。
    """
    if not table.ok:
        return None
    rows = table.rows

    if restrict_projects:
        allow = {_norm(p) for p in restrict_projects}
        rows = [r for r in rows if _norm(r.project) in allow]

    ent_norm = [_norm(e) for e in (entities or []) if _norm(e)]
    if ent_norm:
        rows = [r for r in rows
                if any(en in _norm(r.project) or _norm(r.project) in en for en in ent_norm)]

    avail_attrs: list[str] = []
    for r in rows:
        if r.attr not in avail_attrs:
            avail_attrs.append(r.attr)
    matched_attrs, unmatched = _match_attributes(attributes or [], avail_attrs)
    value_hits: list[str] = []
    if attributes and not matched_attrs:
        # 值反查 fallback：router 給的可能是「值」(順打工法) 而非欄名(開挖工法)。
        # 掃各欄內容，命中的欄整欄拉出來（給 LLM 篩「哪些案有…」）。
        for term in attributes:
            tn = _norm(term)
            if not tn:
                continue
            for a in avail_attrs:
                if a in value_hits:
                    continue
                if any(tn in _norm(r.value) for r in rows if r.attr == a):
                    value_hits.append(a)
        if value_hits:
            matched_attrs, unmatched = value_hits, []
    # 指名了屬性，但欄名、值都沒中 → 回 None，讓上游走「未收錄、不猜」路徑（不塞其他屬性充數）
    if attributes and not matched_attrs:
        return None
    if attributes and matched_attrs:
        rows = [r for r in rows if r.attr in matched_attrs]

    if not rows:
        return None

    # 屬性分組 → 數值 desc（None 殿後）→ 建案名，方便 LLM 直接讀出排序
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.attr, -(r.number if r.number is not None else float("-inf")), r.project),
    )

    lines = ["| 建案 | 屬性 | 值 | 數值 | 單位 | 來源 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows_sorted:
        num = "" if r.number is None else (f"{r.number:g}")
        src = r.src_file + (f" p.{r.src_page}" if r.src_page else "")
        lines.append(f"| {r.project} | {r.attr} | {r.value} | {num} | {r.unit} | {src} |")
    md = "\n".join(lines)

    covered = list(dict.fromkeys(r.project for r in rows_sorted))
    notices: list[str] = []
    if value_hits:
        notices.append("以屬性值反查到欄位：" + "、".join(value_hits))
    if unmatched:
        notices.append("下列屬性資料表未收錄：" + "、".join(unmatched))
    return LookupResult(
        markdown=md, covered_projects=covered,
        matched_attrs=matched_attrs, unmatched_attrs=unmatched, notices=notices,
    )
